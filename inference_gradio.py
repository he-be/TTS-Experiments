"""
Gradio demo for HF-format T5GemmaVoice checkpoints.

Usage:
    python inference_gradio.py --model_dir ./t5gemma_voice_hf --port 7860
"""

import argparse
import os
from dotenv import load_dotenv
load_dotenv()

import re
import random
from functools import lru_cache
from types import MethodType
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import script_generator

from data.tokenizer import AudioTokenizer, tokenize_audio
from duration_estimator import estimate_duration
from inference_tts_utils import (
    get_audio_info,
    get_sample_rate,
    inference_one_sample,
    normalize_text_with_lang,
    transcribe_audio,
    unload_whisper_model,
    segment_text_by_sentences,
    concatenate_audio_segments,
)

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def adaptive_duration_scale(estimated_duration: float, scale_factor: float) -> float:
    """
    Apply adaptive scaling based on estimated duration.

    - Long durations (>= 20s): Apply scale_factor as-is
    - Short durations (<= 5s): Apply minimal scaling (close to 1.0)
    - Medium durations (5-20s): Linear interpolation

    This prevents cutting off audio for short segments while allowing
    aggressive scaling for long segments.
    """
    if scale_factor == 1.0:
        return 1.0

    if estimated_duration >= 20.0:
        # Long duration: apply full scale
        return scale_factor
    elif estimated_duration <= 5.0:
        # Short duration: apply minimal scale (10% of requested)
        return 1.0 + (scale_factor - 1.0) * 0.1
    else:
        # Medium duration: linear interpolation
        ratio = (estimated_duration - 5.0) / (20.0 - 5.0)
        return 1.0 + (scale_factor - 1.0) * ratio


def seed_everything(seed: Optional[int], *, deterministic: bool = False) -> int:
    """
    Seed all RNGs. If seed is None, draw a fresh random seed and return it.
    """
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
        print(f"[Info] No seed provided; using random seed {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism costs performance; prefer speed by default.
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    return seed


# ---------------------------------------------------------------------------
# Model / tokenizer loaders (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_resources(
    model_dir: str,
    xcodec2_model_name: Optional[str],
    xcodec2_sample_rate: Optional[int],
    use_torch_compile: bool,
    cpu_codec: bool,
    whisper_device: str,
):
    """Load model and tokenizers from HF-format directory or repo."""
    if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
        raise ImportError("Please install transformers before running the demo.")

    if torch.cuda.is_available():
        device = "cuda"
        # Favor throughput on modern NVIDIA GPUs (TF32 for matmul/conv where applicable).
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    # Some older exported checkpoints ship remote code without `inference_tts_batch`.
    # Monkey-patch it from this repo's HF wrapper when missing.
    try:
        from hf_export.modeling_t5gemma_voice import T5GemmaVoiceForConditionalGeneration as _LocalHFModel

        target = model._orig_mod if hasattr(model, "_orig_mod") else model

        # Patch inference_tts_batch if missing
        if not hasattr(target, "inference_tts_batch") and hasattr(_LocalHFModel, "inference_tts_batch"):
            patched = MethodType(_LocalHFModel.inference_tts_batch, target)
            setattr(target, "inference_tts_batch", patched)
            if target is not model:
                setattr(model, "inference_tts_batch", MethodType(_LocalHFModel.inference_tts_batch, model))
            print("[Info] Patched missing inference_tts_batch onto loaded model.")

        # Patch inference_tts_batch_multi_text if missing (for sentence segmentation)
        if not hasattr(target, "inference_tts_batch_multi_text") and hasattr(_LocalHFModel, "inference_tts_batch_multi_text"):
            patched_multi = MethodType(_LocalHFModel.inference_tts_batch_multi_text, target)
            setattr(target, "inference_tts_batch_multi_text", patched_multi)
            if target is not model:
                setattr(model, "inference_tts_batch_multi_text", MethodType(_LocalHFModel.inference_tts_batch_multi_text, model))
            print("[Info] Patched missing inference_tts_batch_multi_text onto loaded model.")
    except Exception as e:
        print(f"[Warning] Failed to patch batch methods; will run sequentially: {e}")

    cfg = model.config

    tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    xcodec2_model_name = xcodec2_model_name or getattr(cfg, "xcodec2_model_name", None)

    audio_tokenizer = AudioTokenizer(
        backend="xcodec2",
        device=torch.device("cpu") if cpu_codec else torch.device(device),
        model_name=xcodec2_model_name,
        sample_rate=xcodec2_sample_rate,
    )

    # Apply torch.compile for faster inference (2nd run onwards)
    can_compile = (
        use_torch_compile
        and torch.cuda.is_available()
        and not cpu_codec
    )
    if can_compile:
        print("[Info] Applying torch.compile to model and codec (this may take a minute on first inference)...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            audio_tokenizer.codec = torch.compile(audio_tokenizer.codec, mode="reduce-overhead")
            print("[Info] torch.compile applied successfully.")
        except Exception as e:
            print(f"[Warning] torch.compile failed, falling back to eager mode: {e}")

    codec_audio_sr = getattr(cfg, "codec_audio_sr", audio_tokenizer.sample_rate)
    if xcodec2_sample_rate is not None:
        codec_audio_sr = xcodec2_sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)

    return {
        "model": model,
        "cfg": cfg,
        "text_tokenizer": text_tokenizer,
        "audio_tokenizer": audio_tokenizer,
        "device": device,
        "codec_audio_sr": codec_audio_sr,
        "codec_sr": codec_sr,
        "whisper_device": whisper_device,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    reference_speech: Optional[str],
    reference_text: Optional[str],
    target_text: str,
    target_duration: Optional[float],
    top_k: int,
    top_p: float,
    min_p: float,
    temperature: float,
    seed: Optional[int],
    resources: dict,
    cut_off_sec: int = 100,
    lang: Optional[str] = None,
    batch_count: int = 1,
    duration_scale: float = 1.0,
) -> list:
    """
    Run TTS and return a list of (sample_rate, waveform) tuples for Gradio playback.
    When batch_count > 1, generates multiple samples with different seeds.
    """
    model = resources["model"]
    cfg = resources["cfg"]
    text_tokenizer = resources["text_tokenizer"]
    audio_tokenizer = resources["audio_tokenizer"]
    device = resources["device"]
    codec_audio_sr = resources["codec_audio_sr"]
    codec_sr = resources["codec_sr"]

    silence_tokens = []
    repeat_prompt = 0  # fixed; UI removed
    stop_repetition = 3  # keep sensible default from CLI HF script
    batch_mode = batch_count > 1 and hasattr(model, "inference_tts_batch")
    if batch_count > 1 and not batch_mode:
        print("[Info] Model lacks inference_tts_batch; using sequential per-sample generation.")
    sample_batch_size = batch_count if batch_mode else 1

    no_reference_audio = reference_speech is None or str(reference_speech).strip().lower() in {"", "none", "null"}
    has_reference_text = reference_text not in (None, "", "none", "null")

    if no_reference_audio and has_reference_text:
        raise ValueError("reference_text was provided but reference_speech is missing.")

    if no_reference_audio:
        prefix_transcript = ""
    elif not has_reference_text:
        print("[Info] No reference text; transcribing reference speech with Whisper.")
        prefix_transcript = transcribe_audio(reference_speech, resources["whisper_device"])
        print(f"[Info] Whisper transcription: {prefix_transcript}")
        unload_whisper_model()
    else:
        prefix_transcript = reference_text

    # Language + normalization (Japanese only)
    lang = None if lang in {None, "", "none", "null"} else str(lang)
    target_text, lang_code = normalize_text_with_lang(target_text, lang)
    if prefix_transcript:
        prefix_transcript, _ = normalize_text_with_lang(prefix_transcript, lang_code)

    if target_duration is None:
        estimated_length = estimate_duration(
            target_text=target_text,
            reference_speech=None if no_reference_audio else reference_speech,
            reference_transcript=None if no_reference_audio else prefix_transcript,
            target_lang=lang_code,
            reference_lang=lang_code,
        )
        print(f"[Info] target_duration not provided, estimated as {estimated_length:.2f} seconds.")

        # Apply adaptive duration scale
        if duration_scale != 1.0:
            actual_scale = adaptive_duration_scale(estimated_length, duration_scale)
            target_generation_length = estimated_length * actual_scale
            print(f"[Info] Applied adaptive duration scale {actual_scale:.3f}x (requested {duration_scale:.2f}x for {estimated_length:.2f}s) -> {target_generation_length:.2f} seconds.")
        else:
            target_generation_length = estimated_length
    else:
        # User specified explicit duration, apply scale directly
        target_generation_length = float(target_duration)
        if duration_scale != 1.0:
            target_generation_length *= duration_scale
            print(f"[Info] Applied duration scale {duration_scale:.2f}x to user-specified duration -> {target_generation_length:.2f} seconds.")

    if not no_reference_audio:
        info = get_audio_info(reference_speech)
        prompt_end_frame = int(cut_off_sec * get_sample_rate(info))
    else:
        prompt_end_frame = 0

    decode_config = {
        "top_k": int(top_k),
        "top_p": float(top_p),
        "min_p": float(min_p),
        "temperature": float(temperature),
        "stop_repetition": stop_repetition,
        "codec_audio_sr": codec_audio_sr,
        "codec_sr": codec_sr,
        "silence_tokens": silence_tokens,
        "sample_batch_size": sample_batch_size,
    }

    results = []
    if batch_mode:
        base_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
        used_seed = seed_everything(base_seed, deterministic=bool(int(os.getenv("T5GEMMA_DETERMINISTIC", "0"))))
        print(f"\n[Info] Running batched generation of {batch_count} samples on one GPU (seed base: {used_seed})...")

        concat_audio, gen_audio = inference_one_sample(
            model=model,
            model_args=cfg,
            text_tokenizer=text_tokenizer,
            audio_tokenizer=audio_tokenizer,
            audio_fn=None if no_reference_audio else reference_speech,
            target_text=target_text,
            lang=lang_code,
            device=device,
            decode_config=decode_config,
            prompt_end_frame=prompt_end_frame,
            target_generation_length=target_generation_length,
            prefix_transcript=prefix_transcript,
            multi_trial=[],
            repeat_prompt=repeat_prompt,
            return_frames=False,
        )

        batch_concat = concat_audio if isinstance(concat_audio, (list, tuple)) else [concat_audio]
        batch_gen = gen_audio if isinstance(gen_audio, (list, tuple)) else [gen_audio]

        for i, gen in enumerate(batch_gen):
            gen_audio_tensor = gen[0].detach().cpu()
            if gen_audio_tensor.ndim == 2 and gen_audio_tensor.shape[0] == 1:
                gen_audio_tensor = gen_audio_tensor.squeeze(0)
            waveform = gen_audio_tensor.numpy()

            max_abs = float(np.max(np.abs(waveform)))
            rms = float(np.sqrt(np.mean(waveform**2)))
            print(f"[Info] Sample {i+1} stats -> max_abs: {max_abs:.6f}, rms: {rms:.6f}, seed base: {used_seed}")

            results.append((codec_audio_sr, waveform))
    else:
        # Generate seeds for batch (sequential fallback)
        base_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
        seeds = [base_seed + i for i in range(batch_count)]

        if batch_count > 1:
            print("[Info] Model lacks inference_tts_batch; running samples sequentially on one GPU.")

        for i, s in enumerate(seeds):
            print(f"\n[Info] Generating sample {i+1}/{batch_count} with seed {s}...")
            used_seed = seed_everything(s, deterministic=bool(int(os.getenv("T5GEMMA_DETERMINISTIC", "0"))))

            concat_audio, gen_audio = inference_one_sample(
                model=model,
                model_args=cfg,
                text_tokenizer=text_tokenizer,
                audio_tokenizer=audio_tokenizer,
                audio_fn=None if no_reference_audio else reference_speech,
                target_text=target_text,
                lang=lang_code,
                device=device,
                decode_config=decode_config,
                prompt_end_frame=prompt_end_frame,
                target_generation_length=target_generation_length,
                prefix_transcript=prefix_transcript,
                multi_trial=[],
                repeat_prompt=repeat_prompt,
                return_frames=False,
            )

            # Take generated audio, move to CPU, convert to numpy for Gradio
            gen_audio = gen_audio[0].detach().cpu()
            if gen_audio.ndim == 2 and gen_audio.shape[0] == 1:
                gen_audio = gen_audio.squeeze(0)
            waveform = gen_audio.numpy()

            max_abs = float(np.max(np.abs(waveform)))
            rms = float(np.sqrt(np.mean(waveform**2)))
            print(f"[Info] Sample {i+1} stats -> max_abs: {max_abs:.6f}, rms: {rms:.6f}, seed: {used_seed}")

            results.append((codec_audio_sr, waveform))

    return results


def run_inference_segmented(
    reference_speech: Optional[str],
    reference_text: Optional[str],
    target_text: str,
    target_duration: Optional[float],
    top_k: int,
    top_p: float,
    min_p: float,
    temperature: float,
    seed: Optional[int],
    resources: dict,
    cut_off_sec: int = 100,
    lang: Optional[str] = None,
    batch_count: int = 1,
    inter_segment_silence: float = 0.05,
    duration_scale: float = 1.0,
) -> Tuple[list, list, list, list, list, list]:
    """
    Run segmented TTS inference with chunking: split text into ~550 char chunks,
    then split each chunk by sentences and generate in parallel batch.

    Returns:
        Tuple of (concatenated_results, chunk_results, chunk_texts, segment_results, segment_texts, saved_files).
    """
    from datetime import datetime
    from inference_tts_utils import split_into_adaptive_chunks, save_audio
    from data.tokenizer import tokenize_audio
    
    model = resources["model"]
    cfg = resources["cfg"]
    text_tokenizer = resources["text_tokenizer"]
    audio_tokenizer = resources["audio_tokenizer"]
    device = resources["device"]
    codec_audio_sr = resources["codec_audio_sr"]
    codec_sr = resources["codec_sr"]

    # Create outputs directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Split text into chunks
    chunks = split_into_adaptive_chunks(target_text, target_size=550)
    if not chunks:
        chunks = [target_text]

    print(f"[Info] Split text into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk[:50]}{'...' if len(chunk) > 50 else ''} ({len(chunk)} chars)")

    silence_tokens = []
    stop_repetition = 3

    no_reference_audio = reference_speech is None or str(reference_speech).strip().lower() in {"", "none", "null"}
    has_reference_text = reference_text not in (None, "", "none", "null")

    if no_reference_audio and has_reference_text:
        raise ValueError("reference_text was provided but reference_speech is missing.")

    if no_reference_audio:
        prefix_transcript = ""
    elif not has_reference_text:
        print("[Info] No reference text; transcribing reference speech with Whisper (large-v3-turbo).")
        prefix_transcript = transcribe_audio(reference_speech, resources["whisper_device"])
        print(f"[Info] Whisper transcription: {prefix_transcript}")
        unload_whisper_model()
    else:
        prefix_transcript = reference_text

    # Language + normalization
    lang = None if lang in {None, "", "none", "null"} else str(lang)
    _, lang_code = normalize_text_with_lang(target_text, lang)
    if prefix_transcript:
        prefix_transcript, _ = normalize_text_with_lang(prefix_transcript, lang_code)

    # Prepare reference audio (shared across all chunks)
    if not no_reference_audio:
        info = get_audio_info(reference_speech)
        prompt_end_frame = int(cut_off_sec * get_sample_rate(info))
        encoded_frames = tokenize_audio(
            audio_tokenizer,
            reference_speech,
            offset=0,
            num_frames=prompt_end_frame if prompt_end_frame > 0 else -1,
        )
    else:
        prompt_end_frame = 0
        encoded_frames = torch.empty((1, 1, 0), dtype=torch.long, device=audio_tokenizer.device)

    # Ensure proper shape [1, 1, T]
    if encoded_frames.ndim == 2:
        encoded_frames = encoded_frames.unsqueeze(0)
    if encoded_frames.shape[2] == 1:
        encoded_frames = encoded_frames.transpose(1, 2).contiguous()

    # Add y_sep token if present and has reference
    y_sep_token = getattr(cfg, "y_sep_token", None)
    if y_sep_token is not None and not no_reference_audio and encoded_frames.shape[2] > 0:
        encoded_frames = torch.cat([
            encoded_frames,
            torch.full((1, 1, 1), y_sep_token, dtype=torch.long, device=encoded_frames.device),
        ], dim=2)

    original_audio = encoded_frames.transpose(2, 1).contiguous()  # [1, T, 1]
    prompt_frames = original_audio.shape[1]

    # Tokenization helpers
    x_sep_token = getattr(cfg, "x_sep_token", None)
    add_eos_token = getattr(cfg, "add_eos_to_text", 0)
    add_bos_token = getattr(cfg, "add_bos_to_text", 0)

    def encode_text_hf(text):
        if isinstance(text, list):
            text = " ".join(text)
        return text_tokenizer.encode(text.strip(), add_special_tokens=False)

    eos_token = getattr(cfg, "eos", getattr(cfg, "eog", None))

    def strip_special(frames):
        mask = torch.ones_like(frames, dtype=torch.bool)
        if y_sep_token is not None:
            mask &= frames.ne(y_sep_token)
        if eos_token is not None:
            mask &= frames.ne(eos_token)
        if mask.all():
            return frames
        new_len = int(mask.sum(dim=2).min().item())
        return frames[:, :, :new_len]

    concatenated_results = []
    all_chunk_results = []
    all_saved_files = []
    
    # Store detailed segments for all chunks (first batch)
    all_segments_details = []

    # Store first chunk's segments for UI display (Legacy, can remove or keep empty)
    first_chunk_segment_results = []
    first_chunk_segment_texts = []

    # Run batch_count iterations with different seeds
    base_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    seeds = [base_seed + i for i in range(batch_count)]

    for batch_idx, s in enumerate(seeds):
        print(f"\n[Info] Generating batch {batch_idx+1}/{batch_count} with seed {s}...")
        used_seed = seed_everything(s, deterministic=bool(int(os.getenv("T5GEMMA_DETERMINISTIC", "0"))))

        chunk_audios = []

        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks):
            print(f"\n[Info] Processing chunk {chunk_idx+1}/{len(chunks)} (batch {batch_idx+1}/{batch_count})...")
            
            # Segment the chunk into sentences
            segments = segment_text_by_sentences(chunk)
            if not segments:
                segments = [chunk]

            print(f"[Info] Chunk {chunk_idx+1} split into {len(segments)} segments:")
            for i, seg in enumerate(segments):
                print(f"  Segment {i+1}: {seg[:50]}{'...' if len(seg) > 50 else ''} ({len(seg)} chars)")

            # Normalize each segment
            normalized_segments = []
            for seg in segments:
                norm_seg, _ = normalize_text_with_lang(seg, lang_code)
                normalized_segments.append(norm_seg)

            # Estimate duration for each segment
            segment_durations = []
            for seg in normalized_segments:
                dur = estimate_duration(
                    target_text=seg,
                    reference_speech=None if no_reference_audio else reference_speech,
                    reference_transcript=None if no_reference_audio else prefix_transcript,
                    target_lang=lang_code,
                    reference_lang=lang_code,
                )
                segment_durations.append(dur)

            # Apply adaptive duration scale to each segment
            if duration_scale != 1.0:
                scaled_durations = []
                for i, dur in enumerate(segment_durations):
                    actual_scale = adaptive_duration_scale(dur, duration_scale)
                    scaled_dur = dur * actual_scale
                    scaled_durations.append(scaled_dur)
                segment_durations = scaled_durations

            # Tokenize all segments for batch processing
            all_text_tokens = []
            for seg in normalized_segments:
                tokens = encode_text_hf(seg)
                if prefix_transcript:
                    prefix_tokens = encode_text_hf(prefix_transcript)
                    if x_sep_token is not None:
                        tokens = prefix_tokens + [x_sep_token] + tokens
                    else:
                        tokens = prefix_tokens + tokens
                if add_eos_token:
                    tokens.append(add_eos_token)
                if add_bos_token:
                    tokens = [add_bos_token] + tokens
                all_text_tokens.append(tokens)

            # Pad to same length for batching
            max_len = max(len(t) for t in all_text_tokens)
            pad_token_id = text_tokenizer.pad_token_id or 0

            x_padded = torch.full((len(segments), max_len), pad_token_id, dtype=torch.long, device=device)
            x_lens = torch.zeros(len(segments), dtype=torch.long, device=device)
            for i, tokens in enumerate(all_text_tokens):
                x_padded[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                x_lens[i] = len(tokens)

            # Compute target y lengths
            tgt_y_lens = torch.zeros(len(segments), dtype=torch.long, device=device)
            for i, dur in enumerate(segment_durations):
                tgt_y_lens[i] = int(prompt_frames + codec_sr * dur)

            # Check if model has multi-text batch method
            target = model._orig_mod if hasattr(model, "_orig_mod") else model
            has_multi_text = hasattr(target, "inference_tts_batch_multi_text")

            if has_multi_text and len(segments) > 1:
                # Use parallel multi-text inference for batch processing
                print(f"[Info] Using parallel batch inference for {len(segments)} segments")
                concat_frames, gen_frames = target.inference_tts_batch_multi_text(
                    x_padded,
                    x_lens,
                    original_audio.to(device),
                    tgt_y_lens=tgt_y_lens,
                    top_k=int(top_k),
                    top_p=float(top_p),
                    min_p=float(min_p),
                    temperature=float(temperature),
                    stop_repetition=stop_repetition,
                    silence_tokens=silence_tokens,
                    prompt_frames=prompt_frames,
                )
            else:
                # Fall back to sequential generation
                if len(segments) > 1:
                    print("[Info] Model lacks inference_tts_batch_multi_text; generating segments sequentially")
                concat_frames = []
                gen_frames = []

                decode_config = {
                    "top_k": int(top_k),
                    "top_p": float(top_p),
                    "min_p": float(min_p),
                    "temperature": float(temperature),
                    "stop_repetition": stop_repetition,
                    "codec_audio_sr": codec_audio_sr,
                    "codec_sr": codec_sr,
                    "silence_tokens": silence_tokens,
                    "sample_batch_size": 1,
                }

                for seg_idx, seg in enumerate(normalized_segments):
                    print(f"[Info] Generating segment {seg_idx+1}/{len(normalized_segments)}...")
                    concat_sample, gen_sample, concat_frame, gen_frame = inference_one_sample(
                        model=model,
                        model_args=cfg,
                        text_tokenizer=text_tokenizer,
                        audio_tokenizer=audio_tokenizer,
                        audio_fn=None if no_reference_audio else reference_speech,
                        target_text=seg,
                        lang=lang_code,
                        device=device,
                        decode_config=decode_config,
                        prompt_end_frame=prompt_end_frame,
                        target_generation_length=segment_durations[seg_idx],
                        prefix_transcript=prefix_transcript,
                        multi_trial=[],
                        repeat_prompt=0,
                        return_frames=True,
                    )
                    concat_frames.append(concat_frame)
                    gen_frames.append(gen_frame)

            # Decode each segment's audio
            segment_audios = []
            for i, gf in enumerate(gen_frames):
                gf_clean = strip_special(gf)
                gen_audio = audio_tokenizer.decode(gf_clean)
                gen_tensor = gen_audio[0].detach().cpu()
                if gen_tensor.ndim == 2 and gen_tensor.shape[0] == 1:
                    gen_tensor = gen_tensor.squeeze(0)
                waveform = gen_tensor.numpy()
                segment_audios.append((codec_audio_sr, waveform))

            # Store segments for UI (for the first batch only)
            if batch_idx == 0:
                # Zip texts and audios for this chunk
                chunk_seg_details = []
                for txt, aud in zip(segments, segment_audios):
                    chunk_seg_details.append((txt, aud))
                all_segments_details.append(chunk_seg_details)

            # Concatenate segments within this chunk
            chunk_audio = concatenate_audio_segments(segment_audios, silence_sec=inter_segment_silence)
            chunk_audios.append(chunk_audio)

            # Save chunk audio to file
            chunk_filename = f"output_{timestamp}_chunk_{chunk_idx+1}_batch_{batch_idx+1}.wav"
            chunk_filepath = os.path.join(output_dir, chunk_filename)
            chunk_waveform = torch.from_numpy(chunk_audio[1])
            if chunk_waveform.ndim == 1:
                chunk_waveform = chunk_waveform.unsqueeze(0)
            save_audio(chunk_filepath, chunk_waveform, chunk_audio[0])
            all_saved_files.append(chunk_filepath)
            print(f"[Info] Saved chunk {chunk_idx+1} to {chunk_filepath}")

        all_chunk_results.append(chunk_audios)

        # Concatenate all chunks for this batch
        full_audio = concatenate_audio_segments(chunk_audios, silence_sec=inter_segment_silence)
        concatenated_results.append(full_audio)

        # Save final concatenated audio
        final_filename = f"output_{timestamp}_final_batch_{batch_idx+1}.wav"
        final_filepath = os.path.join(output_dir, final_filename)
        final_waveform = torch.from_numpy(full_audio[1])
        if final_waveform.ndim == 1:
            final_waveform = final_waveform.unsqueeze(0)
        save_audio(final_filepath, final_waveform, full_audio[0])
        all_saved_files.append(final_filepath)
        print(f"[Info] Saved final concatenated audio to {final_filepath}")
        print(f"[Info] Batch {batch_idx+1} complete: {full_audio[1].shape[0]/full_audio[0]:.2f}s")

    return (
        concatenated_results,
        all_chunk_results,
        chunks,
        [first_chunk_segment_results] if first_chunk_segment_results else [],
        first_chunk_segment_texts,
        all_saved_files,
        all_segments_details,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo(
    resources,
    server_port: int,
    share: bool,
    max_batch: int = 64,
    max_segments: int = 64,
    inter_segment_silence: float = 1.00,
):
    description = (
        "Reference speech is optional. If provided without reference text, Whisper (large-v3-turbo) "
        "will auto-transcribe and use it as the prompt text.\n\n"
        "**Batch Generation**: Set batch count > 1 to generate multiple samples with different seeds.\n\n"
        "**Sentence Segmentation**: Enable to split long text by sentence delimiters (。！？!?.) and generate each segment in parallel."
    )

    with gr.Blocks() as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("台本生成 (Script Generation)", id="tab_script"):
                gr.Markdown("### Script Generation (One-sided Manzai)")
                gr.Markdown("#### 1. Theme Selection")
                
                theme_input = gr.Textbox(label="Theme (Enter a theme for the script)", placeholder="e.g. 「頭を冷やす」と言われて冷蔵庫に入ろうとする", interactive=True)
                
                gr.Markdown("#### 2. Character Generation")
                def load_prompt_file(path):
                    if os.path.exists(path):
                        with open(path, encoding='utf-8') as f:
                            return f.read()
                    return "" # No fallback to hardcoded values
                
                default_chars = load_prompt_file("prompts/character_settings.txt")
                char_settings_input = gr.Textbox(label="Character Personas", value=default_chars, lines=10, interactive=True)
                
                gr.Markdown("#### 3. Script Generation")
                
                with gr.Row():
                    script_gen_btn = gr.Button("Generate Script", variant="primary")
                
                script_output = gr.Textbox(label="Generated Script", lines=20, interactive=True)
                
                with gr.Row():
                    script_save_btn = gr.Button("Save to File")
                    script_transfer_btn = gr.Button("Transfer to Speech Synthesis")
                
                script_save_status = gr.Textbox(label="Status", interactive=False, visible=True)
            with gr.Tab("読み上げ (Speech Synthesis)", id="tab_speech"):
                gr.Markdown("## T5Gemma-TTS (HF)")
                gr.Markdown(description)
        
                # State to store hierarchial data:
                # List of dicts (chunks), each dict:
                # {
                #   "full_text": str,
                #   "full_audio": (sr, wav),
                #   "segments": [
                #       { "text": str, "audio": (sr, wav), "candidates": [] }, ...
                #   ]
                # }
                segments_state = gr.State([])
        
                with gr.Row():
                    reference_speech_input = gr.Audio(
                        label="Reference Speech (optional)",
                        type="filepath",
                    )
                    reference_text_box = gr.Textbox(
                        label="Reference Text (optional, leave blank to auto-transcribe)",
                        lines=2,
                    )
        
                target_text_box = gr.Textbox(
                    label="Target Text",
                    value="こんにちは、私はAIです。これは音声合成のテストです。",
                    lines=3,
                )
        
                with gr.Row():
                    target_duration_box = gr.Textbox(
                        label="Target Duration (seconds, optional)",
                        value="",
                        placeholder="Leave blank for auto estimate",
                    )
                    duration_scale_box = gr.Slider(
                        label="Duration Scale / 時間スケール",
                        minimum=0.5,
                        maximum=1.5,
                        step=0.05,
                        value=1.0,
                        info="Adaptive scaling: affects long segments more (>20s), short segments less (<5s) to prevent cutoff",
                    )
                    seed_box = gr.Textbox(
                        label="Random Seed (optional)",
                        value="",
                        placeholder="Leave blank to use a random seed each run",
                    )
                    batch_count_box = gr.Slider(
                        label="Batch Count",
                        minimum=1,
                        maximum=max_batch,
                        step=1,
                        value=1,
                        info="Number of samples to generate with different seeds",
                    )
        
                # Segmentation controls
                with gr.Row():
                    enable_segmentation_box = gr.Checkbox(
                        label="Enable Sentence Segmentation / 文分割を有効化",
                        value=False,
                        info="Split text by sentence delimiters and generate in parallel",
                    )
                    inter_segment_silence_box = gr.Slider(
                        label="Inter-segment Silence / セグメント間の無音（秒）",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=inter_segment_silence,
                        info="Silence duration between segments when concatenating",
                    )
        
                with gr.Row():
                    top_k_box = gr.Slider(label="top_k", minimum=0, maximum=100, step=1, value=30)
                    top_p_box = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, step=0.05, value=0.9)
                    min_p_box = gr.Slider(label="min_p (0 = disabled)", minimum=0.0, maximum=1.0, step=0.05, value=0.0)
                    temperature_box = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, step=0.05, value=0.8)
        
                generate_button = gr.Button("Generate")
        
                # Concatenated audio output (shown first when segmentation is enabled)
                concatenated_audio_outputs = []
                with gr.Row(visible=False) as concat_row:
                    for i in range(max_batch):
                        audio = gr.Audio(
                            label=f"Full Concatenated Audio {i+1}" if i > 0 else "Full Concatenated Audio",
                            type="numpy",
                            interactive=False,
                            visible=(i == 0),
                        )
                        concatenated_audio_outputs.append(audio)
        
                # Saved files display
                saved_files_box = gr.Textbox(
                    label="Saved Files / 保存されたファイル",
                    interactive=False,
                    visible=False,
                    lines=5,
                )
        
                # Chunk outputs in accordion
                chunk_accordion = gr.Accordion("Chunks (Edit Segments)", open=True, visible=False)
                
                max_chunks = 5
                max_segments_per_chunk = 32
                
                # Flattened list of all segment UI dicts, keyed by (chunk_idx, seg_idx)
                all_segment_ui = [] 
        
                # We also need UI for the chunks themselves (read-only text + audio)
                chunk_ui_elements = [] 
        
                with chunk_accordion:
                    for i in range(max_chunks):
                        with gr.Group(visible=False) as chunk_group:
                            gr.Markdown(f"### Chunk {i+1}")
                            with gr.Row():
                                chunk_audio_player = gr.Audio(
                                    label=f"Chunk {i+1} Audio",
                                    type="numpy",
                                    interactive=False,
                                )
                                chunk_text_display = gr.Textbox(
                                    label=f"Chunk {i+1} Full Text",
                                    interactive=False,
                                    lines=2,
                                )
                            
                            with gr.Accordion(f"Edit Segments (Chunk {i+1})", open=False):
                                segment_uis_for_this_chunk = []
                                for j in range(max_segments_per_chunk):
                                    with gr.Group(visible=False) as seg_group:
                                        with gr.Row():
                                            seg_text_input = gr.Textbox(
                                                label=f"Seg {j+1}", 
                                                interactive=True, 
                                                scale=3,
                                                lines=1
                                            )
                                            seg_audio_player = gr.Audio(
                                                label=f"Audio",
                                                type="numpy", 
                                                interactive=False,
                                                container=False,
                                                scale=1,
                                            )
                                            seg_regen_btn = gr.Button("Regen", size="sm", scale=0)
                                        
                                        # Candidates for this segment
                                        with gr.Group(visible=False) as cand_group:
                                            cand_audios = []
                                            with gr.Row():
                                                for c in range(8):
                                                    ca = gr.Audio(
                                                        label=f"C{c+1}",
                                                        type="numpy",
                                                        interactive=False,
                                                        container=True,
                                                        min_width=100,
                                                    )
                                                    cand_audios.append(ca)
                                            
                                            cand_radio = gr.Radio(
                                                choices=[f"C{c+1}" for c in range(8)],
                                                label="Select",
                                                type="index",
                                            )
                                            cand_update_btn = gr.Button("Update Segment", size="sm", variant="primary")
        
                                        segment_uis_for_this_chunk.append({
                                            "group": seg_group,
                                            "text": seg_text_input,
                                            "audio": seg_audio_player,
                                            "regen_btn": seg_regen_btn,
                                            "cand_group": cand_group,
                                            "cand_audios": cand_audios,
                                            "radio": cand_radio,
                                            "update_btn": cand_update_btn,
                                            "chunk_idx": i,
                                            "seg_idx": j,
                                        })
                                all_segment_ui.append(segment_uis_for_this_chunk)
                            
                            chunk_ui_elements.append({
                                "group": chunk_group,
                                "audio": chunk_audio_player,
                                "text": chunk_text_display,
                            })
        
                # Create multiple audio outputs for batch generation (non-segmented mode)
                output_audios = []
                with gr.Row() as batch_row:
                    for i in range(max_batch):
                        audio = gr.Audio(
                            label=f"Generated Audio {i+1}",
                            type="numpy",
                            interactive=False,
                            visible=(i == 0),
                        )
                        output_audios.append(audio)
        
                def toggle_segmentation_ui(enabled):
                    return [
                        gr.update(visible=enabled),  # concat_row
                        gr.update(visible=not enabled),  # batch_row
                        gr.update(visible=enabled),  # saved_files_box
                        gr.update(visible=enabled),  # chunk_accordion
                    ]
        
                enable_segmentation_box.change(
                    fn=toggle_segmentation_ui,
                    inputs=[enable_segmentation_box],
                    outputs=[concat_row, batch_row, saved_files_box, chunk_accordion],
                )
        
                def regenerate_segment(
                    chunk_idx,
                    seg_idx,
                    text,
                    current_state,
                    reference_speech,
                    reference_text,
                    duration_scale,
                    top_k,
                    top_p,
                    min_p,
                    temperature,
                    seed,
                    progress=gr.Progress(),
                ):
                    progress(0.1, desc="Initializing...")
                    print(f"[Info] Regenerating Chunk {chunk_idx+1}-Seg {seg_idx+1}: {text}")
                    seed_val = None
                    if str(seed).strip() not in {"", "None", "none"}:
                        seed_val = int(float(seed))
                    
                    progress(0.3, desc="Generating 8 candidates (batch)...")
                    # Run inference for this segment text as a single sample (batch 8 candidates)
                    results = run_inference(
                        reference_speech=reference_speech,
                        reference_text=reference_text,
                        target_text=text,
                        target_duration=None, 
                        top_k=top_k,
                        top_p=top_p,
                        min_p=min_p,
                        temperature=temperature,
                        seed=seed_val,
                        resources=resources,
                        batch_count=8, 
                        duration_scale=duration_scale,
                    )
                    progress(1.0, desc="Rendering...")
                    
                    # Update state with new text and candidates
                    new_state = list(current_state)
                    if chunk_idx < len(new_state):
                        segs = new_state[chunk_idx]["segments"]
                        if seg_idx < len(segs):
                            segs[seg_idx]["text"] = text
                            segs[seg_idx]["candidates"] = results
                    
                    # Return UI updates for candidates
                    outputs = [gr.update(visible=True)] # show cand_group
                    for i in range(8):
                        if i < len(results):
                            outputs.append(gr.update(value=results[i], visible=True))
                        else:
                            outputs.append(gr.update(value=None, visible=False))
                    outputs.append(gr.update(value=None)) # reset radio
                    outputs.append(new_state)
                    # Reset button state
                    outputs.append(gr.update(value="Regen", interactive=True))
                    return outputs
        
                def start_regen():
                    return gr.update(value="Wait...", interactive=False)
        
                def update_segment(
                    chunk_idx,
                    seg_idx,
                    selected_idx,
                    current_state,
                    inter_silence,
                ):
                    if selected_idx is None:
                        return [gr.update(), gr.update(), gr.update(), gr.update(), current_state]
        
                    new_state = list(current_state)
                    from inference_tts_utils import concatenate_audio_segments
        
                    # 1. Update Segment Audio
                    if chunk_idx < len(new_state):
                        segs = new_state[chunk_idx]["segments"]
                        if seg_idx < len(segs):
                            candidates = segs[seg_idx].get("candidates", [])
                            if selected_idx < len(candidates):
                                segs[seg_idx]["audio"] = candidates[selected_idx]
                    
                    # 2. Stitch Segments -> Chunk Audio
                    chunk_data = new_state[chunk_idx]
                    seg_audios = [s["audio"] for s in chunk_data["segments"] if s.get("audio") is not None]
                    
                    updated_chunk_audio = None
                    if seg_audios:
                        # Use slider value for intra-chunk segments
                        updated_chunk_audio = concatenate_audio_segments(seg_audios, silence_sec=inter_silence)
                        chunk_data["full_audio"] = updated_chunk_audio
                    
                    # 3. Stitch Chunks -> Full Audio
                    all_chunk_audios = [c["full_audio"] for c in new_state if c.get("full_audio") is not None]
                    updated_full_audio = None
                    if all_chunk_audios:
                        updated_full_audio = concatenate_audio_segments(all_chunk_audios, silence_sec=inter_silence)
        
                    return [
                        gr.update(value=new_state[chunk_idx]["segments"][seg_idx]["audio"]), # Segment Player
                        gr.update(visible=False), # Hide Candidates
                        gr.update(value=updated_chunk_audio), # Chunk Player
                        gr.update(value=updated_full_audio), # Full Player
                        new_state
                    ]
        
                # Register Event Handlers
                for chunk_i in range(max_chunks):
                    for seg_j in range(max_segments_per_chunk):
                        ui = all_segment_ui[chunk_i][seg_j]
                        
                        # Regen
                        ui["regen_btn"].click(
                            fn=start_regen,
                            inputs=None,
                            outputs=[ui["regen_btn"]],
                            queue=False,
                        ).then(
                            fn=regenerate_segment,
                            inputs=[
                                gr.Number(value=chunk_i, visible=False),
                                gr.Number(value=seg_j, visible=False),
                                ui["text"],
                                segments_state,
                                reference_speech_input,
                                reference_text_box,
                                duration_scale_box,
                                top_k_box,
                                top_p_box,
                                min_p_box,
                                temperature_box,
                                seed_box,
                            ],
                            outputs=[ui["cand_group"]] + ui["cand_audios"] + [ui["radio"], segments_state, ui["regen_btn"]],
                            show_progress="full",
                        )
        
                        # Update
                        ui["update_btn"].click(
                            fn=update_segment,
                            inputs=[
                                gr.Number(value=chunk_i, visible=False),
                                gr.Number(value=seg_j, visible=False),
                                ui["radio"],
                                segments_state,
                                inter_segment_silence_box,
                            ],
                            outputs=[
                                ui["audio"],
                                ui["cand_group"],
                                chunk_ui_elements[chunk_i]["audio"],
                                concatenated_audio_outputs[0],
                                segments_state
                            ]
                        )
        
                def gradio_inference(
                    reference_speech,
                    reference_text,
                    target_text,
                    target_duration,
                    duration_scale,
                    top_k,
                    top_p,
                    min_p,
                    temperature,
                    seed,
                    batch_count,
                    enable_segmentation,
                    inter_segment_silence_val,
                ):
                    dur = None
                    if str(target_duration).strip() not in {"", "None", "none"}:
                        try:
                            dur = float(target_duration)
                        except ValueError:
                            print(f"[Warning] Invalid duration value, using auto-estimate")
                    seed_val = None
                    if str(seed).strip() not in {"", "None", "none"}:
                        seed_val = int(float(seed))
                    batch_count = int(batch_count)
        
                    use_segmentation = enable_segmentation
                    
                    if use_segmentation:
                        # Need detailed segment info from run_inference_segmented
                        # NOTE: run_inference_segmented currently returns:
                        # (concatenated_results, chunk_results, chunk_texts, segment_results, segment_texts, saved_files)
                        # It does NOT return the necessary granular details for ALL chunks. 
                        # We will need to update `run_inference_segmented` to return `all_segments_details`.
                        # For now, let's assume it returns a 7th element `all_detailed_segments`.
                        
                        # Mocking the call structure for now - we will update the backend function next.
                        res = run_inference_segmented(
                            reference_speech=reference_speech,
                            reference_text=reference_text,
                            target_text=target_text,
                            target_duration=dur,
                            top_k=top_k,
                            top_p=top_p,
                            min_p=min_p,
                            temperature=temperature,
                            seed=seed_val,
                            resources=resources,
                            batch_count=batch_count,
                            inter_segment_silence=inter_segment_silence_val,
                            duration_scale=duration_scale,
                        )
                        
                        # Unpack assuming 7 values or handle partial
                        if len(res) == 7:
                            concat_results, chunk_results, chunk_texts, segment_results, segment_texts, saved_files, all_detailed_segments = res
                        else:
                            # Fallback if function not updated yet
                            concat_results, chunk_results, chunk_texts, segment_results, segment_texts, saved_files = res
                            all_detailed_segments = []
        
                        outputs = []
        
                        # 1. Main Concatenated Audio
                        for i in range(max_batch):
                            if i < len(concat_results):
                                outputs.append(gr.update(value=concat_results[i], visible=True))
                            else:
                                outputs.append(gr.update(value=None, visible=False))
        
                        # 2. Saved Files
                        files_text = "\n".join(saved_files) if saved_files else ""
                        outputs.append(gr.update(value=files_text, visible=bool(saved_files)))
        
                        # 3. Build Initial State
                        new_state = []
                        for i in range(len(chunk_texts)):
                            chunk_text = chunk_texts[i]
                            chunk_audio = chunk_results[0][i] if (chunk_results and i < len(chunk_results[0])) else None
                            
                            segs_data = []
                            # Check if we have detailed segments
                            if i < len(all_detailed_segments):
                                for seg_txt, seg_aud in all_detailed_segments[i]:
                                    segs_data.append({
                                        "text": seg_txt,
                                        "audio": seg_aud,
                                        "candidates": []
                                    })
                            
                            new_state.append({
                                "full_text": chunk_text,
                                "full_audio": chunk_audio,
                                "segments": segs_data
                            })
        
                        # 4. Populate UI
                        for i in range(max_chunks):
                            if i < len(new_state):
                                cdata = new_state[i]
                                outputs.append(gr.update(visible=True)) # Group
                                outputs.append(gr.update(value=cdata["full_audio"])) # Chunk Audio
                                outputs.append(gr.update(value=cdata["full_text"])) # Chunk Text
                                
                                segs = cdata["segments"]
                                for j in range(max_segments_per_chunk):
                                    if j < len(segs):
                                        sdata = segs[j]
                                        outputs.append(gr.update(visible=True)) # Seg Group
                                        outputs.append(gr.update(value=sdata["text"]))
                                        outputs.append(gr.update(value=sdata["audio"]))
                                    else:
                                        outputs.append(gr.update(visible=False))
                                        outputs.append(gr.update(value=""))
                                        outputs.append(gr.update(value=None))
                                    
                                    # Hide candidates
                                    outputs.append(gr.update(visible=False))
                                    for _ in range(8): outputs.append(gr.update(value=None))
                                    outputs.append(gr.update(value=None))
                            else:
                                outputs.append(gr.update(visible=False))
                                outputs.append(gr.update(value=None))
                                outputs.append(gr.update(value=""))
                                for j in range(max_segments_per_chunk):
                                    outputs.append(gr.update(visible=False))
                                    outputs.append(gr.update(value=""))
                                    outputs.append(gr.update(value=None))
                                    outputs.append(gr.update(visible=False))
                                    for _ in range(8): outputs.append(gr.update(value=None))
                                    outputs.append(gr.update(value=None))
        
                        # 5. Hide Batch Outputs
                        for i in range(max_batch):
                            outputs.append(gr.update(value=None, visible=False))
                        
                        # 6. State
                        outputs.append(new_state)
        
                        return outputs
                    else:
                        # Non-segmented
                        results = run_inference(
                            reference_speech=reference_speech,
                            reference_text=reference_text,
                            target_text=target_text,
                            target_duration=dur,
                            top_k=top_k,
                            top_p=top_p,
                            min_p=min_p,
                            temperature=temperature,
                            seed=seed_val,
                            resources=resources,
                            batch_count=batch_count,
                            duration_scale=duration_scale,
                        )
                        
                        outputs = []
                        for _ in range(max_batch): outputs.append(gr.update(visible=False))
                        outputs.append(gr.update(visible=False))
                        
                        # Hide all chunk/segments UI
                        for i in range(max_chunks):
                            outputs.append(gr.update(visible=False))
                            outputs.append(gr.update(value=None))
                            outputs.append(gr.update(value=""))
                            for j in range(max_segments_per_chunk):
                                outputs.append(gr.update(visible=False))
                                outputs.append(gr.update(value=""))
                                outputs.append(gr.update(value=None))
                                outputs.append(gr.update(visible=False))
                                for _ in range(8): outputs.append(gr.update(value=None))
                                outputs.append(gr.update(value=None))
                        
                        for i in range(max_batch):
                            if i < len(results):
                                outputs.append(gr.update(value=results[i], visible=True))
                            else:
                                outputs.append(gr.update(value=None, visible=False))
                        
                        outputs.append([])
                        
                        return outputs
        
                # Construct call outputs list using the same order
                chunk_outputs_flat = []
                for i in range(max_chunks):
                    ui = chunk_ui_elements[i]
                    chunk_outputs_flat.append(ui["group"])
                    chunk_outputs_flat.append(ui["audio"])
                    chunk_outputs_flat.append(ui["text"])
                    
                    for j in range(max_segments_per_chunk):
                        sui = all_segment_ui[i][j]
                        chunk_outputs_flat.append(sui["group"])
                        chunk_outputs_flat.append(sui["text"])
                        chunk_outputs_flat.append(sui["audio"])
                        chunk_outputs_flat.append(sui["cand_group"])
                        chunk_outputs_flat.extend(sui["cand_audios"])
                        chunk_outputs_flat.append(sui["radio"])
        
                all_outputs = concatenated_audio_outputs + [saved_files_box] + chunk_outputs_flat + output_audios + [segments_state]
        
                generate_button.click(
                    fn=gradio_inference,
                    inputs=[
                        reference_speech_input,
                        reference_text_box,
                        target_text_box,
                        target_duration_box,
                        duration_scale_box,
                        top_k_box,
                        top_p_box,
                        min_p_box,
                        temperature_box,
                        seed_box,
                        batch_count_box,
                        enable_segmentation_box,
                        inter_segment_silence_box,
                    ],
                    outputs=all_outputs,
                )
        
        # Script Generation Handlers

        # Script Generation Handlers

        def parse_script_for_one_sided(script_text: str) -> str:
            """
            Extract only Speaker A's lines for One-Sided Manzai.
            Speaker A aliases: 西園寺, A, Shiori, S, 紫織
            """
            lines = script_text.strip().split('\n')
            extracted_lines = []
            
            # Speaker A aliases
            speaker_a_names = ["西園寺", "A", "Shiori", "S", "紫織"]
            
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # Regex to match "Speaker: Content"
                # Matches: "Name:", "Name：", "Name " (if clean enough?)
                # Sticking to colon separator as per standard generation
                match = re.search(r"^([^：:]+)[：:](.+)$", line)
                if match:
                    role = match.group(1).strip()
                    content = match.group(2).strip()
                    
                    # Check if role matches Speaker A
                    is_a = False
                    for name in speaker_a_names:
                        if name in role:
                            is_a = True
                            break
                    
                    if is_a:
                        # Clean content (quotes etc) - basic clean
                        content = content.replace('「', '').replace('」', '')
                        extracted_lines.append(content)
                        
            return "\n".join(extracted_lines)

        def on_generate_script(theme, characters, model="z-ai/glm-4.7"):
            real_key = os.getenv("OPENROUTER_API_KEY")
            if not real_key:
                yield "Error: OPENROUTER_API_KEY is missing."
                return

            yield "Generating Script (A & B)..."
            
            # Use shared generator logic
            # Note: generate_manzai_script returns the raw A/B script
            s1 = script_generator.generate_manzai_script(real_key, theme, characters, model)
            
            yield s1

        script_gen_btn.click(
            fn=on_generate_script,
            inputs=[theme_input, char_settings_input],
            outputs=[script_output]
        )

        def on_parse_and_transfer(text):
            if not text: return gr.update()
            
            # Parse A's lines
            parsed_text = parse_script_for_one_sided(text)
            
            if not parsed_text:
                return gr.update(value="[Error] No lines found for Speaker A (西園寺/A). Please check the script format.")
            
            print(f"[Info] Parsed and transferred {len(parsed_text)} chars.")
            return gr.update(value=parsed_text)

        # Re-purpose the transfer button
        script_transfer_btn.value = "Parse & Copy to TTS Input / パースしてTTS入力へコピー"
        script_transfer_btn.click(
            fn=on_parse_and_transfer,
            inputs=[script_output],
            outputs=[target_text_box]
        )

        def on_save_script(text):
            if not text: return "No text to save."
            path = script_generator.save_script_to_file(text)
            return f"Saved to {path}"

        script_save_btn.click(
            fn=on_save_script,
            inputs=[script_output],
            outputs=[script_save_status]
        )
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=server_port, share=share, debug=True)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gradio demo for HF T5GemmaVoice")
    parser.add_argument("--model_dir", type=str, default="./t5gemma_voice_hf", help="HF model directory or repo id")
    parser.add_argument("--xcodec2_model_name", type=str, default=None, help="Override xcodec2 model name from config")
    parser.add_argument("--xcodec2_sample_rate", type=int, default=None, help="Override xcodec2 sample rate from config")
    parser.add_argument("--cpu_whisper", action="store_true", help="Run Whisper transcription on CPU to save VRAM")
    parser.add_argument("--cpu_codec", action="store_true", help="Run XCodec2 tokenizer on CPU to save VRAM")
    parser.add_argument("--low_vram", action="store_true", help="Preset: cpu_whisper + cpu_codec")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile (faster startup, slower inference)")
    parser.add_argument("--max_batch", type=int, default=4, help="Maximum batch count for parallel generation")
    parser.add_argument("--max_segments", type=int, default=10, help="Maximum number of segments to display in UI")
    parser.add_argument(
        "--inter_segment_silence",
        type=float,
        default=6.0,
        help="Silence (seconds) inserted between sentence segments when segmentation is enabled",
    )
    args = parser.parse_args()

    if args.low_vram:
        args.cpu_whisper = True
        args.cpu_codec = True
        args.no_compile = True

    if args.cpu_whisper or args.low_vram:
        whisper_device = "cpu"
    elif torch.cuda.is_available():
        whisper_device = "cuda"
    elif torch.backends.mps.is_available():
        whisper_device = "mps"
    else:
        whisper_device = "cpu"

    resources = _load_resources(
        args.model_dir,
        args.xcodec2_model_name,
        args.xcodec2_sample_rate,
        use_torch_compile=not args.no_compile,
        cpu_codec=args.cpu_codec,
        whisper_device=whisper_device,
    )
    build_demo(
        resources=resources,
        server_port=args.port,
        share=args.share,
        max_batch=args.max_batch,
        max_segments=args.max_segments,
        inter_segment_silence=args.inter_segment_silence,
    )


if __name__ == "__main__":
    main()
