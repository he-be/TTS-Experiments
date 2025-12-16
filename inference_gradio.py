"""
Gradio demo for HF-format T5GemmaVoice checkpoints.

Usage:
    python inference_gradio.py --model_dir ./t5gemma_voice_hf --port 7860
"""

import argparse
import os
import random
from functools import lru_cache
from types import MethodType
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import whisper

from data.tokenizer import AudioTokenizer, tokenize_audio
from duration_estimator import estimate_duration
from inference_tts_utils import (
    inference_one_sample,
    normalize_text_with_lang,
    get_audio_info,
    get_sample_rate,
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

@lru_cache(maxsize=2)
def _get_whisper_model(device: str):
    """Cache Whisper model to avoid reloading on every inference."""
    print(f"[Info] Loading Whisper model (large-v3-turbo) on {device}...")
    return whisper.load_model("large-v3-turbo", device=device)


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Favor throughput on modern NVIDIA GPUs (TF32 for matmul/conv where applicable).
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
        print("[Info] No reference text; transcribing reference speech with Whisper (large-v3-turbo).")
        wh_model = _get_whisper_model(resources["whisper_device"])
        result = wh_model.transcribe(reference_speech)
        prefix_transcript = result["text"]
        print(f"[Info] Whisper transcription: {prefix_transcript}")
    else:
        prefix_transcript = reference_text

    # Language + normalization (Japanese only)
    lang = None if lang in {None, "", "none", "null"} else str(lang)
    target_text, lang_code = normalize_text_with_lang(target_text, lang)
    if prefix_transcript:
        prefix_transcript, _ = normalize_text_with_lang(prefix_transcript, lang_code)

    if target_duration is None:
        target_generation_length = estimate_duration(
            target_text=target_text,
            reference_speech=None if no_reference_audio else reference_speech,
            reference_transcript=None if no_reference_audio else prefix_transcript,
            target_lang=lang_code,
            reference_lang=lang_code,
        )
        print(f"[Info] target_duration not provided, estimated as {target_generation_length:.2f} seconds.")
    else:
        target_generation_length = float(target_duration)

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
) -> Tuple[list, list, list]:
    """
    Run segmented TTS inference: split text by sentences and generate each in parallel.

    Returns:
        Tuple of (concatenated_results, segment_results, segment_texts) where:
        - concatenated_results: List of (sample_rate, waveform) for full concatenated audio
        - segment_results: List of lists, each inner list contains segment audios for one batch
        - segment_texts: List of segment text strings
    """
    model = resources["model"]
    cfg = resources["cfg"]
    text_tokenizer = resources["text_tokenizer"]
    audio_tokenizer = resources["audio_tokenizer"]
    device = resources["device"]
    codec_audio_sr = resources["codec_audio_sr"]
    codec_sr = resources["codec_sr"]

    # Segment the text
    segments = segment_text_by_sentences(target_text)
    if not segments:
        segments = [target_text]

    print(f"[Info] Split text into {len(segments)} segments:")
    for i, seg in enumerate(segments):
        print(f"  Segment {i+1}: {seg[:50]}{'...' if len(seg) > 50 else ''}")

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
        wh_model = _get_whisper_model(resources["whisper_device"])
        result = wh_model.transcribe(reference_speech)
        prefix_transcript = result["text"]
        print(f"[Info] Whisper transcription: {prefix_transcript}")
    else:
        prefix_transcript = reference_text

    # Language + normalization
    lang = None if lang in {None, "", "none", "null"} else str(lang)
    _, lang_code = normalize_text_with_lang(target_text, lang)
    if prefix_transcript:
        prefix_transcript, _ = normalize_text_with_lang(prefix_transcript, lang_code)

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
        print(f"[Info] Segment duration estimate: {dur:.2f}s for '{seg[:30]}...'")

    # Prepare reference audio
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

    # Tokenize all segments
    x_sep_token = getattr(cfg, "x_sep_token", None)
    add_eos_token = getattr(cfg, "add_eos_to_text", 0)
    add_bos_token = getattr(cfg, "add_bos_to_text", 0)

    def encode_text_hf(text):
        if isinstance(text, list):
            text = " ".join(text)
        return text_tokenizer.encode(text.strip(), add_special_tokens=False)

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

    concatenated_results = []
    all_segment_results = []

    # Run batch_count iterations with different seeds
    base_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    seeds = [base_seed + i for i in range(batch_count)]

    for batch_idx, s in enumerate(seeds):
        print(f"\n[Info] Generating batch {batch_idx+1}/{batch_count} with seed {s}...")
        used_seed = seed_everything(s, deterministic=bool(int(os.getenv("T5GEMMA_DETERMINISTIC", "0"))))

        # Check if model has multi-text batch method
        target = model._orig_mod if hasattr(model, "_orig_mod") else model
        has_multi_text = hasattr(target, "inference_tts_batch_multi_text")

        if has_multi_text and len(segments) > 1:
            # Use parallel multi-text inference
            print(f"[Info] Using parallel multi-text inference for {len(segments)} segments")
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
                print(f"[Info] Generating segment {seg_idx+1}/{len(segments)}...")
                cf, gf = inference_one_sample(
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
                concat_frames.append(cf[2] if isinstance(cf, tuple) else cf)
                gen_frames.append(gf[3] if isinstance(gf, tuple) else gf)

        # Decode each segment's audio
        y_sep_token = getattr(cfg, "y_sep_token", None)
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

        segment_audios = []
        for i, gf in enumerate(gen_frames):
            gf_clean = strip_special(gf)
            gen_audio = audio_tokenizer.decode(gf_clean)
            gen_tensor = gen_audio[0].detach().cpu()
            if gen_tensor.ndim == 2 and gen_tensor.shape[0] == 1:
                gen_tensor = gen_tensor.squeeze(0)
            waveform = gen_tensor.numpy()
            segment_audios.append((codec_audio_sr, waveform))

        all_segment_results.append(segment_audios)

        # Concatenate all segments (also trims leading/trailing silence and prepends 0.5s silence)
        concat_audio = concatenate_audio_segments(segment_audios, silence_sec=inter_segment_silence)

        concatenated_results.append(concat_audio)
        print(f"[Info] Batch {batch_idx+1} complete: concatenated audio {concat_audio[1].shape[0]/concat_audio[0]:.2f}s")

    return concatenated_results, all_segment_results, segments


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo(
    resources,
    server_port: int,
    share: bool,
    max_batch: int = 4,
    max_segments: int = 10,
    inter_segment_silence: float = 0.05,
):
    description = (
        "Reference speech is optional. If provided without reference text, Whisper (large-v3-turbo) "
        "will auto-transcribe and use it as the prompt text.\n\n"
        "**Batch Generation**: Set batch count > 1 to generate multiple samples with different seeds.\n\n"
        "**Sentence Segmentation**: Enable to split long text by sentence delimiters (。！？!?.) and generate each segment in parallel."
    )

    with gr.Blocks() as demo:
        gr.Markdown("## T5Gemma-TTS (HF)")
        gr.Markdown(description)

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

        # Segment outputs in accordion
        segment_accordion = gr.Accordion("Individual Segments / 個別セグメント", open=False, visible=False)
        segment_text_outputs = []
        segment_audio_outputs = []
        with segment_accordion:
            for i in range(max_segments):
                with gr.Row():
                    text_out = gr.Textbox(
                        label=f"Segment {i+1} Text",
                        interactive=False,
                        visible=False,
                    )
                    audio_out = gr.Audio(
                        label=f"Segment {i+1} Audio",
                        type="numpy",
                        interactive=False,
                        visible=False,
                    )
                segment_text_outputs.append(text_out)
                segment_audio_outputs.append(audio_out)

        def toggle_segmentation_ui(enabled):
            """Toggle visibility of segmentation-related UI elements."""
            return [
                gr.update(visible=enabled),  # concat_row
                gr.update(visible=not enabled),  # batch_row
                gr.update(visible=enabled),  # segment_accordion
            ]

        enable_segmentation_box.change(
            fn=toggle_segmentation_ui,
            inputs=[enable_segmentation_box],
            outputs=[concat_row, batch_row, segment_accordion],
        )

        def gradio_inference(
            reference_speech,
            reference_text,
            target_text,
            target_duration,
            top_k,
            top_p,
            min_p,
            temperature,
            seed,
            batch_count,
            enable_segmentation,
        ):
            dur = None
            if str(target_duration).strip() not in {"", "None", "none"}:
                try:
                    dur = float(target_duration)
                except ValueError:
                    print(f"[Warning] Invalid duration value '{target_duration}', using auto-estimate")
            seed_val = None
            if str(seed).strip() not in {"", "None", "none"}:
                seed_val = int(float(seed))
            batch_count = int(batch_count)

            if enable_segmentation:
                # Segmented inference
                concat_results, segment_results, segment_texts = run_inference_segmented(
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
                    inter_segment_silence=inter_segment_silence,
                )

                outputs = []

                # Concatenated audio outputs
                for i in range(max_batch):
                    if i < len(concat_results):
                        sr, wav = concat_results[i]
                        outputs.append(gr.update(value=(sr, wav), visible=True))
                    else:
                        outputs.append(gr.update(value=None, visible=False))

                # Regular batch outputs (hidden in segmentation mode)
                for i in range(max_batch):
                    outputs.append(gr.update(value=None, visible=False))

                # Segment outputs (use first batch's segments)
                first_batch_segments = segment_results[0] if segment_results else []
                for i in range(max_segments):
                    if i < len(segment_texts):
                        outputs.append(gr.update(value=segment_texts[i], visible=True))
                        if i < len(first_batch_segments):
                            sr, wav = first_batch_segments[i]
                            outputs.append(gr.update(value=(sr, wav), visible=True))
                        else:
                            outputs.append(gr.update(value=None, visible=False))
                    else:
                        outputs.append(gr.update(value=None, visible=False))
                        outputs.append(gr.update(value=None, visible=False))

                return outputs
            else:
                # Non-segmented inference
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
                )

                outputs = []

                # Concatenated outputs (hidden in non-segmentation mode)
                for i in range(max_batch):
                    outputs.append(gr.update(value=None, visible=False))

                # Regular batch outputs
                for i in range(max_batch):
                    if i < len(results):
                        sr, wav = results[i]
                        outputs.append(gr.update(value=(sr, wav), visible=True))
                    else:
                        outputs.append(gr.update(value=None, visible=False))

                # Segment outputs (hidden in non-segmentation mode)
                for i in range(max_segments):
                    outputs.append(gr.update(value=None, visible=False))
                    outputs.append(gr.update(value=None, visible=False))

                return outputs

        # All outputs in order
        all_outputs = concatenated_audio_outputs + output_audios + segment_text_outputs + segment_audio_outputs

        generate_button.click(
            fn=gradio_inference,
            inputs=[
                reference_speech_input,
                reference_text_box,
                target_text_box,
                target_duration_box,
                top_k_box,
                top_p_box,
                min_p_box,
                temperature_box,
                seed_box,
                batch_count_box,
                enable_segmentation_box,
            ],
            outputs=all_outputs,
        )

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
        default=0.05,
        help="Silence (seconds) inserted between sentence segments when segmentation is enabled",
    )
    args = parser.parse_args()

    if args.low_vram:
        args.cpu_whisper = True
        args.cpu_codec = True
        args.no_compile = True

    whisper_device = "cpu" if (args.cpu_whisper or args.low_vram) else ("cuda" if torch.cuda.is_available() else "cpu")

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
