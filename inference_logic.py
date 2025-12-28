
import os
import random
import torch
import numpy as np
from typing import Optional, Tuple, List
from datetime import datetime

from data.tokenizer import tokenize_audio
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
    split_into_adaptive_chunks,
    save_audio
)

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
