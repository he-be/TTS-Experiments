import argparse
import logging
import os
import pickle
import random
import re
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import tqdm

from data.tokenizer import AudioTokenizer, tokenize_audio
from duration_estimator import detect_language


# ==== PyTorch 2.9+ audio backend compatibility helpers ====
# Starting with PyTorch 2.9, torchaudio internally depends on TorchCodec for
# audio loading/saving operations. In environments where TorchCodec is
# unavailable or incompatible (e.g., DGX Spark), these helpers provide a
# soundfile-based fallback.

def _torch_ge_29() -> bool:
    """Return True if PyTorch version is >= 2.9."""
    try:
        v = torch.__version__.split("+")[0]
        major, minor = map(int, v.split(".")[:2])
        return (major, minor) >= (2, 9)
    except Exception:
        return False


def get_audio_info(audio_path: str):
    """
    Get audio file info (sample rate).
    Returns an object with `samplerate` attribute (soundfile style).
    For PyTorch < 2.9, returns torchaudio.info result which has `sample_rate`.
    """
    if _torch_ge_29():
        import soundfile as sf
        return sf.info(audio_path)
    else:
        return torchaudio.info(audio_path)


def get_sample_rate(info) -> int:
    """Extract sample rate from audio info object (works for both backends)."""
    if hasattr(info, "samplerate"):
        return info.samplerate
    return info.sample_rate


def save_audio(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    """
    Save audio to file using appropriate backend.
    For PyTorch >= 2.9, uses soundfile to avoid TorchCodec dependency.
    """
    if _torch_ge_29():
        import soundfile as sf
        # Ensure waveform is 1D or 2D numpy array
        wav_np = waveform.squeeze().detach().cpu().numpy()
        sf.write(path, wav_np, sample_rate)
    else:
        torchaudio.save(path, waveform, sample_rate)

# Text normalization (only applied when language is Japanese)
_REPLACE_MAP = {
    r"\t": "",
    r"\[n\]": "",
    r" ": "",
    r"　": "",
    r"[;▼♀♂《》≪≫①②③④⑤⑥]": "",
    r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]": "",
    r"[\uff5e\u301C]": "ー",
    r"？": "?",
    r"！": "!",
    r"[●◯〇]": "○",
    r"♥": "♡",
}
_FULLWIDTH_ALPHA_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(
            list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
            list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
        )
    }
)
_HALFWIDTH_KATAKANA_CHARS = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
_FULLWIDTH_KATAKANA_CHARS = "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
_HALFWIDTH_KATAKANA_TO_FULLWIDTH = str.maketrans(
    _HALFWIDTH_KATAKANA_CHARS, _FULLWIDTH_KATAKANA_CHARS
)
_FULLWIDTH_DIGITS_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A))
    }
)


def _normalize_japanese_text(text: str) -> str:
    """Mirror dataset-side normalization; used only when lang=ja."""
    for pattern, replacement in _REPLACE_MAP.items():
        text = re.sub(pattern, replacement, text)

    text = text.translate(_FULLWIDTH_ALPHA_TO_HALFWIDTH)
    text = text.translate(_FULLWIDTH_DIGITS_TO_HALFWIDTH)
    text = text.translate(_HALFWIDTH_KATAKANA_TO_FULLWIDTH)

    # Collapse long ellipses
    text = re.sub(r"…{3,}", "……", text)
    return text


def normalize_text_with_lang(text: str, lang: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Normalize text if (and only if) the language is Japanese.
    Returns normalized_text and the resolved language code to avoid re-detecting downstream.
    """
    resolved_lang = lang.lower() if isinstance(lang, str) else None
    if not text:
        return text, resolved_lang
    if resolved_lang is None:
        resolved_lang = detect_language(text)
    if resolved_lang and resolved_lang.startswith("ja"):
        return _normalize_japanese_text(text), resolved_lang
    return text, resolved_lang


# ==== Sentence Segmentation Utilities ====

# Default delimiters for sentence segmentation (Japanese + English + newlines)
_SENTENCE_DELIMITERS = r"[。！？!?\.]+|\n+"


def segment_text_by_sentences(
    text: str,
    delimiters: str = _SENTENCE_DELIMITERS,
    min_length: int = 1,
    preserve_delimiter: bool = True,
) -> List[str]:
    """
    Split text into sentences by delimiters.

    Args:
        text: Input text to segment.
        delimiters: Regex pattern for sentence-ending punctuation.
        min_length: Minimum characters for a valid segment (after stripping).
        preserve_delimiter: Keep the delimiter at end of each segment.

    Returns:
        List of sentence segments. If no delimiters found, returns [text].
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    if preserve_delimiter:
        # Split using lookbehind to keep delimiter with the segment
        # This splits AFTER the delimiter pattern
        parts = re.split(f"({delimiters})", text)

        # Merge delimiter back to previous segment
        segments = []
        current = ""
        for i, part in enumerate(parts):
            if not part:
                continue
            if re.fullmatch(delimiters, part):
                # This is a delimiter, append to current segment
                current += part
            else:
                # This is text content
                if current:
                    segments.append(current)
                current = part

        # Don't forget the last segment
        if current:
            segments.append(current)
    else:
        # Simple split, delimiters are removed
        segments = re.split(delimiters, text)

    # Filter out empty or too-short segments
    segments = [s.strip() for s in segments if s and len(s.strip()) >= min_length]

    # If no valid segments, return original text
    if not segments:
        return [text]

    return segments


def concatenate_audio_segments(
    segments: List[Tuple[int, np.ndarray]],
    silence_sec: float = 0.15,
) -> Tuple[int, np.ndarray]:
    """
    Concatenate audio segments with silence gaps between them.

    Before concatenation, leading/trailing near-silence is trimmed from each segment.
    After concatenation, leading/trailing near-silence is trimmed from the full output,
    then a fixed 0.5s silence is prepended to the beginning.

    Args:
        segments: List of (sample_rate, waveform) tuples.
        silence_sec: Duration of silence between segments in seconds.

    Returns:
        (sample_rate, concatenated_waveform) tuple.

    Raises:
        ValueError: If segments list is empty or sample rates don't match.
    """
    if not segments:
        raise ValueError("No segments to concatenate")

    sample_rate = segments[0][0]
    silence_samples = int(silence_sec * sample_rate)
    leading_silence_samples = int(0.5 * sample_rate)

    def _trim_leading_trailing_silence(wav: np.ndarray, *, threshold_db: float = -45.0) -> np.ndarray:
        """
        Trim leading and trailing near-silence based on a peak-relative threshold.

        threshold_db is relative to the segment peak amplitude (e.g., -45 dB).
        """
        if wav.size == 0:
            return wav

        wav_abs = np.abs(wav.astype(np.float32, copy=False))
        peak = float(wav_abs.max(initial=0.0))
        if peak <= 0.0:
            return wav[:0]

        threshold = peak * (10.0 ** (threshold_db / 20.0))
        idx = np.flatnonzero(wav_abs > threshold)
        if idx.size == 0:
            return wav[:0]
        return wav[idx[0] : idx[-1] + 1]

    parts = []
    trimmed_segments: List[np.ndarray] = []
    for i, (sr, wav) in enumerate(segments):
        if sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: segment {i} has {sr}, expected {sample_rate}")

        # Ensure waveform is 1D
        if wav.ndim > 1:
            wav = wav.squeeze()

        wav = _trim_leading_trailing_silence(wav)
        if wav.size > 0:
            trimmed_segments.append(wav)

    if not trimmed_segments:
        dtype = segments[0][1].dtype
        return (sample_rate, np.zeros((leading_silence_samples,), dtype=dtype))

    for i, wav in enumerate(trimmed_segments):
        parts.append(wav)

        # Add silence between segments (not after the last one)
        if i < len(trimmed_segments) - 1 and silence_samples > 0:
            silence = np.zeros(silence_samples, dtype=wav.dtype)
            parts.append(silence)

    concatenated = np.concatenate(parts)
    concatenated = _trim_leading_trailing_silence(concatenated)

    if leading_silence_samples > 0:
        silence = np.zeros(leading_silence_samples, dtype=concatenated.dtype if concatenated.size else segments[0][1].dtype)
        concatenated = np.concatenate([silence, concatenated])
    return (sample_rate, concatenated)


# this script only works for the musicgen architecture
def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest_fn", type=str, default="path/to/eval_metadata_file")
    parser.add_argument("--audio_root", type=str, default="path/to/audio_folder")
    parser.add_argument("--exp_dir", type=str, default="path/to/model_folder")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--codec_audio_sr", type=int, default=16000, help='the sample rate of audio that the codec is trained for')
    parser.add_argument("--codec_sr", type=int, default=50, help='the sample rate of the codec codes')
    parser.add_argument("--top_k", type=int, default=0, help="sampling param")
    parser.add_argument("--top_p", type=float, default=0.8, help="sampling param")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling param")
    parser.add_argument("--min_p", type=float, default=0.0, help="sampling param; 0 disables min_p filter")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--signature", type=str, default=None, help="xcodec2 model id (HF repo) if you want to override the default")
    parser.add_argument("--crop_concat", type=int, default=0)
    parser.add_argument("--stop_repetition", type=int, default=-1, help="used for inference, when the number of consecutive repetition of a token is bigger than this, stop it")
    parser.add_argument("--sample_batch_size", type=int, default=1, help="batch size for sampling, NOTE that it's not running inference for several samples, but duplicate one input sample batch_size times, and during inference, we only return the shortest generation")
    parser.add_argument("--silence_tokens", type=str, default="[]", help="silence token ids (depends on the codec vocab)")
    return parser.parse_args()


@torch.no_grad()
def inference_one_sample(
    model,
    model_args,
    text_tokenizer,
    audio_tokenizer,
    audio_fn,
    target_text,
    lang,
    device,
    decode_config,
    prompt_end_frame,
    target_generation_length,
    prefix_transcript=None,
    quiet=False,
    repeat_prompt=0,
    multi_trial=None,
    return_frames=False,
):
    multi_trial = multi_trial or []

    # XCodec2 always uses a single codebook.
    if int(getattr(model_args, "n_codebooks", 1)) != 1:
        raise ValueError("XCodec2 backend supports only n_codebooks=1.")
    n_codebooks = 1
    empty_token = int(getattr(model_args, "empty_token", 0))
    codec_sr = int(decode_config["codec_sr"])
    y_sep_token = getattr(model_args, "y_sep_token", None)
    x_sep_token = getattr(model_args, "x_sep_token", None)
    eos_token = getattr(model_args, "eos", getattr(model_args, "eog", None))
    add_eos_token = getattr(model_args, "add_eos_to_text", 0)
    add_bos_token = getattr(model_args, "add_bos_to_text", 0)

    silence_tokens = decode_config["silence_tokens"]
    if isinstance(silence_tokens, str):
        silence_tokens = eval(silence_tokens)

    has_reference_audio = (
        audio_fn is not None and str(audio_fn).lower() not in {"", "none", "null"}
    )

    if has_reference_audio:
        encoded_frames = tokenize_audio(
            audio_tokenizer,
            audio_fn,
            offset=0,
            num_frames=prompt_end_frame if prompt_end_frame > 0 else -1,
        )
    else:
        # No reference: start with zero-length prompt; BOS (empty_token) will be added in the model.
        encoded_frames = torch.empty(
            (1, n_codebooks, 0),
            dtype=torch.long,
            device=audio_tokenizer.device,
        )

    if encoded_frames.ndim == 2:
        # Allow [T, 1] or [1, T] inputs for convenience.
        encoded_frames = encoded_frames.unsqueeze(0)
    if encoded_frames.ndim != 3 or encoded_frames.shape[0] != 1:
        raise ValueError(f"Unexpected prompt shape {encoded_frames.shape}")
    # Normalize to [B, 1, frames]
    if encoded_frames.shape[2] == 1:
        encoded_frames = encoded_frames.transpose(1, 2).contiguous()
    if encoded_frames.shape[1] != 1:
        raise ValueError(f"Expected a single codebook axis, got shape {encoded_frames.shape}")

    single_encoded_frames = encoded_frames.clone()

    # VoiceStar support removed; keep variable for backward compatibility.
    effective_delay_inc = 0
    if isinstance(repeat_prompt, int) and repeat_prompt > 0:
        cur_repeat_prompt = repeat_prompt
        while cur_repeat_prompt > 0:
            encoded_frames = torch.cat([encoded_frames, single_encoded_frames], dim=2)
            cur_repeat_prompt -= 1
    elif isinstance(repeat_prompt, str) and repeat_prompt.lower() == "max":
        repeat_prompt = 0
        while (
            encoded_frames.shape[2]
            + codec_sr * target_generation_length
            + effective_delay_inc
            + single_encoded_frames.shape[2]
            < model_args.audio_max_length * codec_sr
        ):
            encoded_frames = torch.cat([encoded_frames, single_encoded_frames], dim=2)
            repeat_prompt += 1

    # Only insert y_sep when we actually have a reference prompt.
    if y_sep_token is not None and has_reference_audio and encoded_frames.shape[2] > 0:
        encoded_frames = torch.cat(
            [
                encoded_frames,
                torch.full(
                    (1, n_codebooks, 1),
                    y_sep_token,
                    dtype=torch.long,
                    device=encoded_frames.device,
                ),
            ],
            dim=2,
        )

    original_audio = encoded_frames.transpose(2, 1).contiguous()  # [B, T, K]
    prompt_frames = original_audio.shape[1]

    # ---- Text Processing (text-only) ----
    # Normalize Japanese text when requested or detected; reuse lang to avoid double detection
    target_text, lang = normalize_text_with_lang(target_text, lang)
    if prefix_transcript:
        prefix_transcript, _ = normalize_text_with_lang(prefix_transcript, lang)

    def encode_text_hf(text):
        if isinstance(text, list):
            text = " ".join(text)
        return text_tokenizer.encode(text.strip(), add_special_tokens=False)

    text_tokens = encode_text_hf(target_text)
    if prefix_transcript:
        prefix_tokens = encode_text_hf(prefix_transcript)
        if x_sep_token is not None:
            text_tokens = prefix_tokens + [x_sep_token] + text_tokens
        else:
            text_tokens = prefix_tokens + text_tokens


    if add_eos_token:
        text_tokens.append(add_eos_token)
    if add_bos_token:
        text_tokens = [add_bos_token] + text_tokens

    text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
    text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])

    if not quiet:
        logging.info(
            f"original audio length: {original_audio.shape[1]} codec frames, "
            f"which is {original_audio.shape[1] / codec_sr:.2f} sec."
        )

    if getattr(model_args, "parallel_pattern", 0) != 0:
        tgt_y_lens = torch.LongTensor([int(original_audio.shape[1] + codec_sr * target_generation_length + 2)])
    else:
        tgt_y_lens = torch.LongTensor(
            [int(original_audio.shape[1] + codec_sr * target_generation_length + effective_delay_inc)]
        )

    sample_batch_size = int(decode_config.get("sample_batch_size", 1))
    stime = time.time()
    assert multi_trial == []

    batch_mode = sample_batch_size > 1 and hasattr(model, "inference_tts_batch")
    if batch_mode:
        if not quiet:
            logging.info("running inference with batch size %d (parallel on one GPU)", sample_batch_size)
        concat_frames, gen_frames = model.inference_tts_batch(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            original_audio.to(device),
            tgt_y_lens=tgt_y_lens.to(device),
            num_samples=sample_batch_size,
            top_k=decode_config["top_k"],
            top_p=decode_config["top_p"],
            min_p=decode_config["min_p"],
            temperature=decode_config["temperature"],
            stop_repetition=decode_config["stop_repetition"],
            silence_tokens=silence_tokens,
            prompt_frames=prompt_frames,
        )
    else:
        if sample_batch_size > 1:
            raise ValueError("Model does not support batched inference_tts_batch; set sample_batch_size=1.")
        if not quiet:
            logging.info("running inference with batch size 1")
        concat_frames, gen_frames = model.inference_tts(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            original_audio.to(device),
            tgt_y_lens=tgt_y_lens.to(device),
            top_k=decode_config["top_k"],
            top_p=decode_config["top_p"],
            min_p=decode_config["min_p"],
            temperature=decode_config["temperature"],
            stop_repetition=decode_config["stop_repetition"],
            silence_tokens=silence_tokens,
            prompt_frames=prompt_frames,
        )

    inference_time = time.time() - stime
    if isinstance(gen_frames, (list, tuple)):
        per_sample_tokens = [int(g.shape[-1]) for g in gen_frames]
        total_generated_tokens = int(sum(per_sample_tokens))
        avg_generated_tokens = float(total_generated_tokens) / max(1, len(per_sample_tokens))
        tokens_per_sec = total_generated_tokens / inference_time if inference_time > 0 else 0.0
        avg_audio_duration = avg_generated_tokens / codec_sr
        real_time_factor = avg_audio_duration / inference_time if inference_time > 0 else 0.0
        if not quiet:
            logging.info(f"inference on batch take: {inference_time:.4f} sec.")
            logging.info(
                f"generated batch sizes: {per_sample_tokens} tokens (avg {avg_generated_tokens:.1f})."
            )
        print(
            f"[Speed] {tokens_per_sec:.2f} tokens/s (total) | RTF: {real_time_factor:.2f}x (avg) | "
            f"Generated {total_generated_tokens} tokens across {len(per_sample_tokens)} samples in {inference_time:.2f}s"
        )
    else:
        num_generated_tokens = int(gen_frames.shape[-1])
        tokens_per_sec = num_generated_tokens / inference_time if inference_time > 0 else 0.0
        audio_duration = num_generated_tokens / codec_sr
        real_time_factor = audio_duration / inference_time if inference_time > 0 else 0.0

        if not quiet:
            logging.info(f"inference on one sample take: {inference_time:.4f} sec.")
            logging.info(
                f"generated encoded_frames.shape: {gen_frames.shape}, "
                f"which is {audio_duration:.2f} sec."
            )
        print(
            f"[Speed] {tokens_per_sec:.2f} tokens/s | RTF: {real_time_factor:.2f}x | "
            f"Generated {num_generated_tokens} tokens in {inference_time:.2f}s"
        )

    def _strip_sep_and_eos(frames: torch.Tensor, sep_token: Optional[int], eos_token: Optional[int]) -> torch.Tensor:
        # keep all by default, then drop sep/eos if provided
        mask = torch.ones_like(frames, dtype=torch.bool)
        if sep_token is not None:
            mask &= frames.ne(sep_token)
        if eos_token is not None:
            mask &= frames.ne(eos_token)
        if mask.all():
            return frames
        keep_counts = mask.sum(dim=2)
        if not torch.all(keep_counts.eq(keep_counts[..., :1])):
            min_len = int(keep_counts.min().item())
            cleaned_batches = []
            for b in range(frames.shape[0]):
                per_codebook = []
                for k in range(frames.shape[1]):
                    vals = frames[b, k][mask[b, k]]
                    if vals.numel() < min_len:
                        pad_val = sep_token if sep_token is not None else 0
                        vals = torch.nn.functional.pad(vals, (0, min_len - vals.numel()), value=pad_val)
                        vals = vals[:min_len]
                    else:
                        vals = vals[:min_len]
                    per_codebook.append(vals.unsqueeze(0))
                cleaned_batches.append(torch.cat(per_codebook, dim=0).unsqueeze(0))
            return torch.cat(cleaned_batches, dim=0).to(frames.device)
        new_len = int(keep_counts[0, 0].item())
        cleaned_vals = frames[mask]
        return cleaned_vals.view(frames.size(0), frames.size(1), new_len)

    # Support batched outputs (lists) and single outputs (tensors)
    concat_frames_list = concat_frames if isinstance(concat_frames, (list, tuple)) else [concat_frames]
    gen_frames_list = gen_frames if isinstance(gen_frames, (list, tuple)) else [gen_frames]

    cleaned_concat_frames = []
    cleaned_gen_frames = []
    concat_samples = []
    gen_samples = []

    for cf, gf in zip(concat_frames_list, gen_frames_list):
        cf_clean = _strip_sep_and_eos(cf, y_sep_token, eos_token)
        gf_clean = _strip_sep_and_eos(gf, y_sep_token, eos_token)
        cleaned_concat_frames.append(cf_clean)
        cleaned_gen_frames.append(gf_clean)

        concat_sample = None
        if has_reference_audio:
            try:
                concat_sample = audio_tokenizer.decode(cf_clean)
            except Exception as exc:
                logging.warning(f"Failed to decode concatenated prompt audio: {exc}")

        gen_sample = audio_tokenizer.decode(gf_clean)

        if concat_sample is None:
            concat_sample = gen_sample

        concat_samples.append(concat_sample)
        gen_samples.append(gen_sample)

    # Clearing the CUDA cache forces synchronization and often hurts latency.
    # Keep it opt-in for low-VRAM environments.
    if torch.cuda.is_available() and os.getenv("T5GEMMA_EMPTY_CACHE", "0") not in {"0", "", "false", "False"}:
        torch.cuda.empty_cache()

    if return_frames:
        concat_cpu = [cf.detach().cpu() for cf in cleaned_concat_frames]
        gen_cpu = [gf.detach().cpu() for gf in cleaned_gen_frames]
        if len(concat_cpu) == 1:
            return concat_samples[0], gen_samples[0], concat_cpu[0], gen_cpu[0]
        return concat_samples, gen_samples, concat_cpu, gen_cpu

    if len(concat_samples) == 1:
        return concat_samples[0], gen_samples[0]
    return concat_samples, gen_samples
