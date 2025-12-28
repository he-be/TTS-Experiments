"""
Gradio demo for HF-format T5GemmaVoice checkpoints.

Usage:
    python inference_gradio.py --model_dir ./t5gemma_voice_hf --port 7860
"""

import argparse
import os
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Trying to convert audio automatically.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch_dtype.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*You are using a model of type xcodec2.*")

import re
import random
from functools import lru_cache
from types import MethodType
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import script_generator


from data.tokenizer import AudioTokenizer
from inference_logic import (
    run_inference,
    run_inference_segmented,
    adaptive_duration_scale,
    seed_everything,
    estimate_duration,
)
from inference_tts_utils import (
    get_audio_info,
    get_sample_rate,
    concatenate_audio_segments,
)

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


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
