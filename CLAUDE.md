# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

T5Gemma-TTS is a multilingual Text-to-Speech model based on an Encoder-Decoder LLM architecture. It uses Google's T5Gemma as the backbone with custom audio generation layers and Progress-Monitoring RoPE (PM-RoPE) for cross-attention alignment.

## Key Commands

### Inference (HuggingFace format)
```bash
python inference_commandline_hf.py \
    --model_dir Aratako/T5Gemma-TTS-2b-2b \
    --target_text "Hello, this is a test." \
    --reference_speech path/to/reference.wav  # optional for voice cloning
```

### Inference (Gradio Web UI)
```bash
python inference_gradio.py --model_dir Aratako/T5Gemma-TTS-2b-2b --port 7860
```

### Training
```bash
# Training from scratch (multi-GPU)
NUM_GPUS=8 examples/training/t5gemma_2b-2b.sh

# Full fine-tuning
NUM_GPUS=8 examples/training/t5gemma_2b-2b-ft.sh

# LoRA fine-tuning (single GPU)
NUM_GPUS=1 examples/training/t5gemma_2b-2b-ft-lora.sh
```

### Model Export
```bash
# Convert .pth to HuggingFace format
python scripts/export_t5gemma_voice_hf.py --ckpt trained.pth --out t5gemma_voice_hf

# Export LoRA checkpoint
python scripts/export_t5gemma_voice_hf_lora.py --ckpt lora.pth --out t5gemma_merged
```

## Architecture

### Model Pipeline
1. **Text Input**: Processed via HuggingFace tokenizer (T5Gemma's native tokenizer)
2. **T5Gemma Encoder**: Encodes text into hidden representations
3. **PM-RoPE Cross-Attention**: Custom decoder cross-attention with Progress-Monitoring Rotary Position Embeddings for duration-aware generation
4. **Audio Token Prediction**: Linear prediction heads output audio codec tokens
5. **XCodec2 Decoder**: Converts audio tokens back to waveforms

### Key Components

- **`models/t5gemma.py`**: `T5GemmaVoiceModel` - main model class with `PMCrossAttention` and `PMDecoderLayer` for PM-RoPE
- **`steps/trainer.py`**: `Trainer` class handling distributed training, validation, logging (WandB/TensorBoard)
- **`steps/optim.py`**: `ScaledAdam` optimizer and `Eden` learning rate scheduler
- **`data/combined_dataset.py`**: Dataset loader supporting multiple datasets, neighbor prompts for voice cloning, and time stretching
- **`hf_export/`**: HuggingFace-compatible model wrappers for inference

### Audio Tokenization

Uses XCodec2 (single codebook) exclusively. Special tokens:
- `AUDIO_VOCAB_SIZE` (65536 for anime variant, 2048 for original): empty token
- `+1`: EOG (end of generation)
- `+2`: audio pad token
- `+3`: EOS
- `+4`: Y separator token

## Training Configuration

Key parameters in training scripts:
- `--t5gemma_model_name`: Base model (e.g., `google/t5gemma-2b-2b-ul2`)
- `--xcodec2_model_name`: Audio codec (`NandemoGHS/Anime-XCodec2-44.1kHz-v2` for Japanese, `HKUSTAudio/xcodec2` for English/Chinese)
- `--optimizer_name ScaledAdam`: Use with Eden scheduler (default lr: 0.035)
- `--use_lora 1`: Enable LoRA adapters
- `--t5_gradient_checkpointing 1`: Enable to reduce VRAM
- `--prune_text_modules 2`: Drop unused text decoder modules to save memory
- `--neighbor_prompt_prob`: Probability of using neighbor audio as voice prompt (0.5 typical)
- `--progress_scale 2000`: PM-RoPE scale factor

## Data Format

Training data structure:
```
dataset_root/
├── text/              # Raw text transcripts (.txt)
├── xcodec2_1cb/       # Pre-encoded audio tokens (.txt, space-separated)
├── manifest_final/    # train.txt, valid.txt (filename\tlength pairs)
└── neighbors/         # Neighbor info for voice prompting
```

## Memory Optimization

For low VRAM environments:
- `--cpu_codec`: Run XCodec2 on CPU (~3.5GB savings)
- `--cpu_whisper`: Run Whisper on CPU (~5GB savings)
- `--low_vram`: Preset enabling both + disabling torch.compile
- `--t5_gradient_checkpointing 1`: For training
