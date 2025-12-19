export CUDA_VISIBLE_DEVICES=1
export T5GEMMA_DETERMINISTIC=0
export T5GEMMA_EMPTY_CACHE=0
fuser -kvn tcp 7860
uv run inference_gradio.py \
    --model_dir ../models/Aratako/T5Gemma-TTS-2b-2b \
    --port 7860 --max_batch 64 --max_segments 32 --inter_segment_silence 0.4 --cpu_codec --cpu_whisper
