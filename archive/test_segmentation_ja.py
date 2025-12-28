from inference_tts_utils import segment_text_by_sentences
import os

def load_text_section(filepath, start_line, end_line):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return ""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 0-indexed slicing vs 1-indexed input
    return "".join(lines[start_line-1:end_line])

target_file = "texts/1227_1.txt"
# Text appears to be lines 2 to 23 (inclusive) based on file content
# Line 1 is comment. Line 2 starts text. Line 23 ends text.
print(f"Loading {target_file} lines 2-23...")
input_text = load_text_section(target_file, 2, 23)

print("--- Input Text Preview ---")
print(input_text[:100] + "...")
print("--------------------------")

print("\nRunning segmentation (max_length=80, preserve_delimiter=True)...")
segments = segment_text_by_sentences(input_text, max_length=80, preserve_delimiter=True)

print(f"\nTotal Segments: {len(segments)}")
for i, seg in enumerate(segments):
    # visualize newline
    vis_seg = seg.replace('\n', '\\n')
    print(f"[{i+1:02d}] ({len(seg)} chars): {vis_seg}")
