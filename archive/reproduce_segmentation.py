
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from inference_tts_utils import segment_text_by_sentences

def load_text(filepath):
    """Load the raw text from the file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract only the "Original Script" part for testing (lines 2-22 in the file based on view_file)
    # Or just use the whole file content but split by newlines? 
    # The user's file has 3 sections: Original, Ideal, Current.
    # We want to process the "Original" text using the function and see if it matches "Current" (baseline)
    # and then "Ideal" (goal).
    
    lines = content.splitlines()
    # Lines 1-22 are the original script part (index 1 to 21)
    # Actually let's just manually extract the lines for the original script based on the view_file output
    # Lines 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 contain the text lines.
    
    original_lines = []
    in_original_section = False
    
    for line in lines:
        if "#元のスクリプト" in line:
            in_original_section = True
            continue
        if "#理想的なセグメント分割" in line:
            in_original_section = False
            break
            
        if in_original_section and line.strip():
            original_lines.append(line.strip())
            
    return original_lines

def run_test():
    filepath = "texts/1227_1.txt"
    input_lines = load_text(filepath)
    
    print(f"Loaded {len(input_lines)} lines from original script section.")
    
    print("\n--- Reproduction Results ---")
    
    total_segments = 0
    all_segments = []
    
    for i, line in enumerate(input_lines):
        # We need to emulate what split_into_adaptive_chunks does essentially, 
        # but the core logic is in segment_text_by_sentences which handles the splitting of a single chunk (paragraph).
        # In this specific case, each line in the text file acts like a turn/paragraph.
        
        segments = segment_text_by_sentences(
            line, 
            min_length=6, 
            max_length=80
        )
        
        print(f"\n[Line {i+1}] Original: {line[:50]}...")
        for j, seg in enumerate(segments):
            print(f"  Segment {j+1}: {seg} ({len(seg)} chars)")
            all_segments.append(seg)
        
        total_segments += len(segments)

    print(f"\nTotal Segments: {total_segments}")
    return all_segments

if __name__ == "__main__":
    run_test()
