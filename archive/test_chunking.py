from inference_tts_utils import split_into_adaptive_chunks

# Create a text with many short lines of 10 chars each
# "Line 0001\n" -> 10 chars
lines = [f"Line {i:04d}\n" for i in range(50)]
full_text = "".join(lines)

print(f"Total lines: {len(lines)}")
print(f"Total chars: {len(full_text)}")
print("\n--- Test 1: Limit by Segment Count (max_segments=20) ---")
# Max size large enough to not trigger split
chunks = split_into_adaptive_chunks(full_text, max_size=1000, max_segments=20)
print(f"Chunks generated: {len(chunks)}")
for i, chunk in enumerate(chunks):
    # Count newlines to estimate segments
    seg_count = chunk.count('\n')
    print(f"  Chunk {i+1}: {len(chunk)} chars, approx {seg_count} segments")
    if seg_count > 20: 
        print(f"  [FAIL] Exceeded max_segments=20")
    else:
        print(f"  [PASS]")

print("\n--- Test 2: Limit by Char Size (max_size=100) ---")
# 10 lines = 100 chars. Should split every ~10 lines.
# But max_segments=20 (default) won't trigger first.
chunks_char = split_into_adaptive_chunks(full_text, max_size=105, max_segments=50) # 105 chars allow ~10 lines
print(f"Chunks generated: {len(chunks_char)}")
for i, chunk in enumerate(chunks_char):
    seg_count = chunk.count('\n')
    print(f"  Chunk {i+1}: {len(chunk)} chars, approx {seg_count} segments")
    if len(chunk) > 105:
         print(f"  [FAIL] Exceeded max_size=105")
    else:
         print(f"  [PASS]")
         
print("\n--- Test 3: Mixed Constraints ---")
# Limit segments to 5
chunks_mixed = split_into_adaptive_chunks(full_text, max_size=1000, max_segments=5)
print(f"Chunks generated: {len(chunks_mixed)}")
for i, chunk in enumerate(chunks_mixed):
    seg_count = chunk.count('\n')
    print(f"  Chunk {i+1}: {seg_count} segments")
    if seg_count > 5:
        print("  [FAIL]")
