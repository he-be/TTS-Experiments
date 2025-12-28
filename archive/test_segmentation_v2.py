from inference_tts_utils import segment_text_by_sentences

test_cases = [
    # Case 1: Short lines (newline priority) - Expect \n preservation
    (
        "Hello world.\nHow are you?\nI am fine.",
        ["Hello world.\n", "How are you?\n", "I am fine."]
    ),
    # Case 2: Long line (adaptive split) - Last segment should have \n if input did
    (
        ("This is a very long line. " * 5) + "\n", 
        # Adaptive split usually preserves delimiters inside, 
        # and our fix should append \n to the last segment.
        # We don't check exact split here, but check the last char.
        "CHECK_LAST_NEWLINE"
    ),
    # Case 3: Mixed
    (
        "Short line.\n" + ("Long line part. " * 5) + "\nAnother short.",
        ["Short line.\n", "CHECK_LAST_NEWLINE_IN_GROUP", "Another short."]
    )
]

print("Running segmentation tests (v2.1 - Delimiter Preservation)...\n")

for i, (text, expected) in enumerate(test_cases):
    print(f"--- Case {i+1} ---")
    segments = segment_text_by_sentences(text, max_length=80, preserve_delimiter=True)
    
    print(f"Segments found: {len(segments)}")
    for j, seg in enumerate(segments):
        debug_seg = seg.replace('\n', '\\n')
        print(f"  [{j}] ({len(seg)} chars): {debug_seg}")
    
    if expected == "CHECK_LAST_NEWLINE":
        if segments[-1].endswith('\n'):
            print("[PASS] Last segment preserved newline.")
        else:
            print("[FAIL] Last segment missing newline.")
    elif isinstance(expected, list):
         # Exact match or partial check
         failed = False
         if len(segments) != len(expected) and "CHECK_" not in str(expected):
             print(f"[FAIL] Length mismatch. Expected {len(expected)}, got {len(segments)}")
             failed = True
         
         if not failed:
             for s_real, s_exp in zip(segments, expected):
                 if s_exp == "CHECK_LAST_NEWLINE_IN_GROUP":
                     continue # skip adaptive group check
                 if s_real != s_exp:
                     print(f"[FAIL] Expected '{s_exp.replace('\n','\\n')}', got '{s_real.replace('\n','\\n')}'")
                     failed = True
         if not failed:
             print("[PASS]")
    print()
