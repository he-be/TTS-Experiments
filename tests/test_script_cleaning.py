import unittest
import sys
import os

# Add parent directory to path to import script_generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script_generator import clean_script_for_speech

class TestScriptCleaning(unittest.TestCase):
    def test_basic_cleaning(self):
        input_text = """
A: こんにちは。
B: さようなら。
A: いや、帰るなよ。
"""
        # Note: Our cleaner removes prefixes but keeps lines. 
        # Wait, the requirement says "B's lines should be deleted" in the Prompt, 
        # BUT the cleaner function logic I wrote just strips prefixes. 
        # The PROMPT is supposed to produce only A's lines in Step 2.
        # If the model outputs Step 1 (A&B) and Step 2 (A only), we strip Step 1.
        # If the input is ALREADY A's lines mostly, we just clean up.
        
        # Let's test the "Step 2" extraction heuristic
        input_text_full = """
### ステップ1
A: Hello
B: Hi

### ステップ2
A: Hello
A: Wait
"""
        expected = "Hello\nWait"
        self.assertEqual(clean_script_for_speech(input_text_full), expected)

    def test_remove_brackets(self):
        input_text = "A: こんにちは（笑）。"
        expected = "こんにちは。"
        self.assertEqual(clean_script_for_speech(input_text), expected)

    def test_replace_quotes(self):
        input_text = 'A: 彼は"天才"だと言った。'
        expected = "彼は「天才」だと言った。"
        self.assertEqual(clean_script_for_speech(input_text), expected)
        
    def test_remove_speaker_prefix_variations(self):
        input_text = """
A: Test 1
A：Test 2
"""
        expected = "Test 1\nTest 2"
        self.assertEqual(clean_script_for_speech(input_text), expected)

if __name__ == '__main__':
    unittest.main()
