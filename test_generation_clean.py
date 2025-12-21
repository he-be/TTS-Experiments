import os
import script_generator
from dotenv import load_dotenv

# Load env
load_dotenv()

def test_full_flow():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY missing.")
        return

    # 1. Define Character Settings (as provided in texts/character.txt)
    # We strip the Markdown headers to just get the text body or pass it as is?
    # The user instruction said "Use this adopted as initial value", meaning the whole text probably.
    # But usually we want just the logic content. The Prompt uses it as is.
    with open("texts/character.txt", "r") as f:
        chars = f.read()

    theme = "「頭を冷やす」と言われて冷蔵庫に入ろうとする"
    
    print(f"--- 1. Generating Script for Theme: {theme} ---")
    print(f"--- Settings Length: {len(chars)} chars ---")
    
    
    # 2. Generate
    raw_script = script_generator.generate_manzai_script(api_key, theme, chars)
    
    if raw_script.startswith("Error"):
        print(raw_script)
        return

    print("\n--- Raw Output from LLM ---")
    print(raw_script)
    print("---------------------------")

    # 3. Parse/Clean
    print("\n--- 2. Testing Parsing/Cleaning ---")
    cleaned_script = script_generator.clean_script_for_speech(raw_script)
    
    print("--- Cleaned Output (Ready for A-side TTS) ---")
    print(cleaned_script)
    print("---------------------------------------------")
    
    # 4. Verification Heuristics
    # Check if names leaked
    if "西園寺" in cleaned_script or "紫織" in cleaned_script or "田中" in cleaned_script:
        print("[WARNING] Name leakage detected in cleaned script!")
    else:
        print("[SUCCESS] No full names detected in cleaned script.")
        
    if "：" in cleaned_script or ":" in cleaned_script:
         # Colon might be largely removed, but maybe some remain in text?
         # Japanese colon is common in text too? No, usually cleaning removes speaker colons.
         print("[INFO] Colons found in text (Check if they are part of dialogue).")

if __name__ == "__main__":
    test_full_flow()
