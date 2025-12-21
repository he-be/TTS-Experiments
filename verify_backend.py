import os
import script_generator
from dotenv import load_dotenv

load_dotenv()

def verify_backend():
    print("--- Testing Theme Generation ---")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: API Key missing in env")
        return

    themes = script_generator.generate_themes(api_key)
    print(f"Generated {len(themes)} themes.")
    for t in themes:
        print(f"- {t}")
        
    if not themes:
        print("Failed to generate themes")
        return

    selected_theme = themes[0]
    print(f"\n--- Testing Script Generation for theme: {selected_theme} ---")
    
    chars = "A: 理屈っぽい大学生。言葉の正確さにこだわる。\nB (不可視): 言葉を全て字義通りに受け取る、空気の読めない友人。"
    
    script = script_generator.generate_manzai_script(api_key, selected_theme, chars)
    print("--- Raw Script ---")
    print(script[:500] + "..." if len(script) > 500 else script)
    
    print("\n--- Cleaning Script ---")
    cleaned = script_generator.clean_script_for_speech(script)
    print(cleaned)
    
    # Save it
    path = script_generator.save_script_to_file(cleaned)
    print(f"\nSaved to {path}")

if __name__ == "__main__":
    verify_backend()
