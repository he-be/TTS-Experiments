import os
import random
import datetime
from script_generator import generate_manzai_script, clean_script_for_speech

# Configuration
THEMES_FILE = "texts/themes_1224.txt"
OUTPUT_DIR = "texts"
NUM_SAMPLES = 10
API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not API_KEY:
    # Try to load from a .env file if available, or just error out
    try:
        from dotenv import load_dotenv
        load_dotenv()
        API_KEY = os.environ.get("OPENROUTER_API_KEY")
    except ImportError:
        pass

if not API_KEY:
    print("Error: OPENROUTER_API_KEY environment variable is not set.")
    exit(1)

def load_themes(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    themes = []
    for line in lines:
        line = line.strip()
        # Skip empty lines and headers
        if not line or line.startswith('#'):
            continue
        # Remove leading numbers (e.g., "1. Theme")
        import re
        clean_line = re.sub(r'^[\d-]+\.\s*', '', line)
        if clean_line:
            themes.append(clean_line)
    return themes

def main():
    print(f"Loading themes from {THEMES_FILE}...")
    themes = load_themes(THEMES_FILE)
    
    if len(themes) < NUM_SAMPLES:
        print(f"Warning: Only found {len(themes)} themes. Using all of them.")
        selected_themes = themes
    else:
        selected_themes = random.sample(themes, NUM_SAMPLES)
        
    print(f"Selected {len(selected_themes)} themes for verification.")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"verification_result_{timestamp}.txt"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, theme in enumerate(selected_themes):
            print(f"[{i+1}/{len(selected_themes)}] Generating for theme: {theme}")
            
            script = generate_manzai_script(API_KEY, theme=theme)
            
            # Optional: Clean it a bit just to see what TTS would get
            # cleaned_script = clean_script_for_speech(script)
            
            f.write(f"--- Theme: {theme} ---\n")
            f.write(script)
            f.write("\n\n" + "="*80 + "\n\n")
            f.flush()
            os.fsync(f.fileno())
            
    print(f"Verification complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
