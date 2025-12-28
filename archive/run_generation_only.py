import os
from script_generator import generate_manzai_script
import time

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("OPENROUTER_API_KEY")
    theme = "都会のネズミと田舎のネズミ"
    model = "z-ai/glm-4.7"

    print(f"Testing generation with model: {model}")
    start = time.time()
    try:
        script = generate_manzai_script(api_key, theme=theme, model=model)
        duration = time.time() - start
        print(f"Generation took {duration:.2f}s")
        print("\n--- Generated Script ---\n")
        print(script)
    except Exception as e:
        print(f"Generation failed: {e}")

if __name__ == "__main__":
    main()
