import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()

from script_generator import generate_manzai_script

# Configuration
API_KEY = os.environ.get("OPENROUTER_API_KEY")
REFERENCE_FILE = "texts/example_1226_1.txt"
# JUDGE_MODEL = "google/gemini-3-pro-preview" # Using a high reasoning model for judging
JUDGE_MODEL = "google/gemini-3-pro-preview"

def load_reference(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def call_judge(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Starttoaster/T5Gemma-TTS",
        "X-Title": "T5Gemma-TTS Script Judge",
    }
    data = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Judge call failed: {e}")
        return None

def main():
    if not API_KEY:
        print("Please set OPENROUTER_API_KEY environment variable.")
        return

    # 1. Generate New Script
    print("Generating new script...")
    new_script = generate_manzai_script(API_KEY, theme="「二手に分かれよう」提案、明らかに悪手なのになぜ採用されるのか")
    print("\n--- New Script ---\n")
    print(new_script)
    print("\n------------------\n")

    # 2. Load Reference
    reference_script = load_reference(REFERENCE_FILE)

    # 3. Construct Comparison Prompt
    prompt = f"""
You are a comedy script judge. Compare the following two scripts based on "Funniness", "Character Fidelity", and "Tempo".

**Character Settings**:
- A (Shiori): Serious, knowledgeable, gets dragged into B's pace.
- B (Poem): Natural airhead, lateral thinker, derails conversations with minor details.

**Reference Script (Ideal Example)**:
{reference_script[:2000]}... (Truncated for context)

**New Generated Script**:
{new_script}

**Task**:
1. Compare the "New Generated Script" against the style of the "Reference Script".
2. Does the New Script capture the "Derailment" logic correctly? (A tries to talk -> B interrupts with irrelevant detail -> A gets confused).
3. Is it funny?
4. **Verdict**: BETTER / EQUAL / WORSE than Reference in terms of style matching.
5. Provide specific feedback for the prompt engineer to improve the new script.
"""

    # 4. Get Judgment
    print("Asking Judge...")
    verdict = call_judge(prompt)
    print("\n--- Verdict ---\n")
    print(verdict)

    # Save Log
    with open("optimization_logs/comparison_log.txt", "w", encoding="utf-8") as f:
        f.write(f"NEW SCRIPT:\n{new_script}\n\nVERDICT:\n{verdict}")

if __name__ == "__main__":
    main()
