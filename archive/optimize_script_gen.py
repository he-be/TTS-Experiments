import os
import requests
import json
import random
import glob
from script_generator import generate_manzai_script, clean_script_for_speech, load_prompt

# --- Logging ---
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# --- Configuration ---
# STRICT USER REQUEST: Use these exact model IDs.
TARGET_MODELS = [
    "z-ai/glm-4.7",
    "x-ai/grok-4.1-fast", 
    "deepseek/deepseek-v3.2"
]
MODEL_JUDGE = "google/gemini-3-pro-preview"

# API Config
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        API_KEY = os.environ.get("OPENROUTER_API_KEY")
    except ImportError:
        pass

if not API_KEY:
    print("[Error] OPENROUTER_API_KEY not found in env or .env")

# File Paths
GOLD_SAMPLE_PATH = "texts/1219_2.txt"
PROMPT_CHAR_PATH = "prompts/character_settings.txt"
PROMPT_SYS_PATH = "prompts/system_instruction.txt"
EVAL_LOG_DIR = "optimization_logs"

# --- Utils ---

def call_llm(messages, model, max_tokens=4000):
    if not API_KEY:
        print("[Error] No API Key")
        return None
        
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Starttoaster/T5Gemma-TTS",
        "X-Title": "T5Gemma-TTS Optimizer",
    }
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=300)
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return None
    except Exception as e:
        print(f"[Error] API Call failed: {e}")
        if 'response' in locals():
            print(f"[Error Details] Status: {response.status_code}, Body: {response.text}")
        return None

def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_log(filename, content):
    os.makedirs(EVAL_LOG_DIR, exist_ok=True)
    path = os.path.join(EVAL_LOG_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# --- Optimization Logic ---

class Optimizer:
    def __init__(self):
        self.gold_sample = load_file(GOLD_SAMPLE_PATH)
        self.character_prompts = load_file(PROMPT_CHAR_PATH)
        self.system_prompt_template = load_file(PROMPT_SYS_PATH) # Assuming this is a template? Or just text.
        # CURRENTLY system_prompt.txt is the FULL instruction. 
        # But script_generator.py expects it to format {character_settings} and {theme}.
        # We need to make sure the external TXT file has the placeholders if we use script_generator logic.
        # Actually generate_manzai_script inside script_generator.py handles the formatting.
        
    def verify_prompt_objectivity(self, character_settings: str, system_instruction_template: str):
        """
        Uses LLM to verify if the prompts are 'Overfitted' to a specific scenario (e.g. including specific proper nouns
        that should be generated) or if they are 'Robust' and generalizable.
        """
        logger.info("Running Prompt Objectivity Check...")
        
        check_prompt = f"""
You are a Prompt Engineer Quality Control Agent.
Your job is to strictly evaluate the following "System Prompt" and "Character Settings" for **Overfitting** and **Generalizability**.

Input Prompts:
---
[Character Settings]
{character_settings}

[System Instruction]
{system_instruction_template}
---

Evaluation Criteria:
1.  **NO Hardcoded Solutions**: Does the prompt contain specific proper nouns (e.g., "Hachioji", "Misaki", "Don Quijote") that are likely "copied" from a specific gold standard?
2.  **Generalizability**: Can this prompt work for ANY theme (e.g., "Peach Boy", "Space War", "Office Politics") without modification?
3.  **Abstract vs Concrete**: Does the prompt instruct *behavior* (e.g., "Use specific details") rather than providing the details itself?

Output:
If the prompts are CLEAN and ROBUST, output: "PASS"
If the prompts likely contain hardcoded specifics or overfitting, output: "FAIL: [Reason]"
"""
        messages = [{"role": "user", "content": check_prompt}] # Corrected messages format
        response = call_llm(messages, MODEL_JUDGE) # Corrected call_llm arguments
        
        if response and "PASS" in response: # Added response check
            logger.info("Prompt Objectivity Check: PASSED")
            return True
        else:
            logger.error(f"Prompt Objectivity Check: FAILED\nReason: {response}")
            return False

    def run_baseline_check(self, theme="都会のネズミと田舎のネズミ"):
        """Run generation and evaluate."""
        # Load current prompts
        char_settings = load_file(PROMPT_CHAR_PATH) # Used PROMPT_CHAR_PATH
        sys_instr = load_file(PROMPT_SYS_PATH) # Used PROMPT_SYS_PATH
        
        # 1. Objectivity Check
        if not self.verify_prompt_objectivity(char_settings, sys_instr):
            print("!!! PROMPTS REJECTED BY OBJECTIVITY CHECK !!!")
            print("Please generalize your prompts and remove hardcoded solutions.")
            return None # Return None if check fails

        # 2. Generation
        logger.info(f"Generating baseline for theme: {theme} using {MODEL_GEN}")
        # We rely on script_generator.py which now loads the files from prompts/ dir by default.
        # But to be safe/explicit, we can read them here and pass them if we modify prompts dynamically.
        # For baseline, we just run defaults.
        baseline_script = generate_manzai_script(API_KEY, theme=theme, model=MODEL_GEN)

        if not baseline_script:
            print("Failed to generate baseline.")
            return None

        evaluation = self.judge_against_gold(baseline_script) # Changed to self.judge_against_gold
        
        # Save results
        log_content = f"""# Baseline Evaluation Report
## Theme
{theme}

## Generated Script (Baseline)
{baseline_script}

## Judge Evaluation (vs Gold Standard)
{evaluation}
"""
        saved_path = save_log("baseline_eval.md", log_content)
        print(f"Baseline evaluation saved to {saved_path}")
        return baseline_script # Return the generated script

    def load_random_theme(self):
        """Loads a random theme from the themes file."""
        theme_path = "texts/themes_1224.txt"
        with open(theme_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        valid_themes = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and headers (start with #)
            if not line or line.startswith("#"):
                continue
            # Remove numbering (e.g. "1. title")
            parts = line.split(" ", 1)
            if len(parts) > 1 and parts[0].replace(".", "").isdigit():
                valid_themes.append(parts[1])
            else:
                valid_themes.append(line)
        
        if not valid_themes:
            return "都会のネズミと田舎のネズミ" # Fallback
            
        return random.choice(valid_themes)

    def run_model_comparison(self):
        """Generates scripts from 3 models and ranks them."""
        theme = self.load_random_theme()
        print(f"[Info] Selected Theme for Comparison: {theme}")
        
        results = {}
        for model in TARGET_MODELS:
            logger.info(f"Generating with {model}...")
            script = generate_manzai_script(API_KEY, theme=theme, model=model)
            if script and "Error" not in script:
                results[model] = script
            else:
                logger.error(f"Failed to generate with {model}")

        if len(results) < 2:
            print("Not enough scripts generated for comparison.")
            return

        ranking = self.judge_ranking(results, theme)
        
        # Save Log
        log_content = f"# Model Comparison Report\n## Theme: {theme}\n\n"
        for model, script in results.items():
            log_content += f"### Model: {model}\n{script}\n\n---\n\n"
        
        log_content += f"## Judge Ranking\n{ranking}"
        
        path = save_log("model_comparison.md", log_content)
        print(f"Comparison saved to {path}")

    def judge_ranking(self, results, theme):
        print(f"[Info] Running Ranking with {MODEL_JUDGE}...")
        
        scripts_text = ""
        model_names = list(results.keys())
        for i, model in enumerate(model_names):
            scripts_text += f"\n[Script {i+1} (Model: {model})]\n{results[model]}\n"

        prompt = f"""
You are an expert comedy editor.
Theme: {theme}
Gold Standard Style: "One-sided Manzai", Circular Flow, Reactive Description, Casual Tone.

Compare the following scripts:
{scripts_text}

Task:
1. Rank them from Best to Worst based on adherence to the Gold Standard style (especially the "Circular Refrain" and "Reactive" logic).
2. Explain WHY the winner is better (e.g., did it loop better? Was the tone softer?).

Output: Markdown ranking report.
"""
        messages = [{"role": "user", "content": prompt}]
        return call_llm(messages, MODEL_JUDGE, max_tokens=4000)

if __name__ == "__main__":
    opt = Optimizer()
    opt.run_model_comparison()
