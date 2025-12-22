import os
import requests
import json
import re
from datetime import datetime
from typing import Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# MODEL_NAME = "openai/gpt-5.2"
MODEL_NAME = "google/gemini-3-pro-preview"

# ==============================================================================
# PROMPT CONFIGURATION
# ==============================================================================
# The user can easily modify these prompts to tweak the generation behavior.

PROMPT_THEMES = """あなたは「片側だけ漫才」のアイディア出し担当です。

「片側だけ漫才」とは、二人の会話のうち一方（A）のセリフだけを抜き出したネタです。
Aの発言だけから、Bが「言葉を字義通りにとってしまう」「比喩が通じない」などのズレた反応をしていることが分かるようにします。

以下の条件を満たすユニークなテーマを10個提案してください。
出力はテーマのリストのみで、1行に1つのテーマを書いてください。

条件：
- **言葉のあや・比喩・慣用句**が鍵となるテーマ
- 「Aが比喩で説明する」→「Bが字義通りに受け取る」というすれ違いが起きやすい状況
- 具体的には「ことわざ」「業界用語」「抽象的な概念の説明」など

出力例：
「胸が痛む」を物理的な痛みの話だと勘違いする
「腹を割って話す」を聞いて手術の準備を始める
「頭を冷やす」と言われて冷蔵庫に入ろうとする
"""

# Base instruction for the Script Generator
# This logic generates ONLY A's lines directly (One-Shot).
# Base instruction for the Script Generator
# This logic generates ONLY A's lines directly (One-Shot).
SYSTEM_INSTRUCTION_TEMPLATE = """あなたは卓越した構成作家兼シナリオライターです。
以下の【Generate Step】に従い、提供された「テーマ」と「キャラクター設定」を元に、キャラクターA（西園寺 紫織）の「一人だけの会話劇（一人漫才）」を出力してください。

# Character Settings
{character_settings}

# Generation Logic (The "Meta-Manzai" Engine)
このスクリプトは以下の論理で構成してください。

1.  **アンカー（本筋）の設定**：
    Aには「どうしても語り終えたい、高尚なテーマに関する一文」を持たせてください。Aは何度もこの文の最初に戻ろうとします（リピート再生）。

2.  **インビジブル・ボケ（不可視の攪乱）**：
    Bの発言は一切出力しません。しかし、Aの反応から以下のパターンのいずれかのボケがあったことを示唆してください。
    - **字義通りの解釈**：（例：「『彼の手は冷たかった』といった比喩に対し『死後硬直？』と解釈する」）
    - **超・飛躍**：（例：「歴史の話から急に『昼ごはん』や『宇宙人』の話へ飛ぶ」）
    - **無意味な具体化**：（例：「『昔々ある所に』に対し『住所は？ 郵便番号は？』と迫る」）

3.  **Aのリアクション（ここを出力）**：
    Bの発言を「〜と言わないで」と引用する安易な表現は禁止です。代わりに以下のように反応してください。
    - **×悪い例**：「『ゴリラ？』じゃないですわ、太宰治です」
    - **◎良い例**：「……待って。なぜここで霊長類が出てきますの？ 太宰の苦悩の話をしていて、バナナの話題にはなりませんわよね？」
    - **マジレス解説**：Bのアホな疑問に対し、Aは知識を総動員して「なぜそれが間違っているか」あるいは「仮にその前提だとどうなるか」を長々と論理的に説明し、自滅してください。

4.  **構成**：
    「本筋を話そうとする」→「Bが遮る」→「Aが解説・修正する」→「冒頭に戻る」を数回繰り返し、最後はAが諦めるか、論理が崩壊した結論（「もうそれでいいですわ」）に至らせてください。

# Input Data
- **テーマ**：{theme}

# Output Format
Markdown形式。Aのセリフのみを改行で区切って記述。ト書きやBのセリフは一切含めないこと。
"""

# ==============================================================================
# API FUNCTIONS
# ==============================================================================

def _call_openrouter(messages: list, api_key: str, model: str) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Starttoaster/T5Gemma-TTS",
        "X-Title": "T5Gemma-TTS Script Generator",
    }
    data = {
        "model": model,
        "messages": messages,
    }
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            print(f"[Error] No choices in response: {result}")
            return None
    except Exception as e:
        print(f"[Error] API call failed: {e}")
        return None

def generate_themes(api_key: str, model: str = MODEL_NAME) -> list:
    if not api_key:
        return ["Error: API Key is missing."]
        
    messages = [
        {"role": "user", "content": PROMPT_THEMES}
    ]
    
    print("[Info] Generating themes...")
    content = _call_openrouter(messages, api_key, model)
    if not content:
        return ["Error: Failed to generate themes."]
        
    lines = content.strip().split('\n')
    themes = []
    for line in lines:
        # Clean up numbering
        line = re.sub(r'^[\d-]+\.\s*', '', line).strip()
        line = re.sub(r'^-\s*', '', line).strip()
        if line:
            themes.append(line)
            
    return themes[:10]

def generate_manzai_script(api_key: str, theme: Optional[str] = None, characters: str = "", model: str = MODEL_NAME) -> str:
    """
    Generates the A-side script directly.
    """
    if not api_key:
        return "Error: API Key is missing."
    
    # Defaults if missing
    if not theme:
        theme = "言葉のあやによるすれ違い"
    
    if not characters:
        characters = """- **A：西園寺 紫織**（出力する話者）：
    - 博識だが融通が利かないお嬢様（エセ）。口調は「〜ですわ」「〜ますの」。
    - 文学的・哲学的な解説を好むが、Bのノイズにより論旨がズレていく。
    - 苛立つと素の口調（「〜じゃない」「〜でしょ」）が出る。
- **B：田中 ぽえむ**（透明な聞き手・セリフは出力しない）：
    - Aの話を聞いているが、単語の響きだけで連想ゲームをしたり、極端に即物的な解釈をする。
    - Aにとって予想外の「些末なディテール」に食いつく。"""

    # Build prompt
    prompt = SYSTEM_INSTRUCTION_TEMPLATE.format(
        theme=theme,
        character_settings=characters
    )

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    print(f"[Info] Generating direct script for theme '{theme}'...")
    content = _call_openrouter(messages, api_key, model)
    if not content:
        return "Error: Failed to generate script."
        
    return content

def clean_script_for_speech(text: str) -> str:
    """
    Simple cleanup for the direct generation output.
    Most of the heavy lifting should be done by the prompt.
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove speaker prefixes like "A:", "西園寺 紫織：", "田中:", etc.
        # Regex: Start of line, any characters that are not a full width/half width colon, followed by colon.
        # We limit the length of the name to avoid stripping actual dialogue that contains colons.
        # e.g. "Name: Text" -> match "Name:"
        # But "This: That" inside a sentence ideally shouldn't be matched if it's not at start.
        # Let's assume speaker names are relatively short (e.g. < 15 chars).
        line = re.sub(r'^.{1,15}[:：]', '', line).strip()
        
        # Replace quotes
        line = re.sub(r'"([^"]*)"', r'「\1」', line)
        line = line.replace('“', '「').replace('”', '」')
        line = line.replace('"', '') 
        
        # Remove parentheses description (actions, emotions)
        line = re.sub(r'[\(（][^\)）]*[\)）]', '', line)
        
        if line:
            cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines)

def save_script_to_file(text: str, output_dir: str = "texts") -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_script_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        return filepath
    except Exception as e:
        return f"Error saving file: {e}"
