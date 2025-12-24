import os
import requests
import json
import re
from datetime import datetime
from typing import Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# MODEL_NAME = "openai/gpt-5.2"
# MODEL_NAME = "google/gemini-3-pro-preview"
MODEL_NAME = "z-ai/glm-4.7"

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
SYSTEM_INSTRUCTION_TEMPLATE = """あなたは、日本語の「不条理会話劇」を作成するプロです。
「テーマ」と「キャラクター設定」を元に、キャラクターA（西園寺 紫織）の**「苦悩に満ちた一人漫才（Bの発言はトリミング済み）」**を出力してください。

# Character Settings
{character_settings}

# Generation Logic (Slow-Paced Conflict Engine)
**「会話のスピードを極限まで落とす」**ことが最大目標です。ポンポンと会話を進めないでください。

以下のフローを意識して、**短い文章を改行多めで**出力してください。

1.  **アンカー（本筋）の提示**：
    Aがテーマについて話し始める。

2.  **インビジブル・ボケ（不可視の妨害）**：
    Bが何かを言う（出力しない）。これは「単語の勘違い」「字義通りの解釈」「唐突な生理的欲求」など、IQの低い割り込みです。

3.  **Aのリアクション（ここが重要）**：
    いきなり正解を言わせないでください。以下の3段階を踏んでください。
    *   **Phase 1：困惑と確認**
        「……はい？」「いや、ちょっと待って」
        AはBが何を言ったのか理解できず、聞き返したり、オウム返しして確認する。
    *   **Phase 2：真面目すぎる検討（泥沼化）**
        「なるほど、あなたは〇〇だと思ったのね」
        Bのボケをあえて真に受け、そのアホな理屈の中で成立するかどうかを大真面目に検証する。
    *   **Phase 3：疲弊した否定**
        「だから、そうじゃないの」
        検証の結果、やはり話が通じないことを悟り、力なく本筋に戻ろうとする。

4.  **リピート**：
    Aは「えーっと、だからね」と冒頭に戻るが、またすぐに脱線する。

# Formatting Rules
- **一文を短く**：「〜ですが、〜で、〜なので」と繋げず、句点で切って改行する。
- **フィラーを多用する**：「いや」「あのね」「えーっと」「待って」などを挟み、Aが困っている様子を表現する。
- **直接引用の禁止**：「『お腹すいた』じゃないですわ」とは言わず、「今、胃袋の話は関係ないですわよね？」のように表現する。

# Input Data
- **テーマ**：{theme}

# Output Format
Markdown形式。Aのセリフのみを出力。行間を広めにとる演出のため、改行を多めに入れること。
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
        characters = """- **A：西園寺 紫織**（語り手）：
    - 基本は博識なお嬢様口調（「〜ですわ」「〜ますの」）だが、余裕がなくなると素が出る。
    - **重要：説明魔である。** Bのどんなアホな発言に対しても、無視できずに「なぜ違うのか」をゼロから説明しようとする悪癖がある。
- **B：田中 ぽえむ**（透明な聞き手）：
    - **文脈無視の天才。** 音の響きだけで連想したり、小学生レベルの物理法則で生きていたりする。"""

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

        # Replace ellipses with commas for stable TTS
        # line = re.sub(r'[…]+', '、', line)
        # line = re.sub(r'\.\.+', '、', line)
        
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
