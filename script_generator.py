import os
import requests
import json
import re
from datetime import datetime
from typing import Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-5.2"

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
SYSTEM_INSTRUCTION_TEMPLATE = """あなたは「片側だけ漫才」の脚本家です。
「片側だけ漫才」とは、漫才コンビ（AとB）の会話から、ツッコミ役（A）のセリフだけを抜き出したものです。
読み手はAのセリフを読むだけで、ボケ役（B）がどんなトンチンカンなことを言ったか想像でき、笑えるようにしなければなりません。

## キャラクター設定
{character_settings}

## テーマ
**{theme}**

## 執筆ルール
1. **出力するのはAのセリフのみ**です。Bのセリフは一切書いてはいけません。
2. Aのセリフの間に、Bのターンがあったことを示す「間（改行）」を入れてください。
3. Aのセリフは、直前のBの発言（想像上のもの）を引用・訂正する形で構成してください。
   - 例: 「金の話じゃない！『光るものが好き』って言ったのは才能の話！」
4. Bは常に「比喩を字義通りに受け取る」「音の響きだけで別の言葉に結びつける」などのアホな反応をします。
5. 形式: Aのセリフ1行（または複数行） → 改行 → Aのセリフ1行...

## 出力例
（テーマ: 「首を長くして待つ」）

首を長くして待ってたよ。本当に。
いや、物理的に伸ばすわけないでしょ。キリンじゃないんだから。
だから、期待して待ってたって意味！筋肉の話に戻さないで。
もういい、座って。

## 本番
それでは、テーマ「{theme}」でAのセリフだけを出力してください。
余計な説明や「---」などの区切り線は不要です。
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
        characters = """A: 理屈っぽい大学生。言葉の正確さにこだわる。
B (不可視): 言葉を全て字義通りに受け取る、空気の読めない友人。"""

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
