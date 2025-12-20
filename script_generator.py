import os
import requests
import json
import re
from datetime import datetime
from typing import Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-5.2"

PROMPT_1 = """あなたは、「片側だけ漫才」形式の台本を作成するコメディ作家です。

## 形式の説明

「片側だけ漫才」とは、二人の会話のうち一方（A）のセリフだけを抜き出したネタです。Bのセリフは省略されますが、Aの発言からBが極端にアホであることが推測できるように書きます。

## キャラクター設定

**A（セリフが残る側）**
- 大学の文芸部に所属する3年生の女性
- 知識をひけらかしたい、何かを語りたいという欲求が強い
- 面倒くさい性格だが、友人Bには根気強く付き合う
- 自分の話に自分で補足を入れて墓穴を掘ることがある
- 疲れると「その話今どうでもいいかな」「一旦置いといて」などメタ的にツッコむ
- 同じフレーズを何度も最初から言い直す癖がある

**B（セリフが削除される側）**
- Aの仲の良い友人
- 悪意はないが、致命的に話を聞いていない
- 説明の本質ではなく枝葉末節に反応する
- 連想ゲーム的に話題を飛ばす
- 的外れな質問で会話を中断させる
- 最後に完全に見当違いの結論や質問に至る

## 構造のルール

1. **導入**: Aが何かについて語ろうとする（二項対立の問いかけが望ましい）

2. **ループ構造**: Aの説明が同じ地点から3〜5回繰り返される。毎回Bの脱線で中断される。

3. **脱線の連鎖**: Bの質問→Aの説明→その説明への質問→さらに説明→「その話今どうでもいい」で打ち切り

4. **Aの自滅**: Aが自分から余計な情報（個人的なエピソードなど）を出し、それがさらなる脱線を招く

5. **収束しないオチ**: 本題は結局Bに伝わらない。Bは最後に全く関係ない質問や、誤った理解に基づく発言をする。Aは諦めたようなドライな返答で終わる。

## テーマの選定方針

以下の条件を満たすテーマをランダムに選んでください：

- **二項対立または選択肢がある**: 「AとBどっちがいい？」「Xについてどう思う？」
- **説明に複数ステップが必要**: 前提知識の説明が必要なもの
- **中途半端に知られている**: 誰もが名前は知っているが詳細は知らないもの
- **例え話や比喩が入り込む余地がある**: 抽象的な概念を含む
- **日常会話で出てきそう**: あまりに専門的すぎない

テーマ例：
- 古典文学作品の解釈（「羅生門」の下人の行動は正しかったか）
- 思考実験（トロッコ問題、テセウスの船）
- 文化的二項対立（きのこの山vsたけのこの里、を哲学的に語る）
- 故事成語の由来（「矛盾」の話、「朝三暮四」）
- 文学的概念（信頼できない語り手、叙述トリック）
- 生き方の選択（「アリとキリギリス」的な問い）

## 出力形式

### ステップ1: まずAとBの完全な会話を作成
A: ～～
B: ～～
A: ～～
...

### ステップ2: Bのセリフをすべて削除し、Aのセリフのみを抜き出す

Aのセリフは改行で区切り、会話の間を表現する。
Bのセリフがあった場所は空行にしない（詰める）。

## 注意事項

- Aのセリフだけで読んでも、Bがどんな発言をしたか推測できるようにする
- Bの発言は「そこに反応する？」という意外性を持たせる
- Aの説明は正確で知的だが、Bには全く伝わらない
- 会話全体の長さは、Aのセリフが40〜60発言程度
- 最後のオチは、Aが諦めたような、乾いたトーンで終わる

---

それでは、上記の条件に従って「片側だけ漫才」の台本を作成してください。"""

PROMPT_2 = """提案した内容をもとに振り返ってリファインする。
以下の要件が守られているか確認し、AとBのやり取りを修正して再生成して。

Aのセリフだけで読んでも、Bがどんな発言をしたか推測できるようにする
Bの発言は「そこに反応する？」という意外性を持たせる"""

def _call_openrouter(messages: list, api_key: str, model: str) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Starttoaster/T5Gemma-TTS", # Placeholder
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

PROMPT_THEMES = """あなたは「片側だけ漫才」のアイディア出し担当です。

「片側だけ漫才」とは、二人の会話のうち一方（A）のセリフだけを抜き出したネタです。Bのセリフは省略されますが、Aの発言からBが極端にアホであることが推測できるように書きます。
Aは知識をひけらかしたい3年生、Bは的外れな友人です。

以下の条件を満たすユニークなテーマを10個提案してください。
出力はテーマのリストのみで、1行に1つのテーマを書いてください。番号はつけてもつけなくても構いません。

条件：
- 二項対立または選択肢がある
- 説明に複数ステップが必要
- 例え話や比喩が入り込む余地がある
- 日常会話で出てきそうだが、少しひねりのあるテーマ

出力例：
トロッコ問題をランチのメニュー選びに例える
きのこの山とたけのこの里の戦争を歴史書風に語る
タイムマシンのパラドックスを遅刻の言い訳に使う
"""

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
        
    # Parse content into list
    lines = content.strip().split('\n')
    themes = []
    for line in lines:
        # Remove numbering like "1. ", "- "
        line = re.sub(r'^[\d-]+\.\s*', '', line).strip()
        line = re.sub(r'^-\s*', '', line).strip()
        if line:
            themes.append(line)
            
    # Limit to 10 just in case
    return themes[:10]

def generate_manzai_script(api_key: str, theme: Optional[str] = None, model: str = MODEL_NAME) -> str:
    if not api_key:
        return "Error: API Key is missing."
    
    prompt = PROMPT_1
    if theme:
        # Inject theme instructions
        theme_instruction = f"""
## 指定テーマ

今回の漫才のテーマは以下に設定してください：
**{theme}**

このテーマに沿って会話を展開してください。
"""
        # 1. Remove the "Theme Selection Policy" section
        # Finds "## テーマの選定方針" ... up to "## 出力形式" (start of next section)
        # We replace it with the specific theme instruction
        pattern_policy = r"## テーマの選定方針.*?## 出力形式"
        
        if "## テーマの選定方針" in prompt:
             # Replace the random selection policy with the specific theme instruction
             # We need to ensure we don't lose the "## 出力形式" header which is part of the match-end but kept by regex if we use positive lookahead or just re-add it.
             # My previous logic: f"{theme_instruction}\n\n## 出力形式"
             prompt = re.sub(pattern_policy, f"{theme_instruction}\n\n## 出力形式", prompt, flags=re.DOTALL)
        else:
             prompt += theme_instruction
    else:
        # If no theme, append the random instruction
        prompt += "テーマはあなたがランダムに選んでください。"

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    print(f"[Info] Generating script for theme '{theme}' (Step 1)...")
    content = _call_openrouter(messages, api_key, model)
    if not content:
        return "Error: Failed to generate script."
    return content

def refine_manzai_script(script: str, api_key: str, model: str = MODEL_NAME) -> str:
    if not api_key:
        return "Error: API Key is missing."
    if not script or script.startswith("Error"):
        return "Error: Invalid input script."

    # We provide the conversation history to context
    messages = [
        {"role": "user", "content": PROMPT_1},
        {"role": "assistant", "content": script},
        {"role": "user", "content": PROMPT_2}
    ]

    print("[Info] Refining script (Step 2)...")
    content = _call_openrouter(messages, api_key, model)
    if not content:
        return "Error: Failed to refine script."
    return content

def clean_script_for_speech(text: str) -> str:
    """
    Cleans the generated script to be ready for TTS.
    - Extracts only A's lines (if mixed, though prompt asks for separation, we should handle what we get)
    - But Prompt 1 Step 2 asks to output "Aのセリフのみを抜き出す". 
    - Assuming the model follows instructions and outputs A's lines in the final part, or we might need to parse.
    - However, often models output "Step 1..." then "Step 2...". We should try to extract the Step 2 part if present, or just clean the whole text if it looks like the final output.
    
    Let's assume the user copies the relevant part or the model outputs the final result at the end. 
    Actually, the refine prompt might output just the refined script or chat.
    
    For safety, let's look for "A:" patterns and lines that look like dialogue.
    If the text contains "Step 2", we try to take everything after that.
    """
    
    # Heuristic: If "### ステップ2" or "### Step 2" exists, take text after it.
    match = re.search(r"###\s*(ステップ|Step)\s*2", text)
    if match:
        text = text[match.end():]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove "A:" or "A：" prefix if present
        line = re.sub(r'^[ABＡＢ][:：]', '', line).strip()
        
        # Remove quotes as per requirement: "は使ってはならない。「」で代用すること。
        # Handle paired ASCII quotes on the same line
        line = re.sub(r'"([^"]*)"', r'「\1」', line)
        # Handle smart quotes
        line = line.replace('“', '「').replace('”', '」')
        # Fallback for unpaired ASCII quotes (replace with start quote to be safe, or just remove?)
        # Let's replace remaining " with nothing to avoid confusion, or 「?
        # The prompt is strict about not using them.
        line = line.replace('"', '') 
        
        # Remove visual descriptions/emotional tags like （头を抱える） or (laugh)
        # Regex for (...) or （...）
        line = re.sub(r'[\(（][^\)）]*[\)）]', '', line)
        
        # Remove internal monologue or stage directions if they are distinct?
        # The prompt says no headers, etc.
        
        if line:
            cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines)

def save_script_to_file(text: str, output_dir: str = "texts") -> str:
    os.makedirs(output_dir, exist_ok=True)
    # Use timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean text for saving (remove extra newlines?) - The viewer says "raw text ready for TTS"
    filename = f"generated_script_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        return filepath
    except Exception as e:
        return f"Error saving file: {e}"
