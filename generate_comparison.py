
import os
import script_generator
from dotenv import load_dotenv

load_dotenv()

theme = "初詣の屋台の価格設定と、それを買う心理"
with open("prompts/character_settings.txt") as f:
    chars = f.read()

key = os.environ.get("OPENROUTER_API_KEY")
print(script_generator.generate_manzai_script(key, theme, chars))
