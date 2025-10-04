# test_openai.py
import os
from dotenv import load_dotenv

# load .env file (nếu .env ở cùng thư mục)
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise SystemExit("ERROR: OPENAI_API_KEY not set. Check your .env file.")

# Cách 1: dùng client hiện đại (new SDK)
from openai import OpenAI
client = OpenAI(api_key=OPENAI_KEY)

resp = client.chat.completions.create(
    model="gpt-4o-mini",  # thay model tuỳ quyền truy cập của bạn
    messages=[{"role":"user","content":"Xin chào, bạn là ai?"}],
    max_tokens=100
)
print("Response:", resp.choices[0].message.content)

# --- Nếu bạn dùng style cũ (tuỳ phiên bản SDK) ---
# import openai
# openai.api_key = OPENAI_KEY
# r = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"user","content":"Xin chào"}])
# print(r.choices[0].message.content)
