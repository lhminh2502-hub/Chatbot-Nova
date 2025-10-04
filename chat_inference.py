import json
import numpy as np
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load môi trường
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load mô hình TensorFlow
model = load_model("model/chatbot_model.h5")
tokenizer = pickle.load(open("model/tokenizer.pkl", "rb"))
lbl_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

with open("data/intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Hàm dự đoán intent
def predict_intent(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=20, truncating='post')
    result = model.predict(padded)
    intent = lbl_encoder.inverse_transform([np.argmax(result)])
    confidence = np.max(result)
    return intent[0], confidence

# Hàm trả lời
def chatbot_response(msg):
    intent, confidence = predict_intent(msg)

    # Nếu mô hình TensorFlow đủ chắc chắn → dùng dữ liệu local
    if confidence > 0.75:
        for i in data["intents"]:
            if i["tag"] == intent:
                return random.choice(i["responses"])

    # Nếu không chắc chắn → gọi OpenAI GPT
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là ChatGPT mini, một trợ lý thân thiện và thông minh."},
                {"role": "user", "content": msg},
            ],
            max_tokens=150
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Lỗi khi gọi GPT: {e}"
