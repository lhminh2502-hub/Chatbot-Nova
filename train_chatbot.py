import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# 1. Load dữ liệu
with open("data/intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

# 2. Chuẩn bị dữ liệu
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding="post")

lbl_encoder = LabelEncoder()
labels_encoded = lbl_encoder.fit_transform(labels)

# 3. Tạo mô hình
# 3. Tạo mô hình
model = Sequential([
    Embedding(1000, 16, input_length=padded.shape[1]),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(len(set(labels_encoded)), activation='softmax')
])


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded, np.array(labels_encoded), epochs=400, verbose=0)

# 4. Lưu mô hình và tokenizer
model.save("model/chatbot_model.h5")
import pickle
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(lbl_encoder, f)

print("✅ Huấn luyện hoàn tất và mô hình đã lưu.")
