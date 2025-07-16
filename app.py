from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# --- Charger modèle et tokenizer ---
model = tf.keras.models.load_model("text_detector_model.keras")

with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

max_len = 300  # doit être le même que dans ton notebook

app = Flask(__name__)

def predict_text(text):
    if not text or text.strip() == "":
        return "Veuillez saisir un texte valide."

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    pred = model.predict(padded)[0][0]

    if pred >= 0.5:
        label = "AI-Generated"
        confidence = pred
    else:
        label = "Human-Written"
        confidence = 1 - pred

    return f"Prediction: {label} (Confidence: {confidence:.2f})"


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        input_text = request.form.get("text", "")
        result = predict_text(input_text)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
