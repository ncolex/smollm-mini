import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# LLAVES HARDCODED (Para que funcione sí o sí en Render)
GROQ_KEY = "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"
HF_TOKEN = "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"

@app.route('/', methods=['GET'])
def home():
    return "<h1>🚀 API Mini LLM Viva y Protegida!</h1><p>Envia POST a /generate con {'prompt': 'tu texto'}</p>"

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json() or {}
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Falta el prompt'}), 400

    # 1. INTENTO CON GROQ (Llama 3.1 - Súper rápido)
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150
            },
            timeout=10
        )
        if res.status_code == 200:
            return jsonify({
                'text': res.json()['choices'][0]['message']['content'],
                'source': 'Groq (Llama 3.1 8B)'
            })
    except Exception as e:
        print(f"Groq error: {e}")

    # 2. FALLBACK A HUGGINGFACE (Qwen 2.5 - Muy estable)
    try:
        res = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150
            },
            timeout=15
        )
        if res.status_code == 200:
            return jsonify({
                'text': res.json()['choices'][0]['message']['content'],
                'source': 'HuggingFace (Qwen 2.5)'
            })
    except Exception as e:
        print(f"HF error: {e}")

    return jsonify({'error': 'Todos los motores fallaron. Probá de nuevo en un segundo.'}), 500

if __name__ == '__main__':
    # Render usa la variable de entorno PORT
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
