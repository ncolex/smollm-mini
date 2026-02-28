import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configuración de APIs
HF_TOKEN = os.environ.get('HF_API_TOKEN', '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')
GROQ_KEY = os.environ.get('GROQ_API_KEY', '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')

@app.route('/', methods=['GET'])
def home():
    return "<h1>🚀 API Mini LLM Viva!</h1><p>Envia POST a /generate con {'prompt': 'tu texto'}</p>"

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json() or {}
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Falta el prompt'}), 400

    # 1. Intentar con GROQ (Es lo mejor y más estable)
    if GROQ_KEY:
        try:
            res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100
                },
                timeout=5
            )
            if res.status_code == 200:
                return jsonify({
                    'text': res.json()['choices'][0]['message']['content'],
                    'model': 'llama-3.1-8b (Groq)',
                    'source': 'Groq Cloud'
                })
        except: pass

    # 2. Intentar con HuggingFace Router (Fallback)
    for model_id in ["meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]:
        try:
            res = requests.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100
                },
                timeout=10
            )
            if res.status_code == 200:
                return jsonify({
                    'text': res.json()['choices'][0]['message']['content'],
                    'model': model_id,
                    'source': 'HF Router'
                })
        except: continue

    return jsonify({'error': 'Todos los modelos fallaron. Revisa las API Keys.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)