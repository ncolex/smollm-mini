import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# HuggingFace Settings
# Usamos el nuevo ROUTER de HuggingFace que es el estándar ahora
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
HF_API_TOKEN = os.environ.get('HF_API_TOKEN', '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')

MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct"
]

@app.route('/', methods=['GET'])
def home():
    return "<h1>🚀 API Mini LLM Viva!</h1><p>Envia POST a /generate con {'prompt': 'tu texto'}</p>"

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json() or {}
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Falta el prompt'}), 400

    # Intentar con cada modelo en el Router
    for model_id in MODELS:
        try:
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100
            }
            res = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=10)
            
            if res.status_code == 200:
                result = res.json()
                return jsonify({
                    'text': result['choices'][0]['message']['content'],
                    'model': model_id,
                    'source': 'HF Router'
                })
        except Exception as e:
            continue

    return jsonify({'error': 'No se pudo generar respuesta. Verifica el HF_API_TOKEN en Render.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
