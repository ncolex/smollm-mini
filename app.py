import os
import requests
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# HuggingFace API settings
HF_API_TOKEN = os.environ.get('HF_API_TOKEN', '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')
# We'll try these models in order on Render
HF_MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "gpt2"
]

# Ollama local settings
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

@app.route('/', methods=['GET'])
def home():
    return f"""
    <h1>🚀 Mini LLM ready!</h1>
    <p><b>Local Ollama:</b> {OLLAMA_URL}</p>
    <p><b>HF API Token:</b> {'Configured' if HF_API_TOKEN else 'Missing'}</p>
    <p><b>POST to /generate with:</b> <code>{{'prompt': 'text'}}</code></p>
    """

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json() or {}
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 128)
    
    if not prompt:
        return jsonify({'error': 'Please provide a prompt'}), 400
    
    # 1. Try Ollama locally first (if not on Render)
    if not os.environ.get('RENDER'):
        try:
            ollama_res = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": "smollm2:135m",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens}
                },
                timeout=30
            )
            if ollama_res.status_code == 200:
                return jsonify({'text': ollama_res.json().get('response', ''), 'source': 'Ollama Local'})
        except Exception as e:
            print(f"DEBUG: Ollama local failed or timed out: {e}")

    # 2. Fallback to HuggingFace API (Required for Render)
    for model_id in HF_MODELS:
        try:
            # We try both v1 (OpenAI format) and legacy Inference API
            # First try v1 chat completions (recommended)
            router_url = "https://router.huggingface.co/v1/chat/completions"
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
            res = requests.post(router_url, headers={"Authorization": f"Bearer {HF_API_TOKEN}"}, json=payload, timeout=15)
            
            if res.status_code == 200:
                return jsonify({
                    'text': res.json()['choices'][0]['message']['content'], 
                    'source': f'HF Router ({model_id})'
                })
            
            # If not a chat model or not supported, try legacy inference API
            inf_url = f"https://api-inference.huggingface.co/models/{model_id}"
            res = requests.post(inf_url, headers={"Authorization": f"Bearer {HF_API_TOKEN}"}, 
                                json={"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}, timeout=15)
            
            if res.status_code == 200:
                res_json = res.json()
                text = res_json[0].get('generated_text', '') if isinstance(res_json, list) else res_json.get('generated_text', str(res_json))
                return jsonify({'text': text, 'source': f'HF Inference API ({model_id})'})
                
        except Exception as e:
            print(f"DEBUG: Model {model_id} failed: {e}")
            continue

    return jsonify({'error': 'All models and endpoints failed', 'details': 'Check logs for more info'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)