import os
import time
import requests
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# LLAVES HARDCODED (Para que funcione sí o sí en Render)
GROQ_KEY = "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"
HF_TOKEN = "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"

def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        return add_cors_headers(Response())

@app.after_request
def after_request(response):
    return add_cors_headers(response)

def generate_text(prompt, max_tokens=150):
    """
    Backend real para generación de texto.
    Intenta Groq (Llama 3.1) y hace fallback a HuggingFace (Qwen 2.5).
    """
    # 1. INTENTO CON GROQ (Llama 3.1 - Súper rápido)
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            },
            timeout=10
        )
        if res.status_code == 200:
            return {
                'text': res.json()['choices'][0]['message']['content'],
                'source': 'Groq (Llama 3.1 8B)'
            }
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
                "max_tokens": max_tokens
            },
            timeout=15
        )
        if res.status_code == 200:
            return {
                'text': res.json()['choices'][0]['message']['content'],
                'source': 'HuggingFace (Qwen 2.5)'
            }
    except Exception as e:
        print(f"HF error: {e}")

    return None

@app.route('/', methods=['GET'])
def home():
    return "<h1>🚀 API Mini LLM Viva y Protegida!</h1><p>Envia POST a /generate con {'prompt': 'tu texto'} o usa /v1/chat/completions</p>"

@app.route('/generate', methods=['POST'])
def generate():
    """Endpoint legacy mantenido como wrapper"""
    data = request.get_json() or {}
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Falta el prompt'}), 400

    result = generate_text(prompt)
    if result:
        return jsonify(result)
    
    return jsonify({'error': 'Todos los motores fallaron. Probá de nuevo en un segundo.'}), 500

# --- OpenAI Compatible Endpoints ---

@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{
            "id": "smollm-mini",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openclaw"
        }]
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json() or {}
        messages = data.get('messages', [])
        model = data.get('model', 'smollm-mini')
        
        if not messages:
            return jsonify({'error': {'message': 'Messages are required', 'type': 'invalid_request_error'}}), 400

        # Extraer el último mensaje de usuario para el prompt
        prompt = ""
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                prompt = msg.get('content', '')
                break
        
        if not prompt and messages:
            # Fallback: usar el contenido del último mensaje sea cual sea el rol si no hay user explícito
            prompt = messages[-1].get('content', '')

        if not prompt:
            return jsonify({'error': {'message': 'No prompt content found', 'type': 'invalid_request_error'}}), 400

        # Usar el backend real
        result = generate_text(prompt)
        
        if not result:
            return jsonify({'error': {'message': 'Internal generation error', 'type': 'api_error'}}), 500

        # Formato de respuesta OpenAI
        return jsonify({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result['text']
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt) // 4, # Estimación
                "completion_tokens": len(result['text']) // 4, # Estimación
                "total_tokens": (len(prompt) + len(result['text'])) // 4
            }
        })
    except Exception as e:
        return jsonify({'error': {'message': str(e), 'type': 'server_error'}}), 500

if __name__ == '__main__':
    # Render usa la variable de entorno PORT
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)