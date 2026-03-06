import os
import time
import requests
from flask import Flask, request, jsonify, Response, render_template_string

app = Flask(__name__)

# Cargar llaves desde entorno para no exponer secretos en código.
GROQ_KEY = os.environ.get("GROQ_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

HOME_HTML = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Mini LLM API</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
    .wrap { max-width: 900px; margin: 40px auto; padding: 0 16px; }
    .card { background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
    h1 { margin-top: 0; font-size: 1.6rem; }
    .muted { color: #94a3b8; }
    textarea { width: 100%; min-height: 120px; border-radius: 8px; border: 1px solid #334155; background: #0b1220; color: #e2e8f0; padding: 10px; }
    button { margin-top: 12px; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; background: #2563eb; color: #fff; }
    pre { white-space: pre-wrap; background: #020617; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; }
    code { background: #1e293b; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>🚀 Mini LLM API</h1>
      <p class="muted">Ahora la raíz muestra una interfaz simple para probar el endpoint <code>/generate</code>.</p>
      <p class="muted">También puedes usar la API OpenAI compatible en <code>/v1/chat/completions</code>.</p>
    </div>

    <div class="card">
      <h2>Probar generación</h2>
      <textarea id="prompt" placeholder="Escribe tu prompt..."></textarea>
      <button onclick="sendPrompt()">Enviar</button>
      <pre id="out">Sin respuesta aún.</pre>
    </div>
  </div>

<script>
async function sendPrompt() {
  const prompt = document.getElementById('prompt').value.trim();
  const out = document.getElementById('out');
  if (!prompt) { out.textContent = 'Escribe un prompt antes de enviar.'; return; }
  out.textContent = 'Consultando...';
  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    out.textContent = 'Error de red: ' + err.message;
  }
}
</script>
</body>
</html>
"""

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
    if GROQ_KEY:
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

    if HF_TOKEN:
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
    return render_template_string(HOME_HTML)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'ok': True,
        'groq_configured': bool(GROQ_KEY),
        'hf_configured': bool(HF_TOKEN)
    })

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

    return jsonify({'error': 'No hay proveedores configurados o todos fallaron.'}), 500

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

        prompt = ""
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                prompt = msg.get('content', '')
                break

        if not prompt and messages:
            prompt = messages[-1].get('content', '')

        if not prompt:
            return jsonify({'error': {'message': 'No prompt content found', 'type': 'invalid_request_error'}}), 400

        result = generate_text(prompt)

        if not result:
            return jsonify({'error': {'message': 'No providers configured or generation failed', 'type': 'api_error'}}), 500

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
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(result['text']) // 4,
                "total_tokens": (len(prompt) + len(result['text'])) // 4
            }
        })
    except Exception as e:
        return jsonify({'error': {'message': str(e), 'type': 'server_error'}}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
