import os
import time
import requests
from flask import Flask, request, jsonify, Response, render_template_string

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
GROQ_KEY = os.environ.get("GROQ_KEY", "")
HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "")
HF_BASE = "https://api-inference.huggingface.co/models"

HOME_HTML = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>🚀 Mini IA — Qwen · SmolLM2 · DeepSeek</title>
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
      <h1>🚀 Mini IA — Qwen · SmolLM2 · DeepSeek</h1>
      <p class="muted">Modelos: Qwen2.5-0.5B (primario) → SmolLM2-1.7B → DeepSeek-R1-1.5B</p>
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
    if (!res.ok) {
      const errorMsg = data.error || 'Error desconocido';
      out.innerHTML =
        `<span style="color:#f87171">✗ ${errorMsg}</span>\n\n` +
        JSON.stringify(data, null, 2);
      return;
    }

    const source = data.source || 'provider';
    out.innerHTML =
      `<span style="color:#3fb950">✓ ${source}</span>\n\n` +
      (data.text || JSON.stringify(data, null, 2));
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


def _hf_call(model_id, prompt, max_tokens):
    """
    Llama a HuggingFace Serverless Inference API.
    Funciona con o sin token (con token tiene más cuota).
    Retorna el texto generado o None si falla.
    """
    url = f"{HF_BASE}/{model_id}"
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False
        }
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=35)
        if res.status_code == 503:
            print(f"[HF] {model_id} cargando (503), esperando 10s...")
            time.sleep(10)
            res = requests.post(url, headers=headers, json=payload, timeout=35)

        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                text = data[0].get("generated_text", "")
            elif isinstance(data, dict):
                text = data.get("generated_text", "")
            else:
                text = str(data)
            text = text.strip()
            return text if text else None

        print(f"[HF] {model_id} respondió {res.status_code}: {res.text[:100]}")
        return None

    except Exception as e:
        print(f"[HF] {model_id} excepción: {e}")
        return None


def provider_gemma2(prompt, max_tokens):
    """
    P0 — Tu Gemma2-2B en HF Space.
    Primario absoluto. Es tu modelo propio.
    """
    if not HF_SPACE_URL:
        return None
    print("[P0] Tu HF Space Gemma2-2B...")
    try:
        res = requests.post(
            f"{HF_SPACE_URL.rstrip('/')}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens},
            timeout=60
        )
        if res.status_code == 200:
            data = res.json()
            return {
                "text": data.get("text", ""),
                "source": data.get("source", "HF Space Gemma2-2B (tuyo)")
            }
        if res.status_code == 503:
            print("[P0] Space cargando modelo (~60s), usando backup...")
    except Exception as e:
        print(f"[P0] Space caído: {e}")
    return None


def provider_qwen(prompt, max_tokens):
    """P1 — Qwen2.5-0.5B: ultra rápido, primary."""
    print("[P1] Qwen2.5-0.5B-Instruct...")
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    text = _hf_call("Qwen/Qwen2.5-0.5B-Instruct", formatted, max_tokens)
    if text:
        return {"text": text, "source": "Qwen2.5-0.5B (primary)"}
    return None


def provider_smollm2(prompt, max_tokens):
    """P2 — SmolLM2-1.7B: mejor calidad, backup-1."""
    print("[P2] SmolLM2-1.7B-Instruct...")
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    text = _hf_call(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        formatted,
        max_tokens
    )
    if text:
        return {"text": text, "source": "SmolLM2-1.7B (backup-1)"}
    return None


def provider_deepseek(prompt, max_tokens):
    """P3 — DeepSeek-R1-Distill-1.5B: razonamiento, backup-2."""
    print("[P3] DeepSeek-R1-Distill-Qwen-1.5B...")
    text = _hf_call(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        prompt,
        max_tokens
    )
    if text:
        return {"text": text, "source": "DeepSeek-R1-1.5B (backup-2)"}
    return None


def provider_groq(prompt, max_tokens):
    """P4 — Groq Llama3.1: backup externo, solo si GROQ_KEY existe."""
    if not GROQ_KEY:
        return None
    print("[P4] Groq Llama3.1-8B...")
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            },
            timeout=12
        )
        if res.status_code == 200:
            text = res.json()["choices"][0]["message"]["content"]
            return {"text": text, "source": "Groq Llama3.1 (backup-3)"}
        print(f"[P4] Groq respondió {res.status_code}")
    except Exception as e:
        print(f"[P4] Groq excepción: {e}")
    return None


def generate_text(prompt, max_tokens=150):
    for provider_fn in [
        provider_gemma2,
        provider_qwen,
        provider_smollm2,
        provider_deepseek,
        provider_groq,
    ]:
        result = provider_fn(prompt, max_tokens)
        if result and result.get("text"):
            print(f"[OK] → {result['source']}")
            return result

    print("[ERROR] Todos los proveedores fallaron")
    return None

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HOME_HTML)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'ok': True,
        'version': '2.0',
        'primary_model': 'Gemma2-2B via HF Space (P0)',
        'hf_space_configured': bool(HF_SPACE_URL),
        'hf_token_configured': bool(HF_TOKEN),
        'groq_configured': bool(GROQ_KEY),
        'providers': [
            'Gemma2-2B en HF Space (P0)',
            'Qwen2.5-0.5B (P1)',
            'SmolLM2-1.7B (P2)',
            'DeepSeek-R1-1.5B (P3)',
            'Groq Llama3.1 (P4 - solo si GROQ_KEY)'
        ],
        'timestamp': int(time.time())
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

    missing_config = []
    if not HF_SPACE_URL:
        missing_config.append('HF_SPACE_URL')
    if not HF_TOKEN:
        missing_config.append('HF_TOKEN (opcional, recomendado)')
    if not GROQ_KEY:
        missing_config.append('GROQ_KEY (opcional, backup externo)')

    return jsonify({
        'error': 'Todos los proveedores fallaron',
        'providers_tried': ['gemma2', 'qwen', 'smollm2', 'deepseek', 'groq'],
        'missing_config': missing_config,
        'hint': 'Configura variables en Render Dashboard: HF_SPACE_URL (requerida para Gemma2), HF_TOKEN (recomendado), GROQ_KEY (opcional).'
    }), 503

# --- OpenAI Compatible Endpoints ---

@app.route('/v1/models', methods=['GET'])
def list_models():
    models = []
    if HF_SPACE_URL:
        models.append({
            "id": "gemma2-2b", "object": "model",
            "created": int(time.time()), "owned_by": "you-via-hf-space"
        })
    models.extend([
        {"id": "qwen2.5-0.5b", "object": "model",
         "created": int(time.time()), "owned_by": "qwen"},
        {"id": "smollm2-1.7b", "object": "model",
         "created": int(time.time()), "owned_by": "huggingface"},
        {"id": "deepseek-r1-1.5b", "object": "model",
         "created": int(time.time()), "owned_by": "deepseek"},
    ])
    if GROQ_KEY:
        models.append({
            "id": "groq-llama3.1-8b", "object": "model",
            "created": int(time.time()), "owned_by": "groq"
        })
    return jsonify({"object": "list", "data": models})

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
