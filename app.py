from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = Flask(__name__)

# Usar modelo más chico: SmolLM2-135M es muy grande para 512MB
# Usamos la API de HuggingFace Inference (gratis)
HF_API_TOKEN = os.environ.get('HF_API_TOKEN', '')

if not HF_API_TOKEN:
    # Fallback: cargar modelo localmente (puede fallar en 512MB)
    print("⚠️ Sin HF_API_TOKEN, cargando modelo localmente...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        print("✅ Modelo cargado!")
        LOCAL_MODE = True
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        LOCAL_MODE = False
else:
    print("✅ Usando HuggingFace Inference API")
    LOCAL_MODE = False

@app.route('/', methods=['GET'])
def home():
    if LOCAL_MODE:
        return "Mini LLM SmolLM2 (local) 🔥 POST /generate"
    else:
        return "Mini LLM (HF API) 🔥 POST /generate"

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 128)
    
    if not prompt:
        return jsonify({'error': 'mandá "prompt"'}), 400
    
    if not LOCAL_MODE:
        # Usar API de HuggingFace
        import requests
        response = requests.post(
            "https://api-inference.huggingface.co/models/HuggingFaceTB/SmolLM2-135M",
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
        )
        if response.status_code == 200:
            result = response.json()
            text = result[0]['generated_text'] if isinstance(result, list) else result.get('generated_text', '')
            return jsonify({'text': text})
        else:
            return jsonify({'error': f'HF API error: {response.status_code}'}), 500
    
    # Modo local
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
