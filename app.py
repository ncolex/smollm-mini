from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import gc

app = Flask(__name__)

# Cargar modelo al inicio (Render tiene 512MB en free)
print("🔥 Cargando SmolLM2-135M en 4-bit...")
quant_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    quantization_config=quant_config,
    device_map="auto",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
print("✅ Modelo cargado!")

@app.route('/', methods=['GET'])
def home():
    return "Mini LLM SmolLM2 listo 🔥 POST /generate con solo {prompt}"

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 128)
    if not prompt:
        return jsonify({'error': 'mandá "prompt"'}), 400
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
