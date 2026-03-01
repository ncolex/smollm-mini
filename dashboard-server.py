from flask import Flask, request, jsonify, send_from_directory
import requests as req
import os
import json

app = Flask(__name__)

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

@app.route('/')
def index():
    return send_from_directory('.', 'ollama-dashboard.html')

@app.route('/status')
def status():
    return send_from_directory('.', 'ollama-status.html')

@app.route('/api/heartbeat')
def get_heartbeat():
    try:
        if os.path.exists('heartbeat.json'):
            with open('heartbeat.json', 'r') as f:
                return jsonify(json.load(f))
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/<path:endpoint>', methods=['GET', 'POST'])
def proxy(endpoint):
    try:
        url = f'{OLLAMA_URL}/api/{endpoint}'
        if request.method == 'POST':
            res = req.post(url, json=request.get_json() or {})
        else:
            res = req.get(url)
        return jsonify(res.json()), res.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('🔥 Dashboard Ollama corriendo en http://localhost:5000')
    print(f'   → Proxy hacia: {OLLAMA_URL}')
    app.run(host='0.0.0.0', port=5000, debug=True)
