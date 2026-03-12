import time
import requests
import json
import os
from datetime import datetime

# Configuración
RENDER_URL = "https://smollm-mini.onrender.com/generate"
INTERVALO_SEGUNDOS = 14 * 60   # 840s — Render duerme a los 15min
MAX_REINTENTOS = 3
ESPERA_REINTENTO = 20        # segundos entre reintentos
LOG_FILE = "heartbeat.json"

def log_message(msg, type="info"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{type.upper()}] {msg}")

def load_history():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    # Guardamos solo los últimos 50 latidos
    with open(LOG_FILE, 'w') as f:
        json.dump(history[:50], f, indent=2)

def run_heartbeat():
    log_message(f"❤️ Heartbeat iniciado — ping cada {INTERVALO_SEGUNDOS // 60} min")

    while True:
        entry = None
        for intento in range(1, MAX_REINTENTOS + 1):
            try:
                log_message(f"Pulso intento {intento}/{MAX_REINTENTOS}...")
                start_time = time.time()
                response = requests.post(
                    RENDER_URL,
                    json={"prompt": "Di OK", "max_tokens": 5},
                    timeout=90
                )
                duration = round((time.time() - start_time) * 1000)
                timestamp = datetime.now().strftime("%H:%M:%S")

                if response.status_code == 200:
                    data = response.json()
                    source = data.get('source', 'Unknown')
                    log_message(f"✅ OK desde {source} ({duration}ms)")
                    entry = {
                        "time": timestamp,
                        "status": "OK",
                        "latency": duration,
                        "source": source,
                        "message": data.get('text', 'pong').strip()
                    }
                    break

                else:
                    log_message(f"❌ HTTP {response.status_code}", "error")
                    if intento < MAX_REINTENTOS:
                        time.sleep(ESPERA_REINTENTO)

            except Exception as e:
                log_message(f"❌ Conexión fallida: {e}", "error")
                if intento < MAX_REINTENTOS:
                    time.sleep(ESPERA_REINTENTO)

        if entry is None:
            entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "status": "FAIL",
                "latency": 0,
                "source": "heartbeat-node",
                "message": f"Falló {MAX_REINTENTOS} reintentos"
            }

        # Actualizar JSON
        history = load_history()
        history.insert(0, entry)
        save_history(history)

        log_message(f"💤 Durmiendo {INTERVALO_SEGUNDOS // 60} minutos...")
        time.sleep(INTERVALO_SEGUNDOS)

if __name__ == "__main__":
    run_heartbeat()
