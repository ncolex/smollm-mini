import time
import requests
import json
import os
from datetime import datetime

# Configuración
RENDER_URL = "https://smollm-mini.onrender.com/generate"
INTERVALO_MINUTOS = 33
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
    log_message(f"❤️ Iniciando Nodo Heartbeat (Intervalo: {INTERVALO_MINUTOS} min)")
    
    while True:
        try:
            log_message("Enviando pulso a la Nube...")
            
            # 1. Llamada a la API en Render
            start_time = time.time()
            response = requests.post(
                RENDER_URL,
                json={
                    "prompt": "Generá un reporte de estado técnico breve y futurista sobre el sistema 'OpenClaw Agent'.",
                    "max_tokens": 100
                },
                timeout=60
            )
            
            duration = round((time.time() - start_time) * 1000)
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if response.status_code == 200:
                data = response.json()
                text = data.get('text', 'Sin respuesta').strip()
                source = data.get('source', 'Unknown')
                
                log_message(f"✅ Respuesta recibida de {source} ({duration}ms)")
                
                # Guardar en historial
                entry = {
                    "time": timestamp,
                    "status": "OK",
                    "latency": duration,
                    "source": source,
                    "message": text
                }
            else:
                log_message(f"❌ Error HTTP {response.status_code}", "error")
                entry = {
                    "time": timestamp,
                    "status": "ERROR",
                    "latency": duration,
                    "source": "System",
                    "message": f"Error: {response.text}"
                }

        except Exception as e:
            log_message(f"❌ Error de conexión: {e}", "error")
            entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "status": "FAIL",
                "latency": 0,
                "source": "Local Node",
                "message": str(e)
            }

        # Actualizar JSON
        history = load_history()
        history.insert(0, entry)
        save_history(history)

        # Esperar 33 minutos
        log_message(f"💤 Durmiendo {INTERVALO_MINUTOS} minutos...")
        time.sleep(INTERVALO_MINUTOS * 60)

if __name__ == "__main__":
    run_heartbeat()
