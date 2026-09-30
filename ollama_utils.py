import time
import requests

OLLAMA_HOST = "http://ollama:11434"

def esperar_y_preparar_ollama(model_name="qwen2.5-coder:7b"):
    """Espera a que el servicio de Ollama esté activo y descarga el modelo si no existe."""
    print("Conectando con el servicio de Ollama...")
    
    # 1. Esperar a que el servidor de Ollama responda
    conectado = False
    for _ in range(30):
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags")
            if r.status_code == 200:
                conectado = True
                print("¡Conexión con Ollama establecida!")
                break
        except requests.exceptions.ConnectionError:
            time.sleep(2)
            
    if not conectado:
        raise Exception("No se pudo conectar con Ollama dentro del contenedor.")
        
    # 2. Descargar el modelo en Ollama si aún no está instalado
    print(f"Asegurando disponibilidad del modelo '{model_name}'...")
    payload = {"name": model_name, "stream": False}
    requests.post(f"{OLLAMA_HOST}/api/pull", json=payload)
    print(f"Modelo '{model_name}' listo para recibir consultas.")