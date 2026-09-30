import sys
from ollama_utils import esperar_y_preparar_ollama
from rag_engine import buscar_contexto, construir_base_vectorial
from agentes import agente_coder, agente_auditor

def main():
    print("=== INICIANDO PRUEBA DE CONCEPTO DE AGENTES AUTÓNOMOS ===\n")
    
    # 1. Preparar Ollama
    esperar_y_preparar_ollama("qwen2.5-coder:7b")
    
    # 2. Asegurar que exista la base RAG
    try:
        # Intentar una búsqueda de prueba para verificar si la DB existe
        buscar_contexto("prueba", k=1)
    except Exception:
        print("Base de datos no encontrada. Generándola por primera vez...")
        construir_base_vectorial()
        
    # 3. Definir la tarea científica de demostración
    tarea_usuario = "Escribe un script en Python para filtrar un conjunto de datos y seleccionar eventos con PT mayor a 25 GeV"
    print(f"\n[SISTEMA] Tarea recibida: '{tarea_usuario}'")
    
    # 4. EJECUCIÓN AGENTE 1: CODER
    print("\n--- 1. CONSULTANDO RAG PARA EL AGENTE CODER ---")
    contexto_coder = buscar_contexto(tarea_usuario, k=3)
    
    print("\n--- 2. EJECUTANDO AGENTE CODER ---")
    codigo_respuesta = agente_coder(tarea_usuario, contexto_coder)
    print("\n[CÓDIGO GENERADO]:\n")
    print(codigo_respuesta)
    
    # 5. EJECUCIÓN AGENTE 2: AUDITOR
    pregunta_auditoria = "Criterios de selección y límites de tolerancia para selección de fotones/muones"
    print("\n--- 3. CONSULTANDO RAG PARA EL AGENTE AUDITOR ---")
    contexto_auditor = buscar_contexto(pregunta_auditoria, k=2)
    
    print("\n--- 4. EJECUTANDO AGENTE AUDITOR ---")
    dictamen = agente_auditor(codigo_respuesta, contexto_auditor)
    print("\n[DICTAMEN DEL AUDITOR]:\n")
    print(dictamen)
    print("\n=== PRUEBA FINALIZADA CON ÉXITO ===")

if __name__ == "__main__":
    main()