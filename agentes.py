from langchain_ollama import ChatOllama

# Configuración del modelo conectado a Ollama en Docker
OLLAMA_URL = "http://ollama:11434"
MODELO_BASE = "qwen2.5-coder:7b"

# Instancia del modelo pequeño
llm = ChatOllama(
    model=MODELO_BASE,
    base_url=OLLAMA_URL,
    temperature=0.2 # Temperatura baja para mayor precisión técnica
)

def agente_coder(tarea: str, contexto_rag: str) -> str:
    """Agente especializado en escribir código según la documentación del RAG."""
    prompt = f"""Eres un Agente Programador Experto en Análisis Científico y Física de Partículas.
    Tu objetivo es escribir el código Python necesario para cumplir la tarea asignada.
    Debes basarte obligatoriamente en la sintaxis, ejemplos y documentación provista.
    
    [DOCUMENTACIÓN / REGLAS EXTRAÍDAS DEL RAG]
    {contexto_rag}
    
    [TAREA A REALIZAR]
    {tarea}
    
    INSTRUCCIONES:
    - Responde ÚNICAMENTE con el bloque de código Python funcional.
    - Agrega breves comentarios en el código explicando los pasos clave en tercera persona.
    - No agregues introducciones ni texto conversacional fuera del código."""
    
    respuesta = llm.invoke(prompt)
    return respuesta.content

def agente_auditor(codigo_generado: str, reglas_rag: str) -> str:
    """Agente especializado en revisar el código frente a normas de calidad o física."""
    prompt = f"""Eres un Agente Auditor de Calidad y Validación Científica.
    Tu trabajo es revisar el código Python generado por el Agente Coder y comprobar que cumple con las reglas oficiales y la sintaxis adecuada.
    
    [REGLAS Y CRITERIOS OFICIALES (DESDE RAG)]
    {reglas_rag}
    
    [CÓDIGO A AUDITAR]
    {codigo_generado}
    
    INSTRUCCIONES:
    - Analiza si el código cumple las reglas.
    - Comienza tu respuesta en la primera línea estrictamente con: [APROBADO] o [RECHAZADO].
    - Si es [RECHAZADO], enumera los errores específicos y sugiere las correcciones necesarias.
    - Si es [APROBADO], justifica brevemente por qué es correcto."""
    
    respuesta = llm.invoke(prompt)
    return respuesta.content