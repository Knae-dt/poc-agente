import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuración de rutas dentro del contenedor
PATH_DOCUMENTOS = "./documentos"
PATH_DB = "./chroma_db"
MODELO_EMBEDDINGS = "all-MiniLM-L6-v2"

def inicializar_embeddings():
    """Descarga/carga el modelo local de embeddings."""
    print("Cargando modelo de embeddings (all-MiniLM-L6-v2)...")
    return HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)

def construir_base_vectorial():
    """Lee los PDFs de la carpeta y crea la base de datos ChromaDB."""
    print(f"\n--- 1. Leyendo archivos PDF desde '{PATH_DOCUMENTOS}' ---")
    if not os.path.exists(PATH_DOCUMENTOS) or not os.listdir(PATH_DOCUMENTOS):
        print(f"ERROR: La carpeta '{PATH_DOCUMENTOS}' está vacía. Agrega al menos un PDF.")
        return None

    loader = PyPDFDirectoryLoader(PATH_DOCUMENTOS)
    docs = loader.load()
    print(f"Éxito: Se cargaron {len(docs)} páginas de documentos.")

    print("\n--- 2. Dividiendo el texto en fragmentos (Chunking) ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"Éxito: Documentos divididos en {len(chunks)} fragmentos.")

    print("\n--- 3. Generando Vectores y Guardando en ChromaDB ---")
    embeddings = inicializar_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PATH_DB
    )
    print(f"¡ÉXITO TOTAL! Base vectorial creada y guardada en '{PATH_DB}'.")
    return vector_store

def buscar_contexto(pregunta: str, k: int = 3) -> str:
    """Función que usará el Compañero 2 para consultar el RAG."""
    embeddings = inicializar_embeddings()
    vector_store = Chroma(persist_directory=PATH_DB, embedding_function=embeddings)
    
    docs_relacionados = vector_store.similarity_search(pregunta, k=k)
    contexto = "\n\n".join([doc.page_content for doc in docs_relacionados])
    return contexto

if __name__ == "__main__":
    # Prueba autónoma: si se ejecuta este archivo directamente, construye la base y hace un test
    vector_db = construir_base_vectorial()
    
    if vector_db:
        print("\n--- 4. Prueba de búsqueda RAG interna ---")
        pregunta_test = "¿De qué trata este documento?"
        resultado = buscar_contexto(pregunta_test, k=2)
        print(f"Pregunta: {pregunta_test}")
        print("Fragmento recuperado del RAG:\n")
        print(resultado[:300] + "...") # Muestra los primeros 300 caracteres