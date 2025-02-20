import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Configurar la API Key de OpenAI
os.environ["OPENAI_API_KEY"] = "API_Key poner aqui"

# Cargar el documento de la circular
circular_path = "texto_circular.txt"
try:
    with open(circular_path, "r", encoding="utf-8") as file:
        texto = file.read()
except FileNotFoundError:
    st.error(f"Error: No se encontró el archivo {circular_path}")
    st.stop()

# Procesar el documento para convertirlo en fragmentos
text_splitter = CharacterTextSplitter(chunk_size=1400, chunk_overlap=50)
splits = text_splitter.split_text(texto)

# Crear la base de datos vectorial para búsqueda eficiente
vectorstore = FAISS.from_texts(splits, OpenAIEmbeddings())

# Implementar la funcionalidad de RAG con LangChain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

def preguntar_chatbot(pregunta):
    respuesta = qa.invoke({"query": pregunta})["result"]
    return respuesta

# Interfaz con Streamlit
st.title("Chatbot de Consulta sobre la Circular")
st.write("Haz una pregunta sobre la circular y obtén una respuesta basada en su contenido.")

pregunta = st.text_input("Escribe tu pregunta aquí:")
if st.button("Consultar"):
    if pregunta:
        respuesta = preguntar_chatbot(pregunta)
        st.write("**Respuesta:**", respuesta)
    else:
        st.warning("Por favor, ingresa una pregunta.")
