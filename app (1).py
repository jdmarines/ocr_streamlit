import streamlit as st
from groq import Groq
import easyocr
from PIL import Image
import numpy as np
import io

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Asistente de Imágenes con IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funciones Principales ---

@st.cache_resource
def cargar_modelo_ocr():
    """
    Carga el modelo de EasyOCR en caché para no tener que recargarlo
    cada vez que se ejecuta el script.
    """
    reader = easyocr.Reader(['es', 'en'], gpu=False) # Soporte para español e inglés, sin GPU
    return reader

def extraer_texto_de_imagen(imagen_bytes, reader):
    """
    Utiliza EasyOCR para extraer texto de una imagen proporcionada en bytes.
    """
    try:
        resultado = reader.readtext(imagen_bytes)
        texto_extraido = " ".join([res[1] for res in resultado])
        return texto_extraido
    except Exception as e:
        st.error(f"Error al procesar la imagen con OCR: {e}")
        return None

def obtener_respuesta_groq(cliente, texto):
    """
    Envía el texto extraído a la API de Groq y obtiene una respuesta del LLM.
    """
    if not texto:
        return "No se pudo extraer texto de la imagen para analizar."

    prompt = f"""
    Eres un asistente experto en analizar y resumir texto. A continuación, se te proporciona un texto extraído de una imagen.
    Tu tarea es analizarlo, corregir posibles errores de OCR si es evidente, y proporcionar un resumen claro y conciso o las ideas principales.
    Si el texto parece ser una pregunta, respóndela. Si son datos, organízalos.

    Texto extraído de la imagen:
    ---
    {texto}
    ---

    Análisis y respuesta:
    """

    try:
        chat_completion = cliente.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="qwen/qwen3-32b", # Modelo rápido y eficiente de Groq
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error al contactar la API de Groq. Verifica tu API Key. Detalle: {e}")
        return None

# --- Interfaz de Usuario de Streamlit ---

st.title("🤖 Asistente Inteligente de Imágenes")
st.markdown("Sube una imagen, y la IA extraerá el texto, lo analizará y te dará una respuesta inteligente.")

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.header("🔑 Configuración")
    st.markdown("Introduce tu API Key de Groq para activar el asistente de IA.")
    
    # Campo para la API Key de Groq
    groq_api_key = st.text_input("Groq API Key", type="password", help="Obtén tu clave desde https://console.groq.com/keys")
    
    st.markdown("---")
    st.header("📄 Instrucciones")
    st.markdown("""
    1.  **Ingresa tu API Key** de Groq en el campo de arriba.
    2.  **Sube una imagen** (JPG, PNG) en el área principal.
    3.  Haz clic en el botón **"Procesar Imagen"**.
    4.  Espera a que la IA analice la imagen y te muestre los resultados.
    """)
    st.markdown("---")
    st.info("Tu API Key es privada y no se almacena en ningún lugar.")


# --- Área Principal de la Aplicación ---
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("🖼️ Carga tu Imagen Aquí")
    uploaded_file = st.file_uploader(
        "Selecciona un archivo de imagen", 
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Mostrar la imagen cargada
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        
        # Convertir el archivo a bytes para el procesamiento
        image_bytes = uploaded_file.getvalue()

        if st.button("✨ Procesar Imagen", use_container_width=True, type="primary"):
            if not groq_api_key:
                st.warning("Por favor, introduce tu API Key de Groq en la barra lateral para continuar.")
            else:
                # Inicializar cliente de Groq
                client = Groq(api_key=groq_api_key)
                
                # Cargar modelo OCR
                reader = cargar_modelo_ocr()

                # Procesamiento con OCR
                with st.spinner("🔍 Extrayendo texto de la imagen..."):
                    texto_ocr = extraer_texto_de_imagen(image_bytes, reader)
                
                if texto_ocr:
                    st.session_state.texto_ocr = texto_ocr
                    
                    # Procesamiento con LLM
                    with st.spinner("🧠 El asistente de IA está pensando..."):
                        respuesta_ia = obtener_respuesta_groq(client, texto_ocr)
                        st.session_state.respuesta_ia = respuesta_ia
                else:
                    st.error("No se pudo extraer texto de la imagen. Intenta con otra imagen más clara.")


with col2:
    st.subheader("💡 Resultados del Análisis")
    if "texto_ocr" in st.session_state and st.session_state.texto_ocr:
        with st.expander("Ver texto extraído por OCR", expanded=False):
            st.text_area(
                "Texto extraído:", 
                st.session_state.texto_ocr, 
                height=150,
                disabled=True
            )
    
    if "respuesta_ia" in st.session_state and st.session_state.respuesta_ia:
        st.markdown("#### Respuesta del Asistente:")
        st.markdown(st.session_state.respuesta_ia)
    else:
        st.info("Los resultados del análisis de la IA aparecerán aquí después de procesar una imagen.")
 
   
