import os
import io
import json
import time
import base64
import streamlit as st
import pandas as pd
from PIL import Image
from openai import OpenAI

st.set_page_config(page_title="ReClasifica Lab - Clasificador con IA", page_icon="♻️", layout="wide")

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
) if api_key else None

MODELOS = ["gemini-flash-lite-latest", "gemini-flash-latest",
           "gemini-3.5-flash-lite", "gemini-3-flash-preview"]

CATEGORIAS = ["Plástico", "Papel", "Vidrio", "Otros / revisión"]

PROMPT = """Eres un clasificador de residuos para una planta de reciclaje.
Analiza la imagen y clasifica el residuo principal en una de estas categorias:
Plástico, Papel, Vidrio u Otros / revisión (si el material no es identificable o es otro).
Usa el color y la forma del objeto como criterios principales.
Responde SOLO con un JSON valido, sin texto adicional ni comillas invertidas, con este formato:
{"categoria": "Plástico", "color": "azul", "forma": "botella", "cantidad_objetos": 1, "confianza": "Alta", "razon": "una frase corta"}
La confianza debe ser Alta, Media o Baja."""


def a_data_url(archivo):
    img = Image.open(archivo).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def clasificar(data_url):
    ultimo_error = None
    for modelo in MODELOS:
        try:
            r = client.chat.completions.create(
                model=modelo,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]}],
            )
            texto = r.choices[0].message.content.strip()
            texto = texto.replace("```json", "").replace("```", "").strip()
            datos = json.loads(texto)
            if datos.get("categoria") not in CATEGORIAS:
                datos["categoria"] = "Otros / revisión"
            return datos, None
        except Exception as e:
            ultimo_error = e
    return None, ultimo_error


st.title("♻️ ReClasifica Lab: clasificador de residuos con IA")
st.caption("Sube fotos de residuos y la IA los clasifica por color y forma en plástico, papel, vidrio u otros.")
st.sidebar.write("Estado de la IA:", "✅ conectada" if client else "❌ sin API key")
st.sidebar.info("Máximo 10 imágenes por análisis (límite del plan gratuito).")

archivos = st.file_uploader("Sube una o varias imágenes (JPG o PNG)",
                            type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if archivos and len(archivos) > 10:
    st.warning("Solo se analizarán las primeras 10 imágenes.")
    archivos = archivos[:10]

if archivos:
    cols = st.columns(min(len(archivos), 5))
    for i, a in enumerate(archivos):
        cols[i % len(cols)].image(a, caption=a.name, use_container_width=True)

if st.button("Clasificar imágenes", type="primary", disabled=not archivos):
    if not client:
        st.error("Falta la API key. Agrégala en Settings > Secrets como GEMINI_API_KEY.")
    else:
        filas = []
        barra = st.progress(0.0, text="Analizando...")
        for i, a in enumerate(archivos):
            datos, error = clasificar(a_data_url(a))
            if datos:
                filas.append({
                    "Imagen": a.name,
                    "Categoría": datos.get("categoria"),
                    "Color": datos.get("color", ""),
                    "Forma": datos.get("forma", ""),
                    "Objetos": datos.get("cantidad_objetos", 1),
                    "Confianza": datos.get("confianza", ""),
                    "Razón": datos.get("razon", ""),
                })
            else:
                filas.append({"Imagen": a.name, "Categoría": "Otros / revisión", "Color": "",
                              "Forma": "", "Objetos": 0, "Confianza": "Baja",
                              "Razón": f"Error: {error}"})
            barra.progress((i + 1) / len(archivos), text=f"Analizando {i + 1} de {len(archivos)}")
            time.sleep(1)
        barra.empty()
        st.session_state["resultados"] = pd.DataFrame(filas)

if "resultados" in st.session_state:
    df = st.session_state["resultados"]
    st.subheader("Resultados")

    conteo = df["Categoría"].value_counts().reindex(CATEGORIAS, fill_value=0)
    total = int(conteo.sum())
    cols = st.columns(4)
    for col, cat in zip(cols, CATEGORIAS):
        pct = (conteo[cat] / total * 100) if total else 0
        col.metric(cat, int(conteo[cat]), f"{pct:.1f}%")

    st.bar_chart(conteo)
    st.dataframe(df, use_container_width=True)
    st.download_button("Descargar resultados (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "resultados_residuos.csv", "text/csv")
    st.caption("La IA puede equivocarse (por ejemplo, plástico transparente vs vidrio). "
               "Los resultados de confianza baja requieren revisión humana.")
