
    import os
    import json
    import re
    import streamlit as st
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter
    import easyocr
    import numpy as np

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    st.set_page_config(page_title="OCR + LLM Analyzer", page_icon="🧠", layout="centered")
    st.title("🖼️ → 📝 OCR + LLM Analyzer")
    st.caption("Sube una imagen, extrae el texto con OCR y pídele a un LLM (Groq o Hugging Face) que lo analice.")

    with st.sidebar:
        st.header("⚙️ Configuración")
        ocr_langs = st.multiselect(
            "Idiomas esperados en la imagen (OCR)",
            options=["es", "en", "pt", "fr", "de", "it"],
            default=["es", "en"]
        )
        contrast_boost = st.slider("Aumentar contraste (%)", 0, 150, 25, step=5)
        apply_sharpen = st.checkbox("Aplicar nitidez (sharpen)", value=True)
        provider = st.selectbox("Proveedor LLM", ["Groq", "Hugging Face", "Auto (Groq→HF)"], index=0)
        model_groq = st.text_input("Modelo Groq", value="llama-3.1-8b-instant")
        model_hf = st.text_input("Modelo Hugging Face", value="meta-llama/Meta-Llama-3.1-8B-Instruct")
        temperature = st.slider("Creatividad (temperature)", 0.0, 1.5, 0.3, 0.1)
        max_tokens = st.number_input("Máx. tokens respuesta", min_value=128, max_value=4096, value=512, step=64)
        st.markdown("---")
        st.write("**Variables de entorno (no pegues claves aquí)**")
        st.code("GROQ_API_KEY=...
HF_API_TOKEN=...")

    uploaded = st.file_uploader("Sube una imagen (PNG/JPG)", type=["png", "jpg", "jpeg"])

    def preprocess(img, contrast_boost=25, sharpen=True):
        img = ImageOps.exif_transpose(img)
        img = ImageOps.autocontrast(img, cutoff=1)
        if contrast_boost and contrast_boost > 0:
            img = ImageEnhance.Color(img).enhance(0.0)  # blanco y negro
            img = ImageEnhance.Contrast(img).enhance(1 + contrast_boost/100.0)
        if sharpen:
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
        return img

    @st.cache_resource(show_spinner=False)
    def get_reader(langs):
        return easyocr.Reader(langs, gpu=False)

    def clean_text(s: str) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def analyze_with_groq(prompt: str, model: str, temperature: float, max_tokens: int):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY no configurada (usa .env o variable de entorno).")
        from groq import Groq
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un analista lingüístico conciso. Devuelve JSON válido y nada más."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content

    def analyze_with_hf(prompt: str, model: str, temperature: float, max_tokens: int):
        token = os.getenv("HF_API_TOKEN")
        if not token:
            raise RuntimeError("HF_API_TOKEN no configurado (usa .env o variable de entorno).")
        import requests
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": int(max_tokens),
                "temperature": float(temperature),
                "return_full_text": False
            }
        }
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Hugging Face API error {r.status_code}: {r.text[:200]}")
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        return json.dumps(data)

    def build_prompt(extracted_text: str):
        return f'''
        Analiza el siguiente texto OCR y devuelve **solo** un JSON válido con estas claves:
        - "idioma": código ISO esperado (es, en, etc.).
        - "limpio": el texto depurado (sin saltos raros, sin artefactos).
        - "resumen": resumen en 1-2 oraciones en el mismo idioma detectado.
        - "sentimiento": "positivo" | "neutral" | "negativo" y "confianza" 0-1.
        - "entidades": lista de objetos con "tipo" y "texto" (personas, organizaciones, lugares, fechas, montos).
        - "palabras_clave": lista de 3-8 términos relevantes.
        - "acciones_sugeridas": lista breve de siguientes pasos (si aplica).

        Texto OCR:
        """{extracted_text}"""
        '''

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.subheader("Vista previa")
        st.image(img, use_container_width=True)

        pre = preprocess(img, contrast_boost, apply_sharpen)
        with st.expander("Ver imagen preprocesada"):
            st.image(pre, use_container_width=True)

        with st.spinner("Ejecutando OCR..."):
            reader = get_reader(ocr_langs if ocr_langs else ["es", "en"])
            results = reader.readtext(np.array(pre), detail=1, paragraph=True)

        # results: list of [bbox, text, conf]
        try:
            results_sorted = sorted(results, key=lambda r: np.mean([p[1] for p in r[0]]))
            text = " ".join([r[1] for r in results_sorted])
        except Exception:
            text = " ".join([r[1] for r in results])

        text = clean_text(text)
        st.subheader("Texto extraído")
        if not text:
            st.warning("No se detectó texto. Revisa la calidad de la imagen o cambia los idiomas de OCR.")
            st.stop()
        st.code(text)

        st.subheader("Análisis con LLM")
        prompt = build_prompt(text)

        run_llm = st.button("Analizar con LLM", type="primary")
        if run_llm:
            provider_choice = provider
            last_err = None
            response_text = None

            if provider_choice in ("Groq", "Auto (Groq→HF)"):
                try:
                    with st.spinner("Consultando Groq..."):
                        response_text = analyze_with_groq(prompt, model_groq, temperature, max_tokens)
                except Exception as e:
                    last_err = str(e)
                    if provider_choice == "Groq":
                        st.error(last_err)

            if (response_text is None) and (provider_choice in ("Hugging Face", "Auto (Groq→HF)")):
                try:
                    with st.spinner("Consultando Hugging Face..."):
                        response_text = analyze_with_hf(prompt, model_hf, temperature, max_tokens)
                except Exception as e:
                    last_err = str(e)

            if response_text is None:
                st.error("No fue posible analizar con LLM. " + (last_err or ""))
                st.stop()

            try:
                data = json.loads(response_text)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", response_text)
                if m:
                    data = json.loads(m.group(0))
                else:
                    st.warning("La respuesta no fue JSON válido. Mostrando texto bruto:")
                    st.write(response_text)
                    st.stop()

            st.success("¡Análisis listo!")
            st.json(data)

            st.markdown("### Resultados")
            col1, col2 = st.columns(2)
            with col1:
                idioma = data.get("idioma", "")
                resumen = data.get("resumen", "")
                st.markdown(f"**Idioma:** {idioma}")
                st.markdown(f"**Resumen:** {resumen}")
                st.markdown(f"**Sentimiento:** {data.get('sentimiento','')} (confianza: {data.get('confianza', data.get('confidence',''))})")
            with col2:
                kws = data.get("palabras_clave", [])
                ents = data.get("entidades", [])
                if kws:
                    st.markdown("**Palabras clave:** " + ", ".join(kws))
                if ents:
                    st.markdown("**Entidades detectadas:**")
                    for e in ents:
                        st.write(f"- {e.get('tipo','?')}: {e.get('texto','')}")

            if data.get("acciones_sugeridas"):
                st.markdown("### Acciones sugeridas")
                for a in data["acciones_sugeridas"]:
                    st.write(f"- {a}")

    else:
        st.info("📤 Sube una imagen para comenzar. Formatos admitidos: PNG/JPG.")
