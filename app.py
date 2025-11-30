import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# Configuração da página
# =========================================================
st.set_page_config(
    page_title="Predição de rendimento agrícola (kg/ha)",
    page_icon="🌾",
    layout="centered",
)

# =========================================================
# Carregar modelo treinado
# =========================================================
@st.cache_resource
def load_model():
    model = joblib.load("modelo_pam_multiculturas.pkl")
    return model

model = None
try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Arquivo **modelo_pam_multiculturas.pkl** não encontrado.\n\n"
        "Coloque o arquivo na mesma pasta do app antes de executar."
    )

# =========================================================
# Sidebar – informações do trabalho
# =========================================================
with st.sidebar:
    st.markdown("### Trabalho de Ciência de Dados")
    st.markdown("**Tema:** Predição de rendimento agrícola (PAM sintético)")
    st.markdown("**Disciplina:** Ciência de Dados")
    st.markdown("**Dupla:**")
    st.markdown("- Jalisson Ternus – RA: 405155")
    st.markdown("- Geslon Gish – RA: 395124")
    st.markdown("---")
    st.markdown("### Como usar o app")
    st.markdown(
        """
        1. Preencha os campos com os valores desejados.\n
        2. Clique em **Calcular rendimento previsto**.\n
        3. Veja o resultado em kg/ha e a estimativa de produção total em toneladas.
        """
    )

# =========================================================
# Cabeçalho principal
# =========================================================
st.title("🌾 Predição de rendimento agrícola (kg/ha)")

st.markdown(
    """
Este aplicativo faz parte de um trabalho prático de **Ciência de Dados**, 
no qual foi treinado um modelo de regressão (Random Forest) para estimar o 
**rendimento agrícola (kg/ha)** de diferentes culturas a partir de dados sintéticos.

Os dados utilizados são inspirados na Produção Agrícola Municipal (PAM), 
considerando variáveis como ano, município, cultura, área plantada, 
precipitação anual e temperatura média.
"""
)

if model is None:
    st.stop()

# Tentar recuperar categorias do OneHotEncoder
municipios = None
culturas = None

try:
    preprocessor = model.named_steps["preprocessor"]
    cat_transformer = preprocessor.named_transformers_["cat"]
    categorias_cat = cat_transformer.categories_
    municipios = list(categorias_cat[0])  # ['municipio']
    culturas = list(categorias_cat[1])    # ['cultura']
except Exception:
    st.warning(
        "Não foi possível carregar automaticamente a lista de municípios/culturas. "
        "Os campos serão livres, mas é importante usar valores compatíveis com o treino."
    )

st.markdown("## Dados de entrada")

col1, col2 = st.columns(2)

with col1:
    ano = st.number_input(
        "Ano da safra",
        min_value=2000,
        max_value=2100,
        value=2024,
        step=1,
    )

    area_plantada_ha = st.number_input(
        "Área plantada (ha)",
        min_value=0.1,
        max_value=1_000_000.0,
        value=100.0,
        step=1.0,
    )

    temp_media_c = st.number_input(
        "Temperatura média anual (°C)",
        min_value=-10.0,
        max_value=50.0,
        value=20.0,
        step=0.1,
    )

with col2:
    precipitacao_mm = st.number_input(
        "Precipitação anual (mm)",
        min_value=0.0,
        max_value=10_000.0,
        value=1800.0,
        step=10.0,
    )

    if municipios:
        municipio = st.selectbox("Município", options=municipios)
    else:
        municipio = st.text_input("Município", value="São Miguel do Oeste")

    if culturas:
        cultura = st.selectbox("Cultura agrícola", options=culturas)
    else:
        cultura = st.text_input("Cultura agrícola", value="Soja")

st.markdown("---")

if st.button("Calcular rendimento previsto"):
    input_data = pd.DataFrame(
        {
            "ano": [ano],
            "area_plantada_ha": [area_plantada_ha],
            "precipitacao_mm": [precipitacao_mm],
            "temp_media_c": [temp_media_c],
            "municipio": [municipio],
            "cultura": [cultura],
        }
    )

    try:
        y_pred = model.predict(input_data)[0]
        rendimento_previsto = float(y_pred)

        producao_t_estimado = (rendimento_previsto * area_plantada_ha) / 1000.0

        st.success(f"**Rendimento previsto:** {rendimento_previsto:,.2f} kg/ha")
        st.info(
            f"**Produção total estimada:** {producao_t_estimado:,.2f} toneladas "
            f"para {area_plantada_ha:,.1f} ha."
        )

    except Exception as e:
        st.error("Ocorreu um erro ao realizar a predição. Verifique os dados de entrada.")
        st.exception(e)
