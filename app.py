import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# Configuração da página
# =========================================================
st.set_page_config(
    page_title="Predição de rendimento agrícola (kg/ha)",
    layout="centered",
)


# =========================================================
# Nome do arquivo CSV usado como base para treino
# =========================================================
CSV_FILENAME = "pam_sintetico_multiculturas_50k.csv"


# =========================================================
# Carregar dataset com cache (usado no treino e nas análises)
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv(CSV_FILENAME)


df_full = load_data()


# =========================================================
# Função auxiliar: treinar modelo do zero (fallback)
# =========================================================
def treinar_modelo_do_zero():
    """
    Treina um novo modelo RandomForest para rendimento_kg_ha
    a partir do arquivo CSV definido em CSV_FILENAME.
    É usado como fallback caso o .pkl não possa ser carregado.
    (Sem exibir aviso na interface.)
    """
    # Usa o dataset já carregado em df_full
    df = df_full.copy()

    target = "rendimento_kg_ha"
    numeric_features = ["ano", "area_plantada_ha", "precipitacao_mm", "temp_media_c"]
    categorical_features = ["municipio", "cultura"]

    X = df[numeric_features + categorical_features]
    y = df[target]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    rf_regressor = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", rf_regressor),
        ]
    )

    model.fit(X, y)

    # Opcional: tentar salvar um novo .pkl compatível com o ambiente
    try:
        joblib.dump(model, "modelo_pam_multiculturas_streamlit.pkl", compress=3)
    except Exception:
        pass

    return model


# =========================================================
# Carregar modelo treinado (com fallback silencioso para treino)
# =========================================================
@st.cache_resource
def load_model():
    """
    Tenta carregar o modelo do arquivo modelo_pam_multiculturas.pkl.
    Se falhar (por incompatibilidade de versão, pickle, etc.),
    treina um novo modelo a partir do CSV, sem mostrar aviso na tela.
    """
    try:
        return joblib.load("modelo_pam_multiculturas.pkl")
    except Exception:
        return treinar_modelo_do_zero()


model = load_model()


# =========================================================
# Sidebar – informações do trabalho
# =========================================================
with st.sidebar:
    st.markdown("### Trabalho de Ciência de Dados")
    st.markdown("Tema: Predição de rendimento agrícola (PAM sintético)")
    st.markdown("Tipo de modelo: Regressão (Random Forest)")
    st.markdown("Dupla:")
    st.markdown("- Jalisson Ternus – RA 405155")
    st.markdown("- Geslon Gish – RA 395124")
    st.markdown("---")
    st.markdown("### Como usar o aplicativo")
    st.markdown(
        """
        1. Preencha os campos com os dados da safra desejada.  
        2. Clique em Calcular rendimento previsto.  
        3. Observe o rendimento estimado, a produção total e a sugestão de culturas.
        """
    )


# =========================================================
# Cabeçalho principal
# =========================================================
st.title("Predição de rendimento agrícola (kg/ha)")

st.markdown(
    """
Este aplicativo faz parte de um trabalho prático de Ciência de Dados, no qual foi treinado
um modelo de regressão (Random Forest) para estimar o rendimento agrícola (kg/ha) de diferentes
culturas a partir de dados sintéticos.

Os dados são inspirados na Produção Agrícola Municipal (PAM/IBGE), considerando variáveis
como ano, município, cultura, área plantada, precipitação anual e temperatura média.
"""
)

st.markdown("---")
st.markdown("## Dados de entrada")


# =========================================================
# Recuperar categorias do OneHotEncoder (se possível)
# =========================================================
municipios = None
culturas = None

try:
    preprocessor = model.named_steps["preprocessor"]
    cat_transformer = preprocessor.named_transformers_["cat"]
    categorias_cat = cat_transformer.categories_
    municipios = list(categorias_cat[0])  # categorias da coluna 'municipio'
    culturas = list(categorias_cat[1])    # categorias da coluna 'cultura'
except Exception:
    st.info(
        "Não foi possível carregar automaticamente a lista de municípios e culturas. "
        "Os campos serão preenchidos manualmente."
    )


# =========================================================
# Formulário de entrada
# =========================================================
col1, col2 = st.columns(2)

with col1:
    ano = st.number_input(
        "Ano da safra",
        min_value=2005,
        max_value=2100,
        value=2024,
        step=1,
    )

    area_plantada_ha = st.number_input(
        "Área plantada (hectares)",
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


# =========================================================
# Função auxiliar: recomendar melhores culturas
# =========================================================
def recomendar_culturas(municipio, ano, df):
    """
    Retorna um ranking das culturas com maior rendimento médio
    para o município e um intervalo de anos em torno do ano informado.
    """
    # Filtra por município e janela de anos (ex: ano-5 até ano)
    janela_inicio = ano - 5
    if janela_inicio < df["ano"].min():
        janela_inicio = df["ano"].min()

    filtro = df[
        (df["municipio"] == municipio)
        & (df["ano"].between(janela_inicio, ano))
    ]

    # Se não houver dados suficientes, usa todo o dataset
    if filtro.empty:
        filtro = df.copy()

    ranking = (
        filtro.groupby("cultura")["rendimento_kg_ha"]
        .mean()
        .sort_values(ascending=False)
    )

    return ranking


# =========================================================
# Predição + recomendações
# =========================================================
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

        st.success(f"Rendimento previsto: {rendimento_previsto:,.2f} kg/ha")
        st.info(
            f"Produção total estimada: {producao_t_estimado:,.2f} toneladas "
            f"para {area_plantada_ha:,.1f} hectares."
        )

        # ----------------------------------------
        # Segunda análise: recomendação de culturas
        # ----------------------------------------
        st.markdown("### Sugestão de culturas para este cenário")

        ranking_culturas = recomendar_culturas(municipio, ano, df_full)
        top3 = ranking_culturas.head(3)

        if not top3.empty:
            st.write(
                f"Com base no histórico do município {municipio} "
                f"e anos próximos de {ano}, as culturas com maior "
                f"rendimento médio (kg/ha) foram:"
            )
            st.table(top3.reset_index().rename(
                columns={"cultura": "Cultura", "rendimento_kg_ha": "Rendimento médio (kg/ha)"}
            ))
        else:
            st.write(
                "Não foi possível calcular recomendações específicas. "
                "Verifique se o município informado existe no dataset."
            )

    except Exception as e:
        st.error("Ocorreu um erro ao realizar a predição. Verifique os dados de entrada.")
        st.exception(e)
