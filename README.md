# Predição de Rendimento Agrícola com Random Forest

Este repositório faz parte de um trabalho prático de Ciência de Dados e contém:

- Um **notebook** (Kaggle ou Google Colab) com todo o processo de análise, pré-processamento e modelagem;
- Um **aplicativo em Streamlit** que carrega o modelo treinado em arquivo `.pkl` e disponibiliza uma interface de teste.

O foco do projeto é a **predição de rendimento agrícola (kg/ha)** a partir de um conjunto de dados sintético inspirado na estrutura da Produção Agrícola Municipal (PAM/IBGE).

---

## 1. Contexto do projeto

O dataset utilizado é sintético, mas foi construído para se assemelhar a dados reais da PAM/IBGE, com:

- Anos de safra (2005 a 2024);
- Municípios do Oeste de Santa Catarina;
- Diferentes culturas agrícolas (soja, milho, trigo, arroz, etc.);
- Informações de área plantada, precipitação anual, temperatura média;
- Rendimento em kg/ha e produção total em toneladas.

O objetivo é aplicar o fluxo completo de Ciência de Dados:

1. Descrição e exploração dos dados (EDA);
2. Pré-processamento;
3. Escolha do tipo de aprendizagem e algoritmo;
4. Treinamento do modelo;
5. Avaliação com métricas adequadas;
6. Discussão dos resultados e conclusão;
7. Disponibilização de uma interface para teste do modelo.

---

## 2. Estrutura do repositório

Estrutura mínima esperada:

```text
/
├── app.py                       # Código principal do aplicativo Streamlit
├── modelo_pam_multiculturas.pkl # Modelo treinado (pipeline) salvo com joblib
├── requirements.txt             # Dependências do projeto para deploy
└── README.md                    # Este arquivo
```

Outros arquivos podem existir (por exemplo, o notebook original em `.ipynb`), dependendo do ambiente de desenvolvimento.

---

## 3. Relação entre o notebook e o aplicativo Streamlit

O fluxo do projeto é o seguinte:

1. **Notebook (Kaggle/Colab)**  
   - Carrega o dataset sintético;
   - Realiza EDA (estatísticas, gráficos, correlações);
   - Define a variável alvo (`rendimento_kg_ha`) e as features numéricas e categóricas;
   - Constrói um `Pipeline` com:
     - `StandardScaler` para variáveis numéricas;
     - `OneHotEncoder` para variáveis categóricas;
     - `RandomForestRegressor` como modelo de regressão;
   - Faz a divisão em treino e teste (`train_test_split`);
   - Treina o modelo e calcula as métricas (MAE, MSE, RMSE, R²);
   - Gera gráficos (real vs predito, importância das variáveis);
   - Exporta o pipeline treinado para o arquivo `modelo_pam_multiculturas.pkl` usando `joblib.dump()`.

2. **Aplicativo Streamlit (`app.py`)**  
   - Carrega o arquivo `modelo_pam_multiculturas.pkl` usando `joblib.load()`;
   - Recupera automaticamente, quando possível, as categorias usadas no `OneHotEncoder` (municípios e culturas);
   - Cria um formulário para o usuário informar:
     - Ano;
     - Área plantada (ha);
     - Precipitação anual (mm);
     - Temperatura média (°C);
     - Município;
     - Cultura agrícola;
   - Monta um `DataFrame` com exatamente as mesmas colunas usadas no treinamento;
   - Envia esses dados para o pipeline (`model.predict()`), que internamente:
     - Padroniza as variáveis numéricas;
     - Codifica as variáveis categóricas;
     - Aplica o modelo Random Forest para prever o rendimento (kg/ha);
   - Exibe:
     - O rendimento previsto em kg/ha;
     - A produção total estimada em toneladas, calculada a partir da área plantada.

Em resumo, **o notebook é responsável por treinar e salvar o modelo**, enquanto **o app Streamlit é responsável por consumir esse modelo e disponibilizar a predição de forma interativa**.

---

## 4. Descrição do código do aplicativo (`app.py`)

O arquivo `app.py` é dividido em seções bem definidas:

### 4.1. Imports e configuração da página

- Importa as bibliotecas principais: `streamlit`, `pandas`, `numpy` e `joblib`;
- Configura título, ícone e layout da página com `st.set_page_config()`.

### 4.2. Carregamento do modelo

- Define a função `load_model()` com `@st.cache_resource` para carregar o modelo uma única vez;
- Tenta carregar o arquivo `modelo_pam_multiculturas.pkl`;
- Se o arquivo não for encontrado, exibe mensagem de erro e interrompe a execução do app.

### 4.3. Barra lateral (sidebar)

- Exibe informações sobre o trabalho (tema, tipo de modelo);
- Mostra a identificação da dupla (nome e RA);
- Inclui um pequeno guia de uso: preencher campos, clicar no botão de predição e interpretar a saída.

### 4.4. Cabeçalho e descrição do app

- Título principal da aplicação;
- Parágrafo explicando o contexto (dados inspirados na PAM/IBGE, modelo de regressão, variáveis usadas);
- Separador visual antes da seção de dados de entrada.

### 4.5. Recuperação das categorias do modelo

- Acessa o `preprocessor` dentro do pipeline, recuperando o transformador categórico (`"cat"`);
- Tenta extrair as categorias das colunas `municipio` e `cultura` diretamente do `OneHotEncoder`;
- Caso não seja possível (por exemplo, alteração na estrutura do pipeline), exibe um aviso e usa campos livres (`text_input`).

### 4.6. Formulário de entrada

- Organiza a interface em duas colunas (`st.columns(2)`);
- Campos numéricos (`st.number_input`) para:
  - Ano;
  - Área plantada (ha);
  - Precipitação (mm);
  - Temperatura média (°C);
- Campos de seleção (`st.selectbox`) ou texto (`st.text_input`) para:
  - Município;
  - Cultura agrícola;
- Após o formulário, um separador (`st.markdown("---")`) prepara a interface para a exibição do resultado.

### 4.7. Predição e exibição do resultado

- Quando o usuário clica em **“Calcular rendimento previsto”**:
  - Os dados são reunidos em um `DataFrame` com as colunas:
    - `ano`, `area_plantada_ha`, `precipitacao_mm`, `temp_media_c`, `municipio`, `cultura`;
  - O `model.predict()` é chamado para obter o rendimento em kg/ha;
  - É calculada a produção total em toneladas:

    produção (t) = (rendimento_kg_ha × área_plantada_ha) / 1000

  - O resultado é exibido usando:
    - `st.success` para destacar o rendimento previsto;
    - `st.info` para destacar a produção estimada;
  - Em caso de erro, o app mostra uma mensagem e o erro detalhado com `st.exception(e)`.

---

## 5. Execução local

Para rodar o app em ambiente local:

1. Clonar o repositório:

   ```bash
   git clone https://github.com/seu-usuario/seu-repo.git
   cd seu-repo
   ```

2. (Opcional) Criar e ativar um ambiente virtual:

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. Instalar as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Garantir que o arquivo `modelo_pam_multiculturas.pkl` está na mesma pasta que o `app.py`.

5. Executar o Streamlit:

   ```bash
   streamlit run app.py
   ```

6. Abrir o link exibido no terminal (por padrão, http://localhost:8501).

---

## 6. Deploy no Streamlit Community Cloud

1. Subir o repositório para o GitHub (incluindo `app.py`, `modelo_pam_multiculturas.pkl` e `requirements.txt`);
2. Acessar o Streamlit Community Cloud (https://streamlit.io) e conectar a conta ao GitHub;
3. Criar um novo app, informando:
   - Repositório;
   - Branch (por exemplo, `main`);
   - Caminho do arquivo principal (`app.py`);
4. Confirmar o deploy e aguardar a instalação das dependências;
5. Utilizar o link gerado para compartilhar e testar o modelo.

---

## 7. Considerações finais

Este projeto demonstra, de forma integrada, como:

- um modelo de Machine Learning pode ser treinado em um notebook (com EDA, pré-processamento, modelagem e avaliação);
- o modelo pode ser salvo em disco como um pipeline completo (incluindo transformações e algoritmo);
- um aplicativo simples em Streamlit pode ser construído para expor esse modelo de forma interativa a usuários não técnicos.

O foco didático é mostrar o ciclo completo de Ciência de Dados aplicado a um problema de predição na área agrícola.
