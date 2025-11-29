import sys
from pathlib import Path

_this_file = Path(__file__).resolve()
project_root = _this_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

from app.qwen_3 import get_model
from app.qwen_3 import answer
from app.feedback import save_feedback_row
from app.constants import FEATURE_COLUMNS, CATEGORICAL_OPTIONS, NUMERIC_RANGES
from app.utils import find_preprocessor_path, load_preprocessor, validate_row, load_models_via_load_method, predict_all

st.set_page_config(page_title="Análise de Risco de Crédito", layout="centered")

def make_input_row() -> pd.DataFrame:
    st.sidebar.header("Informações do Cliente")
    inputs = {}
    for feat in FEATURE_COLUMNS:
        if feat in NUMERIC_RANGES:
            min_v, max_v = NUMERIC_RANGES[feat]
            default = 30 if feat == "Age_years" else min_v
            val = st.sidebar.number_input(feat.replace("_", " "), min_value=min_v, max_value=max_v, value=default, step=1)
            inputs[feat] = val
        else:
            opts = CATEGORICAL_OPTIONS.get(feat)
            if opts:
                choice = st.sidebar.selectbox(feat.replace("_", " "), [d for d, _ in opts])
                code = next(code for d, code in opts if d == choice)
                inputs[feat] = code
            else:
                text = st.sidebar.text_input(feat.replace("_", " "), value="")
                try:
                    v = int(text) if text.strip() != "" else ""
                except Exception:
                    v = text
                inputs[feat] = v
    return pd.DataFrame([inputs], columns=FEATURE_COLUMNS)

st.title("Análise de Risco de Crédito")
st.write("Ao preencher os dados na coluna à esquerda, os três modelos (Ensemble, MLP e Regressão) deverão apresentar sua predição para definir se o cliente é de alto ou baixo risco.")

input_df = make_input_row()
valid, msg = validate_row(input_df)
if not valid:
    st.error(msg)
    st.stop()

try:
    preproc_path = find_preprocessor_path()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

preprocessor = load_preprocessor(preproc_path)

models = load_models_via_load_method()
if not models:
    st.error("No models found. Place logistic.pkl, random_forest.pkl and/or mlp.pt in models/ subfolders.")
    st.stop()

if st.button("Predict"):
    try:
        results_df = predict_all(preprocessor, models, input_df)
        st.subheader("Previsão dos modelos")
        st.table(results_df.set_index("model"))
        model, tokenizer = get_model()
        st.markdown("## Análise via LLM")
        with st.spinner("Carregando análise da LLM"):
            analysis = answer(model, tokenizer, results_df, input_df)
            st.write(analysis)
        st.markdown("### Feedback")
        chosen_label = st.selectbox("Qual classificação será considerada?", ["Good", "Bad"], index=0)
        model_wrong = st.selectbox("O modelo errou na classificação?", ["Não", "Sim"], index=0)
        if st.button("Enviar feedback"):
            input_row = {col: input_df.iloc[0][col] for col in input_df.columns}
            if model_wrong == "Sim":
                save_feedback_row(input_row, chosen_label)
                st.success("Feedback registrado: modelo marcado como erro e linha adicionada ao CSV de treino.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
