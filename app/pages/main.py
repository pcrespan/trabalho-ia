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
from app.train import train_and_persist_models
from app.constants import FEATURE_COLUMNS, CATEGORICAL_OPTIONS, NUMERIC_RANGES
from app.utils import find_preprocessor_path, load_preprocessor, validate_row, load_models_via_load_method, predict_all

st.set_page_config(page_title="Análise de Risco de Crédito", layout="centered")

if "show_all" not in st.session_state:
    st.session_state["show_all"] = False
if "llm_analysis" not in st.session_state:
    st.session_state["llm_analysis"] = ""
if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "input_df" not in st.session_state:
    st.session_state["input_df"] = None
if "models_retrained" not in st.session_state:
    st.session_state["models_retrained"] = False

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
                choice = st.sidebar.selectbox(feat.replace("_", " "), [d for d, _ in opts], key=f"sb_{feat}")
                code = next(code for d, code in opts if d == choice)
                inputs[feat] = code
            else:
                text = st.sidebar.text_input(feat.replace("_", " "), value="", key=f"ti_{feat}")
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
    st.error("Modelos não encontrados. Coloque logistic.pkl, random_forest.pkl e mlp.pt em models / subfolders.")
    st.stop()

if st.button("Analisar", key="analyze"):
    st.session_state["show_all"] = True
    st.session_state["input_df"] = input_df
    try:
        st.session_state["results_df"] = predict_all(preprocessor, models, input_df)
        model, tokenizer = get_model()
        st.session_state["llm_analysis"] = answer(model, tokenizer, st.session_state["results_df"], st.session_state["input_df"])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.session_state["show_all"] = False

if st.session_state["show_all"]:
    results_df = st.session_state["results_df"]
    if results_df is not None:
        st.subheader("Previsão dos modelos")
        st.table(results_df.set_index("model"))
    st.markdown("## Análise via LLM")
    if st.session_state.get("llm_analysis"):
        st.write(st.session_state["llm_analysis"])
    else:
        st.write("Nenhuma análise disponível.")

    st.markdown("### Feedback")
    chosen_label = st.selectbox("Qual classificação será considerada?", ["Good", "Bad"], index=0, key="label_choice")
    model_wrong = st.selectbox("O modelo errou na classificação?", ["Não", "Sim"], index=0, key="wrong_choice")

    if st.button("Enviar feedback", key="send_feedback"):
        st.session_state["show_all"] = True
        saved_df = st.session_state.get("input_df")
        saved_df["Creditability"] = chosen_label
        row_source = saved_df if saved_df is not None else input_df
        input_row = {col: row_source.iloc[0][col] for col in row_source.columns}
        if model_wrong == "Sim":
            try:
                save_feedback_row(input_row, chosen_label)
                st.success("Feedback registrado: modelo marcado como erro e linha adicionada ao CSV de treino.")
            except Exception as exc:
                st.error(f"Falha ao salvar feedback: {exc}")
        else:
            st.info("Feedback não registrado porque o modelo não foi marcado como erro.")

st.markdown("### Treinar modelos com training_data.csv")
csv_path = project_root / "data/training_data.csv"
if st.button("Retrain models", key="retrain"):
    try:
        training_path = Path(st.session_state.get("training_csv_path", csv_path))
        if not training_path.exists():
            st.error(f"{training_path} não encontrado")
        else:
            result = train_and_persist_models(str(training_path))
            st.success("Treinamento concluído e modelos salvos.")
            st.session_state["models_retrained"] = True
            try:
                models = load_models_via_load_method()
            except Exception:
                st.error("Não foi possível carregar os modelos.")
    except Exception as exc:
        st.error(f"Retrain failed: {exc}")
