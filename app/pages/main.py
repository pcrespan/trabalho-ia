import sys
from pathlib import Path

_this_file = Path(__file__).resolve()
project_root = _this_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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
st.write("Ao preencher os dados na coluna à esquerda, os dois modelos (Ensemble e Regressão) deverão apresentar sua predição para definir se o cliente é de alto ou baixo risco.")

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
    st.error("Modelos não encontrados. Coloque logistic.pkl, random_forest.pkl em models / subfolders.")
    st.stop()

if st.button("Analisar", key="analyze"):
    st.session_state["show_all"] = True
    st.session_state["input_df"] = input_df
    try:
        st.session_state["results_df"] = predict_all(preprocessor, models, input_df)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.session_state["show_all"] = False

if st.session_state["show_all"]:
    results_df = st.session_state["results_df"]
    if results_df is not None:
        st.subheader("Previsão dos modelos")
        st.table(results_df.set_index("model"))

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
            try:
                df = pd.read_csv(str(training_path), sep=';')
                if "Creditability" not in df.columns:
                    st.error("training csv não contém coluna 'Creditability'")
                else:
                    X = df[FEATURE_COLUMNS].copy()
                    y_raw = df["Creditability"].astype(str).copy()
                    models_dir = Path(os.environ.get("MODELS_DIR", "../train_pipeline/models"))
                    le_path = models_dir / "label_encoder.pkl"
                    if le_path.exists():
                        label_encoder = joblib.load(le_path)
                        try:
                            y = label_encoder.transform(y_raw)
                        except Exception:
                            y = pd.to_numeric(y_raw, errors='coerce').fillna(0).astype(int)
                    else:
                        y = pd.to_numeric(y_raw, errors='coerce').fillna(0).astype(int)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    X_train_proc = preprocessor.transform(X_train)
                    X_test_proc = preprocessor.transform(X_test)
                    logistic_path = models_dir / "regression" / "logistic.pkl"
                    rf_path = models_dir / "ensemble" / "random_forest.pkl"
                    loaded = {}
                    if logistic_path.exists():
                        loaded["LogisticRegression"] = joblib.load(logistic_path)
                    if rf_path.exists():
                        loaded["RandomForest"] = joblib.load(rf_path)
                    train_scores = {}
                    test_scores = {}
                    cms = {}
                    for name, m in loaded.items():
                        y_train_pred = m.predict(X_train_proc)
                        y_test_pred = m.predict(X_test_proc)
                        train_scores[name] = accuracy_score(y_train, y_train_pred)
                        test_scores[name] = accuracy_score(y_test, y_test_pred)
                        cm = confusion_matrix(y_test, y_test_pred, labels=sorted(list(set(y_test))))
                        cms[name] = (cm, sorted(list(set(y_test))))
                    fig, ax = plt.subplots(figsize=(8, 5))
                    names = list(train_scores.keys())
                    train_vals = [train_scores[n] for n in names]
                    test_vals = [test_scores[n] for n in names]
                    x = range(len(names))
                    ax.bar([i - 0.2 for i in x], train_vals, width=0.4)
                    ax.bar([i + 0.2 for i in x], test_vals, width=0.4)
                    ax.set_xticks(list(x))
                    ax.set_xticklabels(names)
                    ax.set_ylabel("Acurácia")
                    ax.set_ylim(0, 1)
                    ax.legend(["train", "test"])
                    st.pyplot(fig)
                    st.write("Acurácias (treino / teste):")
                    for n in names:
                        st.write(f"{n}: {train_scores[n]:.4f} / {test_scores[n]:.4f}")
                    for n, (cm, labels) in cms.items():
                        fig2, ax2 = plt.subplots(figsize=(4, 4))
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
                        disp.plot(ax=ax2)
                        ax2.set_title(n)
                        st.pyplot(fig2)
            except Exception as exc:
                st.error(f"Falha ao calcular métricas: {exc}")
    except Exception as exc:
        st.error(f"Retrain failed: {exc}")
