# app/utils.py
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import joblib
import importlib
import pandas as pd
import numpy as np
import torch

from app.constants import PREPROCESSOR_CANDIDATES, MODEL_FILENAMES, FEATURE_COLUMNS, NUMERIC_RANGES


def find_preprocessor_path() -> Path:
    candidates = [Path(p) for p in PREPROCESSOR_CANDIDATES]
    for p in candidates:
        if p.exists():
            return p.resolve()
    # fallback: recursive search from project root
    matches = list(Path.cwd().rglob("preprocessor.joblib"))
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(
        "preprocessor.joblib not found. Searched configured locations and project tree."
    )


def load_preprocessor(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor file not found at: {path}")
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load preprocessor from {path}: {e}")


def validate_row(df_row: pd.DataFrame) -> Tuple[bool, str]:
    missing = [c for c in FEATURE_COLUMNS if c not in df_row.columns]
    if missing:
        return False, f"Missing fields: {missing}"
    for col, (min_v, max_v) in NUMERIC_RANGES.items():
        if col in df_row.columns:
            val = df_row.iloc[0][col]
            try:
                v = float(val)
            except Exception:
                return False, f"Invalid numeric value for {col}"
            if not (min_v <= v <= max_v):
                return False, f"Value for {col} out of range ({min_v}–{max_v})"
    return True, ""


def _try_import_model_class(module_basenames: list, class_name: str):
    for base in module_basenames:
        try:
            mod = importlib.import_module(base)
            cls = getattr(mod, class_name)
            return cls
        except Exception:
            continue
    return None


def _load_sklearn_artifact(path: Path):
    payload = joblib.load(path)
    return payload


def _load_torch_checkpoint(path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        return checkpoint
    # if raw state_dict returned, wrap it
    return {"model_state_dict": checkpoint}


def load_models_via_load_method() -> Dict[str, Any]:
    loaded: Dict[str, Any] = {}

    # try loading sklearn logistic/ensemble via joblib or classmethod if available
    for name in ("logistic", "ensemble"):
        for candidate in MODEL_FILENAMES.get(name, []):
            p = Path(candidate)
            if not p.exists():
                continue
            # prefer to use a class load if available in app.models or train_pipeline.models
            class_candidates = [
                f"app.models.{name}_model",
                f"train_pipeline.models.{name}_model",
                f"models.{name}_model",
            ]
            cls = _try_import_model_class(class_candidates, "LogisticModel" if name == "logistic" else "EnsembleModel")
            if cls is not None and hasattr(cls, "load"):
                try:
                    inst = cls.load(p)
                    loaded[name] = inst
                    break
                except Exception:
                    pass
            # fallback: load artifact directly
            try:
                est = _load_sklearn_artifact(p)
                loaded[name] = est
                break
            except Exception:
                continue

    # mlp (PyTorch checkpoint)
    for candidate in MODEL_FILENAMES.get("mlp", []):
        p = Path(candidate)
        if not p.exists():
            continue
        # try to use MLPModel.load from known modules
        class_candidates = [
            "app.models.mlp_model",
            "train_pipeline.models.mlp_model",
            "models.mlp_model",
        ]
        cls = _try_import_model_class(class_candidates, "MLPModel")
        if cls is not None and hasattr(cls, "load"):
            try:
                inst = cls.load(p)
                loaded["mlp"] = inst
                break
            except Exception:
                pass
        # fallback: load checkpoint and attempt to build minimal wrapper if input_dim exists
        try:
            chk = _load_torch_checkpoint(p)
            state = chk.get("model_state_dict") or chk
            input_dim = chk.get("input_dim")
            if input_dim is None:
                # cannot reconstruct model without input_dim and class definition
                # store checkpoint so the caller may handle it
                loaded["mlp"] = {"checkpoint": chk, "path": p}
            else:
                # try to import app.models.mlp_model.TorchMLP or MLP class
                torch_cls = _try_import_model_class(class_candidates, "TorchMLP") or _try_import_model_class(class_candidates, "MLP")
                if torch_cls is not None:
                    inst = torch_cls(input_dim=input_dim) if callable(torch_cls) else None
                    if inst is not None:
                        inst_state = inst.state_dict()
                        # create model instance, load state dict, return wrapper with predict method
                        inst.load_state_dict(state)
                        inst.eval()
                        loaded["mlp"] = inst
                    else:
                        loaded["mlp"] = {"checkpoint": chk, "path": p}
                else:
                    loaded["mlp"] = {"checkpoint": chk, "path": p}
            break
        except Exception:
            continue

    return loaded


def transform_input(preprocessor, input_df: pd.DataFrame) -> Any:
    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=np.float32)


def predict_all(preprocessor, models: Dict[str, Any], input_df: pd.DataFrame) -> pd.DataFrame:
    import torch
    import torch.nn as nn
    import re

    X = transform_input(preprocessor, input_df)
    results = []

    class _TorchMLPForInfer(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(1)

    def _attempt_load_state(net: nn.Module, state: dict):
        try:
            net.load_state_dict(state)
            return True, None
        except RuntimeError as e:
            err_msg = str(e)
            # try common renaming strategies
            mapped_states = []

            # strategy 1: replace 'layers.' -> 'net.' and vice-versa
            mapped_states.append({k.replace("layers.", "net."): v for k, v in state.items()})
            mapped_states.append({k.replace("net.", "layers."): v for k, v in state.items()})

            # strategy 2: strip 'module.' prefix (common from DataParallel)
            mapped_states.append({re.sub(r"^module\.", "", k): v for k, v in state.items()})
            mapped_states.append({("net." + k) if not k.startswith("net.") else k: v for k, v in state.items()})

            # strategy 3: replace 'layers.' with 'layers.' inside 'module.' combos
            mapped_states.append({re.sub(r"^module\.layers\.", "net.", k): v for k, v in state.items()})
            mapped_states.append({re.sub(r"^module\.net\.", "net.", k): v for k, v in state.items()})

            for ms in mapped_states:
                try:
                    net.load_state_dict(ms)
                    return True, None
                except Exception:
                    continue
            return False, err_msg

    for name, model in models.items():
        preds = None
        probs = None

        if isinstance(model, dict) and "checkpoint" in model:
            chk = model["checkpoint"]
            state = chk.get("model_state_dict", chk)
            input_dim = chk.get("input_dim")
            if input_dim is None:
                input_dim = int(X.shape[1])

            try:
                net = _TorchMLPForInfer(input_dim)
                # load_state may fail due to key name differences; try and recover
                ok, err = _attempt_load_state(net, state if isinstance(state, dict) else {})
                if not ok:
                    # if state is nested (e.g., contains keys like 'model': {...}), try to find nested dict
                    if isinstance(state, dict):
                        # look for a nested dict candidate
                        nested = None
                        for v in state.values():
                            if isinstance(v, dict) and any("weight" in k or "bias" in k for k in v.keys()):
                                nested = v
                                break
                        if nested:
                            ok, err = _attempt_load_state(net, nested)
                    if not ok:
                        raise RuntimeError(f"Could not map checkpoint keys to model: {err}")
                net.eval()
                device = torch.device("cpu")
                net.to(device)
                with torch.no_grad():
                    tx = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
                    logits = net(tx)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
                pred_val = int(np.asarray(preds).ravel()[0])
                prob_val = float(np.asarray(probs).ravel()[0])
                results.append({"model": name, "prediction": "Good" if pred_val == 1 else "Bad", "probability": prob_val})
                continue
            except Exception as e:
                raise RuntimeError(f"Failed to run MLP checkpoint inference for '{name}': {e}")

        # sklearn-style or class wrapper
        try:
            out = None
            if hasattr(model, "predict"):
                out = model.predict(X)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]
                preds = (probs >= 0.5).astype(int)
            elif out is not None:
                if isinstance(out, tuple) and len(out) == 2:
                    preds, probs = out
                else:
                    preds = out
        except Exception:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]
                preds = (probs >= 0.5).astype(int)
            elif hasattr(model, "predict"):
                preds = model.predict(X)

        if preds is None:
            raise RuntimeError(f"Could not obtain predictions for model '{name}'. The loaded object type: {type(model)}")

        pred_val = int(np.asarray(preds).ravel()[0])
        prob_val = float(np.asarray(probs).ravel()[0]) if probs is not None else float(pred_val)
        results.append({"model": name, "prediction": "Good" if pred_val == 1 else "Bad", "probability": prob_val})

    return pd.DataFrame(results)
