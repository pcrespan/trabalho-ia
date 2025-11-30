import os
import time
import json
import sys
from pathlib import Path
import pandas as pd

_this_file = Path(__file__).resolve()
project_root = _this_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.train import train_and_persist_models, DEFAULT_TRAINING_CSV

STATE_PATH = os.environ.get("RETRAIN_STATE_PATH", ".retrain_state.json")
SLEEP_SECONDS = int(os.environ.get("RETRAIN_INTERVAL_SECONDS", "15"))
CSV_PATH = os.environ.get("TRAINING_CSV_PATH", DEFAULT_TRAINING_CSV)

def _load_state() -> dict:
    p = Path(STATE_PATH)
    if not p.exists():
        return {"last_count": 0}
    with p.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {"last_count": 0}

def _save_state(state: dict):
    p = Path(STATE_PATH)
    with p.open("w", encoding="utf-8") as f:
        json.dump(state, f)

def _get_csv_len(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    try:
        df = pd.read_csv(p)
        return len(df)
    except Exception:
        return 0

def main():
    state = _load_state()
    last_count = int(state.get("last_count", 1000))
    while True:
        try:
            current_count = _get_csv_len(CSV_PATH)
            if current_count > last_count:
                print(f"[watcher] Detected new rows: previous={last_count} now={current_count}. Starting retrain.")
                try:
                    out = train_and_persist_models(CSV_PATH)
                    print(f"[watcher] Retrain finished. Models saved: {out}")
                    last_count = current_count
                    state["last_count"] = last_count
                    _save_state(state)
                except Exception as e:
                    print(f"[watcher] Retrain failed: {e}")
            else:
                print(f"[watcher] No new rows. count={current_count}")
        except Exception as e:
            print(f"[watcher] Error in loop: {e}")
        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
