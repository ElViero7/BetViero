from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base import LoLPipelineError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for First Blood Map 1 using a prepared feature dataset."
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("models/modelo_firstblood.pkl"),
        help="Path to the trained model artifact.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="CSV or Parquet file with matchup feature rows ready for scoring.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/predictions/firstblood_predictions.csv"),
        help="Where to save the scored predictions.",
    )
    return parser.parse_args()


def load_frame(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise LoLPipelineError(f"Inference input file not found: {input_path}")
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path, low_memory=False)
    if input_path.suffix.lower() in {".parquet", ".pq"}:
        return duckdb.read_parquet(str(input_path)).df()
    raise LoLPipelineError(f"Unsupported inference input format: {input_path.suffix}")


def main() -> None:
    args = parse_args()

    with args.artifact_path.open("rb") as file_obj:
        artifact = pickle.load(file_obj)

    frame = load_frame(args.input_path)
    feature_columns = artifact["feature_columns"]
    missing_columns = [column for column in feature_columns if column not in frame.columns]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise LoLPipelineError(f"Inference input is missing required feature columns: {missing_list}")

    probabilities = artifact["model"].predict_proba(frame[feature_columns])[:, 1]
    output = frame.copy()
    output["model_name"] = artifact.get("metrics", {}).get("selected_model_name", "unknown")
    output["blue_firstblood_prob"] = probabilities
    output["red_firstblood_prob"] = 1.0 - output["blue_firstblood_prob"]
    output["predicted_side"] = output["blue_firstblood_prob"].apply(
        lambda value: "blue" if value >= 0.5 else "red"
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)

    summary = {
        "rows_scored": int(len(output)),
        "model_name": output["model_name"].iloc[0] if not output.empty else None,
        "artifact_path": str(args.artifact_path),
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
