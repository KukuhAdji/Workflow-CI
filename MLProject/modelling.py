"""
modelling.py

Training dengan hyperparameter tuning dan manual MLflow logging.
Versi ini sudah disesuaikan agar aman dijalankan melalui MLflow Project
di GitHub Actions.

Contoh local:
python modelling.py --data_dir namadataset_preprocessing --target_col target

Contoh via MLflow Project:
mlflow run MLProject --env-manager=local -P data_dir=namadataset_preprocessing -P target_col=target
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def load_train_test(data_dir: str | Path, target_col: str):
    data_dir = Path(data_dir)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"File train.csv tidak ditemukan di: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"File test.csv tidak ditemukan di: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        target_col = metadata.get("target_col", metadata.get("target_column", target_col))

    if target_col not in train_df.columns:
        raise ValueError(
            f"Kolom target '{target_col}' tidak ditemukan. "
            f"Kolom tersedia: {list(train_df.columns)}"
        )

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test, target_col


def safe_roc_auc(model, X_test, y_test):
    try:
        if not hasattr(model, "predict_proba"):
            return None

        y_proba = model.predict_proba(X_test)
        classes = np.unique(y_test)

        if len(classes) == 2:
            return roc_auc_score(y_test, y_proba[:, 1])

        return roc_auc_score(
            y_test,
            y_proba,
            multi_class="ovr",
            average="weighted",
        )

    except Exception as exc:
        print("ROC AUC dilewati:", exc)
        return None


def save_artifacts(model, X_test, y_test, y_pred, artifact_dir: Path):
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(artifact_dir / "confusion_matrix.png")
    plt.close(fig)

    report = classification_report(y_test, y_pred, zero_division=0)
    report_path = artifact_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        fi.to_csv(artifact_dir / "feature_importance.csv", index=False)

    sample_pred = X_test.head(20).copy()
    sample_pred["actual"] = list(y_test.head(20))
    sample_pred["prediction"] = list(y_pred[:20])
    sample_pred.to_csv(artifact_dir / "sample_predictions.csv", index=False)

    return artifact_dir


def setup_mlflow(experiment_name: str):
    """
    Jika dijalankan oleh MLflow Project di GitHub Actions,
    jangan set tracking URI manual.

    Jika dijalankan langsung secara lokal,
    gunakan folder mlruns lokal.
    """
    is_mlflow_project_run = os.environ.get("MLFLOW_RUN_ID") is not None

    if not is_mlflow_project_run:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)


def start_mlflow_run(run_name: str):
    """
    Jika dijalankan melalui MLflow Project, gunakan run yang sudah dibuat
    oleh MLflow Project.

    Jika dijalankan langsung, buat run baru.
    """
    existing_run_id = os.environ.get("MLFLOW_RUN_ID")

    if existing_run_id:
        return mlflow.start_run(run_id=existing_run_id)

    return mlflow.start_run(run_name=run_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="namadataset_preprocessing")
    parser.add_argument("--target_col", default="target")
    parser.add_argument("--experiment_name", default="MSML-Skilled-ManualLogging")
    args = parser.parse_args()

    setup_mlflow(args.experiment_name)

    X_train, X_test, y_train, y_test, target_col = load_train_test(
        args.data_dir,
        args.target_col,
    )

    base_model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    with start_mlflow_run("RandomForest-Tuning-ManualLogging"):
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        roc_auc = safe_roc_auc(best_model, X_test, y_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "recall_weighted": recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "f1_weighted": f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "best_cv_score": search.best_score_,
        }

        if roc_auc is not None:
            metrics["roc_auc"] = roc_auc

        mlflow.log_params(search.best_params_)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        mlflow.log_metrics(metrics)

        artifact_dir = save_artifacts(
            best_model,
            X_test,
            y_test,
            y_pred,
            Path("artifacts"),
        )

        mlflow.log_artifacts(
            str(artifact_dir),
            artifact_path="evaluation_artifacts",
        )

        signature = infer_signature(X_train, best_model.predict(X_train))
        input_example = X_train.head(5)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        run = mlflow.active_run()

        print("Training tuning selesai.")
        print("Run ID:", run.info.run_id)
        print("Best params:", search.best_params_)
        print("Metrics:", metrics)
        print("Artifact tersimpan di MLflow.")
        print("Model tersimpan di artifact path: model")


if __name__ == "__main__":
    main()
