from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass
class TrainConfig:
    nome: str = "Antonio"
    cognome: str = "Sirignano"

    data_dir: Path = Path("Allegati")
    artifacts_dir: Path = Path("Artifacts")

    # opzionale: best params ottenuti da Grid Search (già fatto)
    best_params_json: Path = Path("Artifacts/best_params.json")

    # opzionale: stopwords italiane in data/stopwords_it.txt (una per riga)
    stopwords_it_txt: Path = Path("data/stopwords_it.txt")

    test_size: float = 0.20
    random_state: int = 42


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _first_existing(self, candidates: List[Path]) -> Path:
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            "Nessun file trovato tra: " + ", ".join(str(c) for c in candidates)
        )

    def _train_file(self) -> Path:
        # supporta i nomi “strani” (xslx/xlsx)
        return self._first_existing(
            [
                self.cfg.data_dir / "Allegato 1 - data_classification.xlsx",
                self.cfg.data_dir / "Allegato 1 - data_classification.xslx",
                self.cfg.data_dir / "Allegato 1 - data_classification.xslx.xlsx",
            ]
        )

    def _load_stopwords_it(self) -> Optional[list[str]]:
        p = self.cfg.stopwords_it_txt
        if not p.exists():
            return None
        words = [
            w.strip()
            for w in p.read_text(encoding="utf-8").splitlines()
            if w.strip() and not w.strip().startswith("#")
        ]
        return words or None

    def _default_params(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        tfidf_params: Dict[str, Any] = {
            "lowercase": True,
            "strip_accents": "unicode", 
            "stop_words": self._load_stopwords_it(),
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
            "max_features": 200_000,
        }
        clf_params: Dict[str, Any] = {
            "C": 1.0,
            "class_weight": "balanced",
            "random_state": self.cfg.random_state,
        }
        return tfidf_params, clf_params

    def _load_best_params(self) -> Optional[Dict[str, Any]]:
        p = self.cfg.best_params_json
        if not p.exists():
            return None
        payload = json.loads(p.read_text(encoding="utf-8"))
        return payload.get("best_params", payload)

    def _fixed_params(self):
        tfidf_params, clf_params = self._default_params()
        best = self._load_best_params()
        if not best:
            return tfidf_params, clf_params

        for k, v in best.items():
            if k.startswith("tfidf__"):
                tfidf_params[k.replace("tfidf__", "")] = v
            elif k.startswith("clf__"):
                clf_params[k.replace("clf__", "")] = v

        # --- SANITIZE: JSON -> sklearn types ---
        ngr = tfidf_params.get("ngram_range")
        if isinstance(ngr, list):
            tfidf_params["ngram_range"] = tuple(ngr)

        # alcuni cast utili
        if "max_features" in tfidf_params and tfidf_params["max_features"] is not None:
            tfidf_params["max_features"] = int(tfidf_params["max_features"])
        if "min_df" in tfidf_params and tfidf_params["min_df"] is not None:
            tfidf_params["min_df"] = int(tfidf_params["min_df"])
        if "max_df" in tfidf_params and tfidf_params["max_df"] is not None:
            tfidf_params["max_df"] = float(tfidf_params["max_df"])

        # safety: stopwords italiane (evita english se finita nei best_params)
        if tfidf_params.get("stop_words") == "english":
            tfidf_params["stop_words"] = self._load_stopwords_it()

        return tfidf_params, clf_params


    def _build_model(self) -> Pipeline:
        tfidf_params, clf_params = self._fixed_params()
        return Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                ("clf", LinearSVC(**clf_params)),
            ]
        )

    def _load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        train_path = self._train_file()
        df = pd.read_excel(train_path, engine="openpyxl")

        if "Review" not in df.columns or "Promotore" not in df.columns:
            raise ValueError(f"Attese colonne: Review, Promotore. Trovate: {list(df.columns)}")

        X = df["Review"].astype(str).to_numpy()
        y = df["Promotore"].astype(int).to_numpy()
        return X, y

    def _save_roc_plot(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> Path:
        out = self.cfg.artifacts_dir / "roc_curve.png"
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        return out

    def run(self) -> None:
        # 1) split train/test
        X, y = self._load_training_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.cfg.test_size,
            stratify=y,
            random_state=self.cfg.random_state,
        )

        # 2) train su train, metriche su test
        model = self._build_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = float(f1_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # 3) ROC + AUC (LinearSVC -> decision_function)
        scores = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = float(auc(fpr, tpr))
        roc_path = self._save_roc_plot(fpr, tpr, roc_auc)

        # 4) confusion matrix (cm già calcolata)

        # 5) training su dataset completo
        model.fit(X, y)

        # salva modello come richiesto
        model_filename = f"{self.cfg.nome}_{self.cfg.cognome}_model.pickle"
        model_path = Path(model_filename)
        joblib.dump(model, model_path)

        # salva metriche (utile per consegna)
        tfidf_params, clf_params = self._fixed_params()
        metrics_payload = {
            "f1_test": f1,
            "auc_test": roc_auc,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "roc_curve_png": str(roc_path),
            "fixed_params_used": {"tfidf": tfidf_params, "clf": clf_params},
            "test_size": self.cfg.test_size,
            "random_state": self.cfg.random_state,
        }
        (self.cfg.artifacts_dir / "train_metrics.json").write_text(
            json.dumps(metrics_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n=== TRAIN COMPLETATO ===")
        print(f"F1 (test): {f1:.4f}")
        print(f"AUC (test): {roc_auc:.4f}")
        print("Confusion Matrix:", cm.tolist())
        print("ROC salvata in:", roc_path)
        print("Modello (FULL) salvato in:", model_path)
        print("Metriche salvate in:", self.cfg.artifacts_dir / "train_metrics.json")


def main() -> None:
    cfg = TrainConfig(
        nome="Antonio",   
        cognome="Sirignano" 
    )
    Trainer(cfg).run()


if __name__ == "__main__":
    main()
