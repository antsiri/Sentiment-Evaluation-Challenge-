from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd


@dataclass
class EvalConfig:
    nome: str = "Antonio"
    cognome: str = "Sirignano"

    data_dir: Path = Path("Allegati")


class Evaluator:
    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg

    def _first_existing(self, candidates: List[Path]) -> Path:
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            "Nessun file trovato tra: " + ", ".join(str(c) for c in candidates)
        )

    def _eval_file(self) -> Path:
        return self._first_existing(
            [
                self.cfg.data_dir / "Allegato 2 - data_evaluation.xlsx.xlsx",
                self.cfg.data_dir / "Allegato 2 - data_evaluation.xslx",
                self.cfg.data_dir / "Allegato 2 - data_evaluation.xslx.xlsx",
            ]
        )

    def _model_file(self) -> Path:
        return Path(f"{self.cfg.nome}_{self.cfg.cognome}_model.pickle")

    def run(self) -> None:
        model_path = self._model_file()
        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato: {model_path}. Esegui prima main_train.py")

        model = joblib.load(model_path)

        eval_path = self._eval_file()
        df = pd.read_excel(eval_path, engine="openpyxl")

        if "Review" not in df.columns:
            raise ValueError(f"Nel file di evaluation manca la colonna 'Review'. Colonne: {list(df.columns)}")

        reviews = df["Review"].astype(str).to_numpy()
        preds = model.predict(reviews).astype(int)

        df["promotore_pred"] = preds

        out_name = f"output_pred_{self.cfg.nome}__{self.cfg.cognome}.xlsx"
        out_path = Path(out_name)
        df.to_excel(out_path, index=False, engine="openpyxl")

        print("\n=== EVAL COMPLETATO ===")
        print("Input evaluation:", eval_path)
        print("Modello usato:", model_path)
        print("Output salvato in:", out_path)


def main() -> None:
    cfg = EvalConfig(
        nome="Antonio",   
        cognome="Sirignano" 
    )
    Evaluator(cfg).run()


if __name__ == "__main__":
    main()
