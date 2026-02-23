# ==================== scripts/train_extended.py ====================
# Treina o modelo a partir do CSV gerado (train_corpus.csv) e avalia.
# Uso:
#   python3 scripts/train_extended.py --csv data/train_corpus.csv --model model_cls.joblib
#
# Dica: use junto com generate_training_corpus.py para alcançar 500–1000 exemplos.

import os, csv, argparse, json
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_CSV   = os.path.join(BASE_DIR, "data", "train_corpus.csv")
DEFAULT_MODEL = os.path.join(BASE_DIR, "model_cls.joblib")

def load_csv(path: str) -> Tuple[List[str], List[str]]:
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            text = (row.get("text") or "").strip()
            label = (row.get("label_id") or "").strip()
            # Se não tiver id (vindo só do JSON), vamos usar (label_code + label_title) como rótulo,
            # mas o /predict atual espera ID da tabela. Melhor treinar com itens do banco (use reseed antes).
            # Ainda assim, suportamos fallback:
            if not label:
                label = (row.get("label_code") or row.get("label_title") or "").strip()
            if text and label:
                X.append(text)
                y.append(label)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=DEFAULT_CSV)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--max_features", type=int, default=30000)
    ap.add_argument("--ngrams", type=str, default="1,2", help="ex.: 1,2 ou 1,3")
    ap.add_argument("--C", type=float, default=2.0, help="regularização (LogReg)")
    args = ap.parse_args()

    ngram_range = tuple(int(x) for x in args.ngrams.split(","))

    X, y = load_csv(args.csv)
    if not X:
        raise SystemExit(f"Nenhum dado em {args.csv}. Gere com generate_training_corpus.py primeiro.")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=args.test_size, random_state=42, stratify=y_enc)

    vec = TfidfVectorizer(min_df=1, max_features=args.max_features, ngram_range=ngram_range)
    Xtr_vec = vec.fit_transform(Xtr)
    Xte_vec = vec.transform(Xte)

    clf = LogisticRegression(max_iter=400, C=args.C, n_jobs=None, solver="lbfgs", multi_class="auto")
    clf.fit(Xtr_vec, ytr)

    # Avaliação
    ypred = clf.predict(Xte_vec)
    acc = accuracy_score(yte, ypred)
    try:
        prob = clf.predict_proba(Xte_vec)
        top3 = top_k_accuracy_score(yte, prob, k=3)
        top5 = top_k_accuracy_score(yte, prob, k=5) if prob.shape[1] >= 5 else None
    except Exception:
        prob, top3, top5 = None, None, None

    print("=== Resultados ===")
    print("Amostras total:", len(X))
    print("Acurácia (top1):", f"{acc*100:.2f}%")
    if top3 is not None:
        print("Top-3 accuracy:", f"{top3*100:.2f}%")
    if top5 is not None:
        print("Top-5 accuracy:", f"{top5*100:.2f}%")
    print("\nRelatório por classe (top1):")
    print(classification_report(yte, ypred, zero_division=0))

    # Salvar modelo (compatível com /predict atual)
    bundle = {"vec": vec, "cls": clf, "label_encoder": le}
    joblib.dump(bundle, args.model)
    print(f"[OK] Modelo salvo em: {args.model}")

if __name__ == "__main__":
    main()
# ==================== /scripts/train_extended.py ====================