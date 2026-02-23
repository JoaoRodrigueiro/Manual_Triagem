# ==================== scripts/generate_training_corpus.py ====================
# Gera um corpus de treinamento a partir do JSON e/ou do banco, com variações sintéticas.
# Uso (no diretório do projeto):
#   python3 scripts/generate_training_corpus.py --per_error 15 --out data/train_corpus.csv
#
# Dica: 50 erros * 20 variações = 1000 amostras.

import os, json, csv, argparse, random, sqlite3, re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_JSON = os.path.join(BASE_DIR, "data", "erros_base.json")
DB_PATH   = os.path.join(BASE_DIR, "database.db")
OUT_DEFAULT = os.path.join(BASE_DIR, "data", "train_corpus.csv")

random.seed(42)

# Sinônimos simples (PT-BR) para variações de texto
SYN = {
    "impressora": ["impressora", "equipamento", "máquina"],
    "erro": ["erro", "falha", "mensagem de erro"],
    "travar": ["travar", "congelar", "parar"],
    "atolamento": ["atolamento", "papel preso", "travamento de papel"],
    "digitalização": ["digitalização", "scanner", "escaneamento"],
    "enviar": ["enviar", "remeter", "mandar"],
    "e-mail": ["e-mail", "email", "correio eletrônico"],
    "fila": ["fila", "spool", "pendente"],
    "reiniciar": ["reiniciar", "desligar e ligar", "resetar"],
    "rede": ["rede", "conexão", "comunicação"],
}

def syn(s: str) -> str:
    # troca uma palavra por um sinônimo se existir
    words = s.split()
    ixs = list(range(len(words)))
    random.shuffle(ixs)
    for i in ixs:
        w = re.sub(r"[^\wáéíóúàèìòùâêîôûãõçÁÉÍÓÚÂÊÎÔÛÀÈÌÒÙÃÕÇ-]", "", words[i].lower())
        if w in SYN:
            words[i] = s.replace(w, random.choice(SYN[w]))
            break
    return " ".join(words)

def read_json_items():
    if not os.path.exists(DATA_JSON):
        return []
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data

def read_db_items():
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, codigo, titulo, descricao, termos FROM erros").fetchall()
    items = []
    for r in rows:
        terms = []
        if r["termos"]:
            terms = [t.strip() for t in r["termos"].split(",") if t.strip()]
        items.append({
            "id_db": r["id"],
            "codigo": r["codigo"],
            "titulo": r["titulo"],
            "descricao": r["descricao"],
            "termos": terms
        })
    conn.close()
    return items

def combos_from_item(item, per_error: int):
    """
    Cria variações de texto para um item (erro) combinando título, descrição, termos e templates.
    """
    label_code = (item.get("codigo") or "").strip()
    label_title = (item.get("titulo") or "").strip()
    label_id = str(item.get("id_db") or "")  # se vier do DB já tem ID; se vier só do JSON fica vazio (vamos mapear)
    desc = (item.get("descricao") or "").strip()
    termos = item.get("termos") or []
    termos_txt = ", ".join(t for t in termos if t)

    base_sentences = [
        f"{random.choice(SYN['erro'])} {label_code} ao usar a {random.choice(SYN['impressora'])}.",
        f"Cliente relata {random.choice(SYN['erro'])} {label_code} na {random.choice(SYN['impressora'])}.",
        f"Problema: {label_title}.",
        f"Ocorre durante impressão: {label_title}.",
        f"Descrição: {desc}" if desc else f"Descrição não informada. {label_title}.",
        f"Palavras-chave: {termos_txt}" if termos_txt else ""
    ]

    # Templates de narrativa
    templates = [
        "Ao tentar imprimir um documento PDF grande, a {imp} apresenta {err} {code}. {extra}",
        "Usuário informa que a {imp} mostra {err} {code} e não conclui o trabalho. {extra}",
        "Ocorrência de {err} {code} intermitente. {extra}",
        "Durante a {acao}, aparece {err} {code}. {extra}",
        "A {imp} está lenta e depois ocorre {err} {code}. {extra}"
    ]
    acao_opts = ["impressão", "cópia", "digitalização", "varredura de pastas", "envio por e-mail"]

    result = []
    for i in range(per_error):
        extra = random.choice(base_sentences)
        text = random.choice(templates).format(
            imp=random.choice(SYN["impressora"]),
            err=random.choice(SYN["erro"]),
            code=label_code or "",
            acao=random.choice(acao_opts),
            extra=syn(extra)
        ).strip()
        # Fallback: se ficou muito curto, adiciona título/desc
        if len(text) < 40:
            text += " " + (label_title or "")
            if desc: text += " " + desc
        result.append({
            "text": text,
            "label_id": label_id,
            "label_code": label_code,
            "label_title": label_title
        })
    return result

def map_json_to_db_ids(json_items, db_items):
    """
    Tenta mapear itens do JSON para IDs do DB usando (codigo, titulo) como chaves.
    Retorna lista de dicts com id_db preenchido quando possível.
    """
    key_db = {}
    for d in db_items:
        key = ((d.get("codigo") or "").strip().lower(), (d.get("titulo") or "").strip().lower())
        key_db[key] = d["id_db"]
    mapped = []
    for j in json_items:
        key = ((j.get("codigo") or "").strip().lower(), (j.get("titulo") or "").strip().lower())
        id_db = key_db.get(key)
        mapped.append({
            "id_db": id_db,
            "codigo": j.get("codigo"),
            "titulo": j.get("titulo"),
            "descricao": j.get("descricao"),
            "termos": j.get("termos")
        })
    return mapped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_error", type=int, default=10, help="quantidade de variações por erro")
    ap.add_argument("--out", type=str, default=OUT_DEFAULT, help="arquivo CSV de saída")
    ap.add_argument("--use_db_only", action="store_true", help="usar apenas o banco (ignora JSON)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    db_items = read_db_items()
    if args.use_db_only:
        items = db_items
    else:
        json_items = read_json_items()
        items = map_json_to_db_ids(json_items, db_items)

    # Filtra itens válidos (tem pelo menos título)
    items = [it for it in items if (it.get("titulo") or "").strip()]

    total = 0
    rows = []
    for it in items:
        var = combos_from_item(it, args.per_error)
        rows.extend(var)
        total += len(var)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label_id", "label_code", "label_title", "generated_at"])
        for r in rows:
            w.writerow([r["text"], r["label_id"], r["label_code"], r["label_title"], datetime.utcnow().isoformat()])

    print(f"[OK] Corpus salvo em: {args.out}  | exemplos: {total}  | itens de erro: {len(items)}")
    print("Exemplo:", rows[0]["text"] if rows else "(sem linhas)")

if __name__ == "__main__":
    main()
# ==================== /scripts/generate_training_corpus.py ====================