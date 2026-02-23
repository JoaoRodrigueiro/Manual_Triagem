# -*- coding: utf-8 -*-
import json, re, os
from copy import deepcopy
from unidecode import unidecode

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "erros_base.json")
BACKUP_PATH = os.path.join(BASE_DIR, "data", "erros_base.backup.json")

SPACE_RE = re.compile(r"[ \t]+")
MULTILINE_EMPTY_RE = re.compile(r"\n{3,}")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;:\.\!\?])")

def end_with_period(s: str) -> str:
    s = s.strip()
    if not s: return s
    if s[-1] in ".!?": return s
    return s + "."

def smart_capitalize(s: str) -> str:
    s = s.strip()
    if not s: return s
    s = s[0].upper() + s[1:]
    return s

def normalize_list(items):
    if not items: return []
    seen, out = set(), []
    for it in items:
        if not it: continue
        s = SPACE_RE.sub(" ", str(it)).strip(" -•\t")
        if not s: continue
        s = end_with_period(smart_capitalize(s))
        if s.lower() not in seen:
            out.append(s); seen.add(s.lower())
    return out

def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit("data/erros_base.json não encontrado.")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(BACKUP_PATH, "w", encoding="utf-8") as b:
        json.dump(data, b, ensure_ascii=False, indent=2)
    formatted = []
    for it in data:
        it = dict(it)
        if it.get("titulo"):
            it["titulo"] = smart_capitalize(it["titulo"])
        if isinstance(it.get("causas"), list):
            it["causas"] = normalize_list(it["causas"])
        if isinstance(it.get("solucoes"), list):
            it["solucoes"] = normalize_list(it["solucoes"])
        if isinstance(it.get("perguntas"), list):
            it["perguntas"] = normalize_list(it["perguntas"])
        formatted.append(it)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)
    print("Formatado! Backup em", BACKUP_PATH)

if __name__ == "__main__":
    main()
