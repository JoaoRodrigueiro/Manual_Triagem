# ==================== main.py ====================
from fastapi import FastAPI, Request, Query, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Header, HTTPException, Depends
from fastapi import APIRouter, Depends

import os, json, io, csv, sqlite3
from typing import List, Dict, Any, Optional

# ML (opcional, mas já incluído)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

# ---------------- Paths / App ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "erros_base.json")
DB_PATH   = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "model_cls.joblib")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

app = FastAPI(title="Manual Inteligente – Triagem")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ---------------- DB helpers ----------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

NEW_COLUMNS = [
    ("marca", "TEXT"),
    ("modelo", "TEXT"),
    ("acao_recomendada", "TEXT"),
    ("perguntas_cliente", "TEXT"),  # JSON list
    ("perguntas_n1", "TEXT"),       # JSON list
    ("termos", "TEXT"),             # CSV/str
]

def migrate_db():
    with get_conn() as conn:
        # tabela base
        conn.execute("""
        CREATE TABLE IF NOT EXISTS erros (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            codigo TEXT,
            titulo TEXT NOT NULL,
            marca TEXT,
            modelo TEXT,
            descricao TEXT,
            causas TEXT,           -- JSON
            solucoes TEXT,         -- JSON
            perguntas_cliente TEXT,-- JSON
            perguntas_n1 TEXT,     -- JSON
            nivel TEXT,
            procedimento_link TEXT,
            criticidade INTEGER,
            termos TEXT,
            acao_recomendada TEXT
        );
        """)
        # adiciona colunas que faltarem
        cols = {r[1] for r in conn.execute("PRAGMA table_info(erros)").fetchall()}
        for col, ctype in NEW_COLUMNS:
            if col not in cols:
                conn.execute(f"ALTER TABLE erros ADD COLUMN {col} {ctype}")
        # feedback
        conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            erro_id INTEGER,
            resolvido INTEGER,
            comentario TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)
        # log de buscas
        conn.execute("""
        CREATE TABLE IF NOT EXISTS queries_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            termo TEXT,
            marca TEXT,
            nivel TEXT,
            criticidade_min INTEGER,
            results_count INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.commit()

# ---------------- Humanização de perguntas (Cliente) ----------------
def has_term(item: dict, *keys) -> bool:
    texto = " ".join([
        " ".join(item.get("termos") or []),
        item.get("titulo") or "",
        item.get("descricao") or "",
        item.get("codigo") or ""
    ]).lower()
    return any(k.lower() in texto for k in keys)

def pick_theme(item: dict) -> str:
    cod = (item.get("codigo") or "").upper()
    if cod.startswith("SC5"): return "fusor_ricoh"
    if cod.startswith("900."): return "fw_lexmark"
    if cod.startswith("49."): return "fw_hp"
    if cod.startswith("JAM-"): return "atolamento"
    if cod.startswith("SMTP-"): return "smtp"
    if cod.startswith("ADF-") or cod.startswith("SCAN-"): return "scanner"
    if cod.startswith("PQ-"): return "qualidade"
    if cod.startswith("NET-"): return "rede"
    if cod in ("KIT-RESET","DRUM-RESET","WASTE-101"): return "suprimentos"
    if cod.startswith("USB-"): return "usb"
    if has_term(item, "smtp","scan to email","e-mail"): return "smtp"
    if has_term(item, "jam","atolamento"): return "atolamento"
    if has_term(item, "scanner","adf","linha","skew"): return "scanner"
    if has_term(item, "qualidade","mancha","faixa","pálida","desbotada"): return "qualidade"
    if has_term(item, "rede","offline","ip","servidor","spooler"): return "rede"
    if has_term(item, "firmware","travamento"): return "firmware_geral"
    return "geral"

TEMPLATES_CLIENTE = {
    "fusor_ricoh": [
        "O aviso aparece logo que liga a impressora?",
        "A impressora foi movida ou tomou algum tranco recentemente?",
        "O local teve queda de energia hoje?",
        "Consegue desligar, esperar 30 segundos e ligar novamente?"
    ],
    "fw_lexmark": [
        "O erro aparece quando imprime de um programa específico (Excel, navegador)?",
        "Se imprimir uma página de teste simples, o erro aparece?",
        "Tem documentos parados na fila de impressão do computador/servidor?",
        "A impressora ficou muito tempo sem atualizar?"
    ],
    "fw_hp": [
        "O erro aparece com qualquer arquivo ou só com alguns (PDF grande, por exemplo)?",
        "Você já desligou e ligou a impressora e o computador?",
        "Tem muitos documentos parados na fila de impressão?",
        "O cabo de rede/USB está bem encaixado?"
    ],
    "smtp": [
        "O envio por e-mail já funcionou antes neste equipamento?",
        "Você consegue enviar e receber e-mails normalmente no computador/celular?",
        "Sabe qual serviço de e-mail usa (Gmail, Outlook, corporativo)?",
        "Se a conta tem verificação em duas etapas, existe uma senha especial para aplicativos?",
        "Pode tirar uma foto da tela da impressora com a mensagem de erro?"
    ],
    "qualidade": [
        "O problema aparece em todas as páginas ou só às vezes?",
        "A falha sempre aparece no mesmo lugar da folha?",
        "Acontece usando papel de bandejas diferentes?",
        "Já trocou o papel por um pacote novo (papel seco e liso)?"
    ],
    "atolamento": [
        "O papel prende mais na entrada, no meio ou na saída da impressora?",
        "Você vê algum pedacinho de papel preso? Consegue remover com cuidado?",
        "Está usando papel normal e em bom estado (não amassado/úmido)?",
        "Quando trava, aparece algum número/código na tela? Pode tirar uma foto?"
    ],
    "scanner": [
        "As linhas aparecem usando o alimentador (passa folhas) ou também no vidro grande?",
        "Se colocar a folha no vidro grande, o problema continua?",
        "As guias do alimentador (as abinhas) estão ajustadas ao papel?",
        "Consegue limpar o vidro (inclusive a faixa estreita) com pano macio?"
    ],
    "rede": [
        "Outros computadores usam a internet normalmente aí no local?",
        "O cabo da impressora está bem encaixado na impressora e no roteador?",
        "Na tela da impressora aparece um número de IP? (pode tirar uma foto?)",
        "Alguém mexeu no roteador ou na rede recentemente?",
        "Consegue reiniciar a impressora e o roteador?"
    ],
    "usb": [
        "O cabo USB está bem encaixado?",
        "Pode testar outra porta USB do computador?",
        "Tem como testar com outro cabo USB (se possível, mais curto)?",
        "O computador reconhece a impressora quando conecta o cabo?"
    ],
    "suprimentos": [
        "A peça (kit/unidade de imagem/toner residual) foi trocada fisicamente?",
        "Depois da troca, apareceu alguma mensagem diferente?",
        "Qual é o modelo exato da impressora (etiqueta frontal ou tela)?",
        "Pode tirar uma foto da mensagem atual na tela?"
    ],
    "firmware_geral": [
        "O erro aparece sempre ao imprimir ou só em alguns arquivos?",
        "Você já desligou/ligou a impressora e limpou a fila de impressão?",
        "A impressora ficou muito tempo sem atualizar?"
    ],
    "geral": [
        "Quando o problema começou? Acontece sempre ou às vezes?",
        "Aparece algum número/código na tela? Pode tirar uma foto?",
        "Já tentou desligar e ligar a impressora?",
        "Isso acontece com todos os documentos ou só com alguns?"
    ]
}

def humanize_cliente(item: dict) -> List[str]:
    theme = pick_theme(item)
    return TEMPLATES_CLIENTE.get(theme, TEMPLATES_CLIENTE["geral"])

# ---------------- Seed (defaults seguros) ----------------
def _normalize_text_for_compare(s: str) -> str:
    """Normaliza texto para comparar e tirar duplicados."""
    return (
        s.lower()
         .replace(".", "")
         .replace(",", "")
         .replace("  ", " ")
         .strip()
    )

def seed_if_empty():

    def is_at_direto(e: dict) -> bool:
        codigo = (e.get("codigo") or "").upper()
        termos = " ".join(e.get("termos") or []).lower()
        titulo = (e.get("titulo") or "").lower()
        descricao = (e.get("descricao") or "").lower()

        texto = " ".join([titulo, descricao, termos])

        if any(k in codigo for k in ["FUSOR", "DRUM", "WASTE", "UNIT"]):
            return True
        
        if any (k in texto for k in [
            "troca",
            "substituir",
            "quebrado",
            "defeito físico",
            "unidade de imagem",
            "sensor",
        ]):
            return True
        
        return False

    frase_final = "Caso não seja possível resolver remotamente, enviar para AT."
    frase_final_key = _normalize_text_for_compare(frase_final)

    if not os.path.exists(DATA_PATH):
        print("SEED DEBUG - arquivo JSON não encontrado")
        return

    with get_conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM erros").fetchone()[0]
        if n != 0:
            print("SEED DEBUG - banco já populado, pulando seed")
            return

        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("SEED DEBUG - total de erros no JSON:", len(data))

        for e in data:
            codigo    = (e.get("codigo") or "").strip()
            titulo    = (e.get("titulo") or "").strip()
            descricao = (e.get("descricao") or "").strip()

            # termos precisam existir antes da inferência
            termos = e.get("termos") or []

            # marca explícita ou inferida
            marca = (e.get("marca") or "").strip()
            if not marca:
                marca_inf = infer_modelo(titulo, termos, descricao)
                if marca_inf:
                    marca = marca_inf.split()[0]

            # modelo explícito ou inferido
            modelo = (e.get("modelo_explicit") or e.get("modelo") or "").strip()
            if not modelo:
                modelo_inf = infer_modelo(titulo, termos, descricao)
                if modelo_inf:
                    partes = modelo_inf.split()
                    if len(partes) > 1:
                        modelo = partes[-1]
                    else:
                        modelo = ""

            # causas e soluções
            causas = e.get("causas") or []
            solucoes_raw = e.get("solucoes") or []

            # limpeza das soluções
            solucoes_limpa = []
            vistos = set()

            for s in solucoes_raw:
                s2 = (s or "").strip()
                if not s2:
                    continue

                key = _normalize_text_for_compare(s2)
                if key == frase_final_key:
                    continue

                if key not in vistos:
                    vistos.add(key)
                    solucoes_limpa.append(s2)

            if not is_at_direto(e):
                solucoes_limpa.append(frase_final)
            solucoes = solucoes_limpa

            if is_at_direto(e):
                acao_recomendada = "at_direto"
            else:
                acao_recomendada = "remota"

            perguntas_n1 = e.get("perguntas") or e.get("perguntas_n1") or []
            perguntas_cliente = e.get("perguntas_cliente") or humanize_cliente(e)
            nivel    = (e.get("nivel") or "atendente").strip().lower()
            proc     = (e.get("procedimento_link") or "").strip()
            crit     = int(e.get("criticidade") or 3)
            termos_s = ",".join([str(t).strip().lower() for t in termos if str(t).strip()])
            acao     = (e.get("acao_recomendada") or acao_recomendada).strip()

            conn.execute("""
                INSERT INTO erros
                (codigo,titulo,marca,modelo,descricao,causas,solucoes,
                 perguntas_cliente,perguntas_n1,nivel,procedimento_link,
                 criticidade,termos,acao_recomendada)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                codigo, titulo, marca, modelo, descricao,
                json.dumps(causas, ensure_ascii=False),
                json.dumps(solucoes, ensure_ascii=False),
                json.dumps(perguntas_cliente, ensure_ascii=False),
                json.dumps(perguntas_n1, ensure_ascii=False),
                nivel, proc, crit, termos_s, acao
            ))

        conn.commit()
        print("SEED DEBUG - seed finalizado com sucesso")


# ---------------- Modelo/Marca inferência simples ----------------
BRANDS = ["Ricoh", "Lexmark", "Kyocera", "HP", "Epson", "Konica Minolta", "Konica", "Minolta"]
import re
MODEL_REGEX = re.compile(
    r"""
    \b(
      (?:SP|MP|IM|CX|CS|MX|MS|X|C|T|E|W|P|M)
      [ -]?\d{3,4}[A-Z]{0,3}
      |
      [A-Z]{2,4}[ -]?\d{3,4}[A-Z]{0,3}
    )\b
    """, re.VERBOSE | re.IGNORECASE
)

def infer_modelo(titulo: Optional[str], termos: Optional[List[str]], descricao: Optional[str]) -> Optional[str]:
    chunks = []
    if titulo: chunks.append(titulo)
    if descricao: chunks.append(descricao)
    if termos: chunks.append(" ".join(termos))
    big = " ".join(chunks)
    brand = next((b for b in BRANDS if b.lower() in big.lower()), None)
    m = MODEL_REGEX.search(big)
    model = m.group(1).upper().replace(" ", "") if m else None
    if brand and model: return f"{brand} {model}"
    return model or brand

# ---------------- Busca + Log ----------------
def buscar(
    termo: str | None,
    limit: int = 20,
    filtro_marca: str | None = None,
    filtro_nivel: str | None = None,
    filtro_modelo: str | None = None,
    min_criticidade: int | None = None
) -> List[Dict[str, Any]]:

    where = []
    params = []

    # ---------------- TEXTO ----------------
    score_params = ["%", "%", "%"]  # default neutro

    if termo and len(termo.strip()) >= 3:
        like = f"%{termo}%"
        where.append("(titulo LIKE ? OR termos LIKE ? OR descricao LIKE ? OR codigo LIKE ?)")
        params.extend([like, like, like, like])
        score_params = [like, like, like]

    if not where:
        where.append("1=1")

    # ---------------- MARCA ----------------
    if filtro_marca:
        where.append("UPPER(TRIM(marca)) = UPPER(?)")
        params.append(filtro_marca)

    # ---------------- MODELO ----------------
    if filtro_modelo:
        where.append("""
            (
                modelo IS NULL
                OR TRIM(modelo) = ''
                OR REPLACE(REPLACE(UPPER(modelo), ' ', ''), '-', '')
                   LIKE REPLACE(REPLACE(UPPER(?), ' ', ''), '-', '')
            )
        """)
        params.append(f"%{filtro_modelo}%")

    # ---------------- NÍVEL ----------------
    if filtro_nivel:
        where.append("COALESCE(nivel, '') = ?")
        params.append(filtro_nivel)

    # ---------------- CRITICIDADE ----------------
    if min_criticidade is not None:
        where.append("COALESCE(criticidade, 0) >= ?")
        params.append(int(min_criticidade))

    where_sql = " AND ".join(where)

    query = f"""
        SELECT id, codigo, titulo, termos, descricao, causas, solucoes,
               perguntas_cliente, perguntas_n1, nivel,
               procedimento_link, criticidade, marca, modelo,
               (CASE WHEN titulo LIKE ? THEN 2 ELSE 0 END
                + CASE WHEN termos LIKE ? THEN 2 ELSE 0 END
                + CASE WHEN descricao LIKE ? THEN 1 ELSE 0 END) AS score
        FROM erros
        WHERE {where_sql}
        ORDER BY score DESC, criticidade DESC
        LIMIT ?
    """

    with get_conn() as conn:
        rows = conn.execute(
            query,
            score_params + params + [limit]
        ).fetchall()

    results = []
    for r in rows:
        termos_list = (r["termos"] or "").split(",") if r["termos"] else []
        modelo_inferido = r["modelo"] or infer_modelo(r["titulo"], termos_list, r["descricao"])
        acao = "verificar"
        destino = "N1"

        titulo_lower = (r["titulo"] or "").lower()
        descricao_lower = (r["descricao"] or "").lower()

        if "reset" in titulo_lower or "reset" in descricao_lower:
            acao = "executar"

        if "firmware" in titulo_lower:
            acao = "atualizar"

        if "hardware" in titulo_lower:
            destino = "AT"

        if "não seja possível resolver" in descricao_lower:
            destino = "AT"

        results.append({
            "id": r["id"],
            "codigo": r["codigo"],
            "titulo": r["titulo"],
            "termos": termos_list,
            "descricao": r["descricao"],
            "causas": json.loads(r["causas"]) if r["causas"] else [],
            "solucoes": json.loads(r["solucoes"]) if r["solucoes"] else [],
            "perguntas_cliente": json.loads(r["perguntas_cliente"]) if r["perguntas_cliente"] else [],
            "perguntas_n1": json.loads(r["perguntas_n1"]) if r["perguntas_n1"] else [],
            "nivel": r["nivel"],
            "procedimento_link": r["procedimento_link"],
            "criticidade": r["criticidade"],
            "marca": r["marca"],
            "modelo": modelo_inferido,
            "score": r["score"],
            "acao": acao,
            "destino": destino,
        })

    return results

def log_busca(termo: str, marca: str | None, nivel: str | None, criticidade: int | None, results_count: int):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO queries_log (termo, marca, nivel, criticidade_min, results_count)
            VALUES (?, ?, ?, ?, ?)
        """, (
            termo or "",
            marca or "",
            nivel or "",
            int(criticidade) if criticidade is not None else None,
            int(results_count),
        ))
        conn.commit()

# ---------------- Startup ----------------
@app.on_event("startup")
def on_startup():
    migrate_db()
    seed_if_empty()

# ---------------- API: Marcas e Modelos ----------------

@app.get("/api/marcas")
def listar_marcas():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT marca
            FROM erros
            WHERE marca IS NOT NULL AND marca <> ''
            ORDER BY marca
        """).fetchall()
    return [r["marca"] for r in rows]


@app.get("/api/modelos")
def listar_modelos(marca: str = Query(...)):
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT modelo
            FROM erros
            WHERE marca = ?
              AND modelo IS NOT NULL
              AND modelo <> ''
            ORDER BY modelo
        """, (marca,)).fetchall()
    return [r["modelo"] for r in rows]

# ---------------- Views ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request, q: Optional[str] = None, marca: Optional[str] = None,
         nivel: Optional[str] = None, criticidade: Optional[int] = None):
    results: List[Dict[str, Any]] = []
    if q:
        results = buscar(q, 20, marca, nivel, criticidade)
        log_busca(q, marca, nivel, criticidade, len(results))
    return templates.TemplateResponse("index.html", {
        "request": request, "query": q or "", "results": results
    })

# API de busca com filtros
@app.get("/api/buscar")
def api_buscar(
    termo: str = Query(None),
    limit: int = 20,
    marca: str | None = None,
    modelo: str | None = None,
    nivel: str | None = None,
    criticidade: int | None = None
):
    data = buscar(termo, limit, marca, modelo, nivel, criticidade)
    log_busca(termo, marca, nivel, criticidade, len(data))
    return JSONResponse(data)

# ---------------- Admin / Import / Export / Métricas ----------------
@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.post("/admin/reseed")
def admin_reseed():
    with get_conn() as conn:
        conn.execute("DELETE FROM erros")
        conn.commit()
    seed_if_empty()
    with get_conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM erros").fetchone()[0]
    return {"ok": True, "rows": n}

@app.get("/admin/metrics")
def admin_metrics():
    with get_conn() as conn:
        total_erros = conn.execute("SELECT COUNT(*) FROM erros").fetchone()[0]
        total_buscas = conn.execute("SELECT COUNT(*) FROM queries_log").fetchone()[0]
        total_fb = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        resolvidos = conn.execute("SELECT COUNT(*) FROM feedback WHERE resolvido = 1").fetchone()[0]
        top_termos = conn.execute("""
            SELECT termo, COUNT(*) as c FROM queries_log
            WHERE termo IS NOT NULL AND termo <> ''
              AND created_at >= datetime('now','-30 day')
            GROUP BY termo ORDER BY c DESC LIMIT 10
        """).fetchall()
        top_marcas = conn.execute("""
            SELECT marca, COUNT(*) as c FROM queries_log
            WHERE marca IS NOT NULL AND marca <> ''
              AND created_at >= datetime('now','-30 day')
            GROUP BY marca ORDER BY c DESC LIMIT 10
        """).fetchall()
    return {
        "totais": {
            "erros": total_erros,
            "buscas": total_buscas,
            "feedbacks": total_fb,
            "taxa_resolucao": (resolvidos / total_fb) if total_fb else None
        },
        "top_termos_30d": [{"termo": r[0], "count": r[1]} for r in top_termos],
        "top_marcas_30d": [{"marca": r[0], "count": r[1]} for r in top_marcas],
    }

# Export CSV/XLSX
@app.get("/export/csv")
def export_csv(
    termo: str = Query(...),
    marca: str | None = None,
    nivel: str | None = None,
    criticidade: int | None = None,
    limit: int = 1000
):
    rows = buscar(termo, limit, marca, nivel, criticidade)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id","codigo","titulo","marca","modelo","nivel","criticidade","termos","descricao","causas","solucoes","perguntas_cliente","perguntas_n1","procedimento_link","acao_recomendada"])
    for r in rows:
        writer.writerow([
            r["id"], r.get("codigo") or "", r.get("titulo") or "",
            r.get("marca") or "", r.get("modelo") or "",
            r.get("nivel") or "", r.get("criticidade") or "",
            ",".join(r.get("termos") or []),
            (r.get("descricao") or "").replace("\n"," ").strip(),
            " | ".join(r.get("causas") or []),
            " | ".join(r.get("solucoes") or []),
            " | ".join(r.get("perguntas_cliente") or []),
            " | ".join(r.get("perguntas_n1") or []),
            r.get("procedimento_link") or "",
            "AT"
        ])
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="export.csv"'}
    )

@app.get("/export/xlsx")
def export_xlsx(
    termo: str = Query(...),
    marca: str | None = None,
    nivel: str | None = None,
    criticidade: int | None = None,
    limit: int = 1000
):
    rows = buscar(termo, limit, marca, nivel, criticidade)
    df = pd.DataFrame(rows)
    for col in ["termos","causas","solucoes","perguntas_cliente","perguntas_n1"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Resultados")
    out.seek(0)
    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="export.xlsx"'}
    )

# Importador CSV/XLSX
@app.post("/admin/import")
async def admin_import(file: UploadFile = File(...)):
    content = await file.read()
    name = (file.filename or "").lower()
    try:
        if name.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception:
        df = pd.read_csv(io.BytesIO(content), sep=";", encoding="latin1")

    def pick(*cands):
        for c in df.columns:
            for cand in cands:
                if str(c).strip().lower() == cand:
                    return c
        return None

    col_titulo = pick("titulo","título","erro","problema","issue","mensagem") or df.columns[0]
    col_codigo = pick("codigo","código","code")
    col_desc   = pick("descricao","descrição","description")
    col_marca  = pick("marca","brand")
    col_modelo = pick("modelo","model","modelo_explicit")
    col_crit   = pick("criticidade","severidade","severity")
    col_nivel  = pick("nivel","nível","level")
    col_termos = pick("termos","tags","palavras","keywords")

    frase_final = "Caso não seja possível resolver remotamente, enviar para AT."
    frase_final_key = _normalize_text_for_compare(frase_final)

    inserted = 0
    with get_conn() as conn:
        for _, row in df.iterrows():
            titulo = str(row.get(col_titulo, "")).strip()
            if not titulo:
                continue
            codigo = (str(row.get(col_codigo)) if col_codigo else "") or ""
            descricao = str(row.get(col_desc, "")).strip() if col_desc else ""
            marca = (str(row.get(col_marca, "")) or "").strip()
            modelo= (str(row.get(col_modelo, "")) or "").strip()
            criticidade = row.get(col_crit)
            try:
                criticidade = int(criticidade) if criticidade == criticidade else 3  # NaN check
            except Exception:
                criticidade = 3
            nivel = (str(row.get(col_nivel, "")).strip().lower() if col_nivel else "atendente")
            if nivel not in ("n1","atendente"):
                nivel = "atendente"
            termos: List[str] = []
            if col_termos:
                termos = [t.strip().lower() for t in str(row.get(col_termos,"")).split(",") if t.strip()]

            perguntas_cliente = humanize_cliente({
                "codigo": codigo, "titulo": titulo, "termos": termos, "descricao": descricao
            })
            perguntas_n1: List[str] = []

            # por enquanto, só garantimos a frase final padrão, sem duplicar
            solucoes = [frase_final]

            conn.execute("""
                INSERT INTO erros
                (codigo,titulo,marca,modelo,descricao,causas,solucoes,perguntas_cliente,perguntas_n1,
                 nivel,procedimento_link,criticidade,termos,acao_recomendada)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                codigo, titulo, marca, modelo, descricao,
                json.dumps([], ensure_ascii=False),
                json.dumps(solucoes, ensure_ascii=False),
                json.dumps(perguntas_cliente, ensure_ascii=False),
                json.dumps(perguntas_n1, ensure_ascii=False),
                nivel, "", criticidade, ",".join(termos), "AT"
            ))
            inserted += 1
        conn.commit()
    return {"ok": True, "inserted": inserted}

# ---------------- Feedback ----------------
@app.post("/feedback")
async def feedback(erro_id: int = Form(...), resolvido: int = Form(...), comentario: str = Form("")):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback (erro_id, resolvido, comentario) VALUES (?, ?, ?)",
            (erro_id, 1 if int(resolvido) else 0, comentario.strip())
        )
        conn.commit()
    return {"ok": True}

# ---------------- Treino e Predição ----------------
def _load_training_data():
    X_text: List[str] = []
    y_idx: List[int] = []
    with get_conn() as conn:
        rows = conn.execute("SELECT id, titulo, descricao FROM erros").fetchall()
    for r in rows:
        text = f"{r['titulo']} {r['descricao'] or ''}".strip()
        if text:
            X_text.append(text)
            y_idx.append(int(r["id"]))
    return X_text, y_idx

@app.post("/admin/train")
def admin_train():
    X_text, y_idx = _load_training_data()
    if not X_text:
        return {"ok": False, "detail": "Sem dados para treinar."}
    vec = TfidfVectorizer(min_df=1, max_features=20000, ngram_range=(1,2))
    cls = LogisticRegression(max_iter=250)
    X = vec.fit_transform(X_text)
    cls.fit(X, y_idx)
    joblib.dump({"vec": vec, "cls": cls}, MODEL_PATH)
    return {"ok": True, "trained": len(y_idx)}

@app.post("/predict")
def predict(texto: str = Form(...), topk: int = 3):
    if not os.path.exists(MODEL_PATH):
        return {"ok": False, "detail": "Modelo não treinado. Use /admin/train."}
    bundle = joblib.load(MODEL_PATH)
    vec, cls = bundle["vec"], bundle["cls"]

    import numpy as np

    X = vec.transform([texto])
    probs = cls.predict_proba(X)[0]
    classes = cls.classes_
    top_idx = np.argsort(probs)[::-1][:topk]
    ids = [int(classes[i]) for i in top_idx]

    with get_conn() as conn:
        placeholders = ",".join(["?"] * len(ids))
        rows = conn.execute(f"SELECT * FROM erros WHERE id IN ({placeholders})", ids).fetchall()

    id_to_row = {int(r["id"]): r for r in rows}
    ordered = []
    for i, rid in enumerate(ids):
        r = id_to_row.get(int(rid))
        if not r:
            continue
        termos_list = (r["termos"] or "").split(",") if r["termos"] else []
        ordered.append({
            "id": int(r["id"]),
            "codigo": r["codigo"],
            "titulo": r["titulo"],
            "nivel": r["nivel"],
            "criticidade": r["criticidade"],
            "marca": r["marca"],
            "modelo": r["modelo"],
            "score": float(probs[top_idx[i]]),
            "termos": termos_list,
        })
    return {"ok": True, "results": ordered}

# ---------------- Buscar -------------------

@app.get("/buscar")
def buscar_html(
    request: Request,
    termo: str | None = None,
    marca: str | None = None,
    modelo: str | None = None,
):
    # if not marca:
    #     return templates.TemplateResponse(
    #         "buscar.html",
    #         {"request": request,
    #          "resultados": None,
    #          "query_error": "Selecione uma marca para buscar",
    #          "marca": None,
    #          "modelo": None}
    #     ) 
    
    resultados = buscar(
        termo=termo,
        limit = 20,
        filtro_marca = marca,
        filtro_modelo = modelo,
    )

    return templates.TemplateResponse(
        "buscar.html",
        {
            "request": request,
            "resultados": resultados,
            "marca": marca,
            "modelo": modelo,
            "query_error": None
        }
    )

@app.get("/")
def home():
    return {
        "projeto": "Manual Triagem",
        "status": "online",
        "ambiente": "produção"
    }

def verificar_admin(x_token: str = Header(None)):
    if x_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Acesso Negado")
    
@app.get("/admin")
def admin(_:str = Depends(verificar_admin)):
    return {"area": "admin"}

@app.post("/treinar-modelo")
def treinar_modelo(_:str = Depends(verificar_admin)):
    return {"status" : "Treinamento Iniciado"}

admin_router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(verificar_admin)]
)

@admin_router.get("")
def admin_home():
    return {"area": "admin"}

@admin_router.post("/train")
def train():
    return {"status" : "Treinando"}

app.include_router(admin_router)