import os
import fitz  # PyMuPDF
import re
import csv

ROOT_DIR = r"C:\Users\joao.rodrigueiro\Desktop\Cópia manuais"


def iter_pdfs(root_dir):
    """
    Percorre a estrutura:
    ROOT/Marca/Modelo/*.pdf
    Retorna caminhos + marca + modelo
    """
    for marca in os.listdir(root_dir):
        path_marca = os.path.join(root_dir, marca)
        if not os.path.isdir(path_marca):
            continue

        for modelo in os.listdir(path_marca):
            path_modelo = os.path.join(path_marca, modelo)
            if not os.path.isdir(path_modelo):
                continue

            for file in os.listdir(path_modelo):
                if file.lower().endswith(".pdf"):
                    yield {
                        "marca": marca,
                        "modelo": modelo,
                        "arquivo": file,
                        "pdf_path": os.path.join(path_modelo, file),
                    }

def extract_text_sample(pdf_path, max_pages=5):
    """
    Extrai texto das primeiras páginas do PDF (se houver).
    Retorna uma string única.
    """
    text_chunks = []

    try:
        doc = fitz.open(pdf_path)
        total_pages = min(len(doc), max_pages)

        for i in range(total_pages):
            page = doc[i]
            text = page.get_text().strip()
            if text:
                text_chunks.append(text)

        doc.close()
    except Exception as e:
        print(f"[ERRO] Falha ao ler PDF: {pdf_path} -> {e}")

    return "\n".join(text_chunks)


ERROR_CODE_PATTERNS = [
    r"\bSC\d{3,4}\b",        # SC542, SC553
    r"\b\d{3}\.\d{2}\b",     # 900.43, 49.38
    r"\bJAM[-\s]?\d+\b",     # JAM-01
    r"\bADF[-\s]?\w+\b",     # ADF-XX
]

def extract_error_codes(text):
    found = set()
    for pattern in ERROR_CODE_PATTERNS:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            found.add(match.upper())
    return list(found)

def extract_error_related_text(pdf_path, max_pages=50):
    """
    Varre páginas do PDF e retorna texto apenas
    das páginas que parecem tratar de erros.
    """
    relevant_text = []

    try:
        doc = fitz.open(pdf_path)
        total_pages = min(len(doc), max_pages)

        for i in range(total_pages):
            page = doc[i]
            text = page.get_text().strip()
            if not text:
                continue

            text_lower = text.lower()

            if any(k in text_lower for k in ERROR_PAGE_KEYWORDS):
                relevant_text.append(text)

        doc.close()
    except Exception as e:
        print(f"[ERRO] Falha ao ler PDF: {pdf_path} -> {e}")

    return "\n".join(relevant_text)


ERROR_PAGE_KEYWORDS = [
    "error",
    "erro",
    "fault",
    "troubleshoot",
    "troubleshooting",
    "jam",
    "code",
    "message",
]

GENERIC_ERROR_PHRASES = [
    "paper jam",
    "ink error",
    "maintenance error",
    "service error",
    "printer error",
    "cartridge error",
    "cover open",
    "out of paper",
]

def extract_error_phrases(text):
    found = set()
    text_lower = text.lower()

    for phrase in GENERIC_ERROR_PHRASES:
        if phrase in text_lower:
            found.add(phrase)

    return list(found)

PHRASE_TO_INTERNAL_CODE = {
    "paper jam": {
        "codigo": "JAM-GENERICO",
        "titulo": "Atolamento de papel",
        "descricao": "Atolamento de papel detectado durante a operação",
    },
    "maintenance error": {
        "codigo": "MAINT-ERROR",
        "titulo": "Erro de manutenção",
        "descricao": "Erro relacionado à manutenção do equipamento",
    },
}

def normalize_error_phrase(phrase):
    return PHRASE_TO_INTERNAL_CODE.get(phrase)

def write_csv(rows, output_file="erros_extraidos.csv"):
    headers = ["codigo", "titulo", "descricao", "marca", "modelo", "termos"]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

def main():
    pdfs = list(iter_pdfs(ROOT_DIR))
    print(f"PDFs encontrados: {len(pdfs)}")

    rows = []
    seen = set()  # evita duplicados (marca, modelo, codigo)

    for p in pdfs:
        text = extract_error_related_text(p["pdf_path"])
        if not text:
            continue

        phrases = extract_error_phrases(text)

        for ph in phrases:
            norm = normalize_error_phrase(ph)
            if not norm:
                continue

            key = (p["marca"], p["modelo"], norm["codigo"])
            if key in seen:
                continue

            seen.add(key)

            rows.append({
                "codigo": norm["codigo"],
                "titulo": norm["titulo"],
                "descricao": norm["descricao"],
                "marca": p["marca"],
                "modelo": p["modelo"],
                "termos": ph.replace(" ", ","),
            })

    write_csv(rows)
    print(f"CSV gerado com {len(rows)} registros.")


if __name__ == "__main__":
    main()
