import sys
import os

# Add project root to sys.path so "ingest" is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ingest.parsers.parse_pdf import parse_pdf
from ingest.parsers.parse_csv import parse_csv_file
from ingest.parsers.parse_json import parse_json_file
from ingest.parsers.parse_text import clean_text
from ingest.parsers.parse_excel import parse_excel_file
from ingest.parsers.parse_docx import parse_docx_file

OUTPUT_FOLDER = "output2"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Map extension to parser
PARSER_MAP = {
    ".pdf": lambda fp: parse_pdf(fp, OUTPUT_FOLDER),
    ".csv": lambda fp: parse_csv_file(fp, OUTPUT_FOLDER),
    ".json": lambda fp: parse_json_file(fp, OUTPUT_FOLDER),
    ".xls": lambda fp: parse_excel_file(fp, OUTPUT_FOLDER),
    ".xlsx": lambda fp: parse_excel_file(fp, OUTPUT_FOLDER),
    ".doc": lambda fp: parse_docx_file(fp, OUTPUT_FOLDER),
    ".docx": lambda fp: parse_docx_file(fp, OUTPUT_FOLDER),
    ".txt": None,   # Handled below
    ".text": None   # Handled below
}

# Where to look for files
FOLDERS = [
    os.path.join("Justlit-dir", "downloads", "pdf"),
    os.path.join("Justlit-dir", "downloads", "csv"),
    os.path.join("Justlit-dir", "downloads", "excel"),
    os.path.join("Justlit-dir", "downloads", "json"),
    os.path.join("Justlit-dir", "downloads", "docs"),
    os.path.join("Justlit-dir", "downloads", "text"),
]

def route_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    parser = PARSER_MAP.get(ext)
    if parser:
        return parser(file_path)
    elif ext in [".txt", ".text"]:
        # Special handling: read, clean, save
        with open(file_path, "r", encoding="utf-8") as infile:
            content = infile.read()
        cleaned_content = clean_text(content)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_cleaned.txt")
        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.write(cleaned_content)
        print(f"[✓] Cleaned: {file_path} → {output_path}")
        return output_path
    else:
        print(f"[✗] Unsupported file type: {file_path}")
        return None

def main():
    print("\n=== Batch Parsing Started ===\n")
    for folder in FOLDERS:
        if not os.path.exists(folder):
            print(f"Skipping missing folder: {folder}")
            continue
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        if not files:
            print(f"No files found in {folder}")
        for file_path in files:
            print(f"Processing: {file_path}")
            route_file(file_path)
    print("\n=== Batch Parsing Finished ===\n")

if __name__ == "__main__":
    main()
