import os
import re
import logging
import sys

# --------- USER: Set your folders here ---------
INPUT_PDF_DIR = "Justlit-dir/downloads/pdf"
OUTPUT_TXT_DIR = "output2"
# -----------------------------------------------

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/pdf_only_parser.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

try:
    import pdfplumber
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
    from io import BytesIO
except ImportError:
    print("Required libraries missing! Please install pdfplumber, pymupdf, pytesseract, and pillow.")
    sys.exit(1)

# Add this at the top of your script, after imports
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = re.sub(r'\s+', ' ', line.strip())
        if line:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def parse_pdf(file_path, output_folder):
    pdf_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file_path = os.path.join(output_folder, f"{pdf_name}_parsed.txt")
    try:
        with open(output_file_path, "w", encoding="utf-8") as out_file:
            with pdfplumber.open(file_path) as pdf:
                doc = fitz.open(file_path)
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if not page_text or len(page_text.strip()) < 10:
                        # OCR fallback
                        page_obj = doc.load_page(i)
                        pix = page_obj.get_pixmap(dpi=300)
                        img_bytes = pix.tobytes("png")
                        img = Image.open(BytesIO(img_bytes))
                        page_text = pytesseract.image_to_string(img)
                    combined_text = f"\n--- Page {i+1} ---\n{clean_text(page_text.strip())}\n"
                    out_file.write(combined_text)
        msg = f"Saved: {output_file_path}"
        print(msg)
        logging.info(msg)
    except Exception as e:
        msg = f"Failed to process PDF {file_path}: {e}"
        print(msg)
        logging.error(msg)

def process_pdfs(input_folder, output_folder):
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    if not files:
        print(f"No PDF files found in {input_folder}")
        return
    for filename in files:
        file_path = os.path.join(input_folder, filename)
        parse_pdf(file_path, output_folder)

if __name__ == "__main__":
    process_pdfs(INPUT_PDF_DIR, OUTPUT_TXT_DIR)
