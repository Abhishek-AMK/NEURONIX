import os
import re
from docx import Document

def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = re.sub(r'\s+', ' ', line.strip())  # Normalize spaces/tabs
        if line:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def parse_docx_file(file_path, output_folder):
    try:
        doc = Document(file_path)
        raw_text = '\n'.join([para.text for para in doc.paragraphs])
        cleaned = clean_text(raw_text)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_folder, f"{base_name}.txt")
        with open(output_path, 'w', encoding='utf-8') as out_file:
            out_file.write(cleaned)

        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def parse_all_docx_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".docx"):
            parse_docx_file(os.path.join(input_folder, filename), output_folder)

# === Set your folders ===
docx_folder = "Justlit-dir/downloads/docs"
output_folder = "output2"
parse_all_docx_files(docx_folder, output_folder)
