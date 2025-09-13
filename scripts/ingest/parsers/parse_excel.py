import os
import re
import pandas as pd

def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = re.sub(r'\s+', ' ', line.strip())  # Remove extra spaces and tabs
        if line:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def parse_excel_file(file_path, output_folder):
    try:
        df = pd.read_excel(file_path, sheet_name=None)  # All sheets
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        for sheet_name, sheet_df in df.items():
            output_name = f"{base_name}_{sheet_name}.txt"
            output_path = os.path.join(output_folder, output_name)

            raw_text = sheet_df.to_string(index=False)
            cleaned = clean_text(raw_text)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned)

            print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def parse_all_excel_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".xlsx", ".xls")):
            parse_excel_file(os.path.join(input_folder, filename), output_folder)

# === Set your folders ===
excel_folder = "Justlit-dir/downloads/excel"  # Folder with original .xlsx files
output_folder = "output2"
parse_all_excel_files(excel_folder, output_folder)
