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

def parse_csv_file(file_path, output_folder):
    try:
        df = pd.read_csv(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_folder, f"{base_name}.txt")

        raw_text = df.to_string(index=False)
        cleaned = clean_text(raw_text)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def parse_all_csv_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".csv"):
            parse_csv_file(os.path.join(input_folder, filename), output_folder)

# === Set your folders ===
csv_folder = "Justlit-dir/downloads/csv"
output_folder = "output2"
parse_all_csv_files(csv_folder, output_folder)
