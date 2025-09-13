import os
import json
import re

def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = re.sub(r'\s+', ' ', line.strip())  # Normalize spaces/tabs
        if line:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def flatten_json(obj, prefix=''):
    items = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            items.extend(flatten_json(v, f"{prefix}{k}." if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.extend(flatten_json(v, f"{prefix}{i}."))
    else:
        items.append(f"{prefix}: {obj}")
    return items

def parse_json_file(file_path, output_folder):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        flat_items = flatten_json(data)
        raw_text = '\n'.join(flat_items)
        cleaned = clean_text(raw_text)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_folder, f"{base_name}.txt")
        with open(output_path, 'w', encoding='utf-8') as out_file:
            out_file.write(cleaned)

        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def parse_all_json_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".json"):
            parse_json_file(os.path.join(input_folder, filename), output_folder)

# === Set your folders ===
json_folder = "Justlit-dir/downloads/json"
output_folder = "output2"
parse_all_json_files(json_folder, output_folder)
