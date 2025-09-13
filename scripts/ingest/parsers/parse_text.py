import os
import re

def clean_text(content):
    lines = content.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        line = re.sub(r'\s+', ' ', line)  # Normalize spaces and tabs
        if line:  # Skip blank lines
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def clean_all_text_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".text" ) or filename.lower().endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            with open(input_path, "r", encoding="utf-8") as infile:
                content = infile.read()

            cleaned_content = clean_text(content)

            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{base_name}_cleaned.txt")
            with open(output_path, "w", encoding="utf-8") as outfile:
                outfile.write(cleaned_content)

            try:
                print(f"Cleaned: {filename.encode('utf-8', errors='replace').decode('utf-8')} -> {output_path.encode('utf-8', errors='replace').decode('utf-8')}")
            except Exception as e:
                print(f"Error processing {repr(filename)}: {e}")

input_folder = "Justlit-dir/downloads/text"           # Folder with original .txt files
output_folder = "output2"  # Folder to store cleaned .text files

clean_all_text_files(input_folder, output_folder)
