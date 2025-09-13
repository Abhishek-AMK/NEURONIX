import os

folders = [
    "backend",
    "frontend",
    "scripts"
]

backend_files = [
    "backend/app.py",
    "backend/requirements.txt"
]

frontend_files = [
    "frontend/App.jsx",
    "frontend/App.css"
]

script_files = [
    "scripts/scraping.py",
    "scripts/parse_pdf.py",
    "scripts/parse_csv.py",
    "scripts/chunking.py",
    "scripts/embedding.py",
    "scripts/rag_query.py",
    "scripts/db_exporter.py"
]

other_files = [
    "README.md"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for f in backend_files + frontend_files + script_files + other_files:
    with open(f, "w", encoding="utf-8") as file:
        file.write("")

print("Project structure created successfully!")
