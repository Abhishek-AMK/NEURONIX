import os
import json
import pandas as pd
import PyPDF2
import yake
import re
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
blacklist = {
    "one", "two", "three", "four", "licensed", "umcg", "ncs", "ncsa",
    "mednerdette", "image", "cc", "av", "modifiedtricuspid", "stethescope",
    "separates", "and", "the", "of", "in", "to", "for", "a"
}

def clean_text(text):
    """Clean and normalize text for keyword extraction."""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[•–"""\'-]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text.strip()

def read_text_file(path):
    """Read content from text file."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def read_pdf_file(path):
    """Extract text from PDF file."""
    text = ""
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def read_csv_file(path):
    """Read and concatenate CSV content."""
    df = pd.read_csv(path)
    return "\n".join(df.astype(str).apply(lambda x: " ".join(x), axis=1))

def yake_keywords(text, top_n=20):
    """Extract keywords using YAKE algorithm."""
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=top_n)
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0].lower() for kw in keywords if kw[0].isalpha() and len(kw[0]) > 2 and kw[0].lower() not in blacklist]

def main():
    """Main function for pipeline integration."""
    try:
        # Get file path from environment variable
        file_path = os.environ.get('file_path', '')
        top_n = int(os.environ.get('top_n', '20'))
        
        if not file_path:
            print("Error: No file path provided")
            sys.exit(1)
        
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

        print(f"Processing file: {file_path}")
        
        # Set up output directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '..', '..', 'keywords_output')
        os.makedirs(output_dir, exist_ok=True)

        # Read file content based on extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".txt":
            text = read_text_file(file_path)
        elif ext == ".pdf":
            text = read_pdf_file(file_path)
        elif ext == ".csv":
            text = read_csv_file(file_path)
        else:
            print(f"Error: Unsupported file type: {ext}")
            print("Supported formats: .pdf, .txt, .csv")
            sys.exit(1)

        # Extract keywords
        clean = clean_text(text)
        keywords = yake_keywords(clean, top_n)

        # Prepare result
        result = {
            "filename": os.path.basename(file_path),
            "file_path": file_path,
            "yake_keywords": keywords,
            "keyword_count": len(keywords)
        }

        # Save results
        base = os.path.splitext(os.path.basename(file_path))[0]
        json_file = os.path.join(output_dir, f"keywords_{base}.json")
        txt_file = os.path.join(output_dir, f"keywords_{base}.txt")

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Keywords extracted from: {os.path.basename(file_path)}\n")
            f.write("=" * 50 + "\n")
            for i, keyword in enumerate(keywords, 1):
                f.write(f"{i:2d}. {keyword}\n")

        # Terminal output
        print(f"\nKeyword extraction completed successfully!")
        print(f"File processed: {os.path.basename(file_path)}")
        print(f"Keywords extracted: {len(keywords)}")
        print(f"\nTop {min(10, len(keywords))} Keywords:")
        print("-" * 30)
        for i, keyword in enumerate(keywords[:10], 1):
            print(f"{i:2d}. {keyword}")
        
        if len(keywords) > 10:
            print(f"... and {len(keywords) - 10} more keywords")

        print(f"\nResults saved to:")
        print(f"   JSON: {json_file}")
        print(f"   TXT:  {txt_file}")

    except Exception as e:
        print(f"Keyword extraction failed: {e}")
        logger.error(f"Keyword extraction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
