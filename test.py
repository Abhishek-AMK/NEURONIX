import os
import re
import PyPDF2
import pandas as pd
import yake
import ollama
from io import StringIO
import nltk
from nltk.corpus import stopwords
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download stopwords if not already present
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except Exception as e:
    logger.warning(f"Could not download stopwords: {e}")
    stop_words = set()

# === Step 1: Extract text from PDF ===
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file with error handling."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            logger.info(f"PDF has {len(reader.pages)} pages")
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
                except Exception as e:
                    logger.warning(f"Error reading page {page_num}: {e}")
                    continue
                    
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
    
    if not text.strip():
        raise ValueError("No text could be extracted from the PDF")
        
    return text

# === Step 2: Split into chunks ===
def split_into_chunks(text, chunk_size=1500):
    """Split text into chunks of specified size."""
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

# === Step 3: Extract keywords ===
def extract_yake_keywords(text_chunks, max_keywords=50):
    """Extract keywords using YAKE algorithm with improved filtering."""
    if not text_chunks:
        return []
    
    kw_extractor = yake.KeywordExtractor(
        top=max_keywords * 3,  # Extract more initially for better selection
        stopwords=None,
        n=1,  # Single words only
        dedupLim=0.7,
        windowsSize=1
    )
    
    keywords = set()
    
    for chunk in text_chunks:
        try:
            extracted = kw_extractor.extract_keywords(chunk)
            for phrase, score in extracted:
                phrase = phrase.lower().strip()
                
                # Only keep alphabetic words longer than 2 characters
                if (phrase.isalpha() and 
                    len(phrase) > 2 and 
                    phrase not in stop_words and
                    not phrase.isdigit()):
                    keywords.add(phrase)
                
                # Stop if we have enough keywords
                if len(keywords) >= max_keywords:
                    break
                    
        except Exception as e:
            logger.warning(f"Error extracting keywords from chunk: {e}")
            continue
    
    return sorted(list(keywords))[:max_keywords]

# === Step 4: Categorize keywords using Ollama ===
def categorize_keywords_with_ollama(keywords, model="llama3.2", max_retries=3):
    """Categorize keywords using Ollama with strict CSV format validation."""
    if not keywords:
        return pd.DataFrame(columns=['Keyword', 'Category'])
    
    # Define valid categories to guide the AI
    valid_categories = [
        'Country', 'City', 'Technology', 'Health', 'Transportation', 
        'Weather', 'Environment', 'Education', 'Business', 'Science',
        'Politics', 'Sports', 'Entertainment', 'Food', 'Religion',
        'Military', 'Economics', 'Geography', 'History', 'Culture',
        'Medicine', 'Engineering', 'Agriculture', 'Energy', 'Communication',
        'Finance', 'Law', 'Art', 'Literature', 'Music', 'Other'
    ]
    
    prompt = f"""
You are a keyword categorization system. Your task is to classify each keyword into exactly ONE of the following categories:

{', '.join(valid_categories)}

IMPORTANT INSTRUCTIONS:
1. Output ONLY a valid CSV format with exactly these two columns: Keyword,Category
2. Each keyword must be paired with exactly ONE category from the list above
3. Do NOT include any explanations, notes, or additional text
4. Do NOT include markdown formatting or code blocks
5. Each line must follow the format: keyword,category
6. Use exactly these column headers: Keyword,Category

Keywords to categorize:
{chr(10).join(keywords)}

Output format example:
Keyword,Category
technology,Technology
london,City
cancer,Health
"""

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1} to categorize {len(keywords)} keywords")
            
            response = ollama.chat(
                model=model, 
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1}  # Lower temperature for more consistent output
            )
            
            csv_text = response["message"]["content"].strip()
            
            # Clean the response - remove any markdown formatting or extra text
            csv_text = clean_csv_response(csv_text)
            
            # Validate and parse CSV
            df = validate_and_parse_csv(csv_text, valid_categories, keywords)
            
            if df is not None and not df.empty:
                logger.info(f"Successfully categorized {len(df)} keywords")
                return df
            else:
                logger.warning(f"Attempt {attempt + 1} failed: Invalid CSV format")
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            
        if attempt < max_retries - 1:
            logger.info("Retrying...")
    
    # If all attempts fail, create a fallback DataFrame
    logger.warning("All attempts failed. Creating fallback categorization.")
    return create_fallback_categorization(keywords, valid_categories)

def clean_csv_response(csv_text):
    """Clean the CSV response from any unwanted formatting."""
    lines = csv_text.split('\n')
    cleaned_lines = []
    
    found_header = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Remove markdown code block markers
        if line.startswith('``````'):
            continue
            
        # Remove any lines that don't look like CSV data
        if ',' not in line:
            continue
            
        # Check for header
        if line.lower().startswith('keyword,category') or not found_header:
            if line.lower().startswith('keyword,category'):
                cleaned_lines.append('Keyword,Category')
                found_header = True
            elif ',' in line and not found_header:
                cleaned_lines.insert(0, 'Keyword,Category')
                cleaned_lines.append(line)
                found_header = True
            else:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def validate_and_parse_csv(csv_text, valid_categories, original_keywords):
    """Validate and parse the CSV response."""
    try:
        # Parse CSV
        df = pd.read_csv(StringIO(csv_text))
        
        # Check if required columns exist
        if 'Keyword' not in df.columns or 'Category' not in df.columns:
            logger.error("CSV missing required columns: Keyword, Category")
            return None
        
        # Clean and validate data
        df = df.dropna()  # Remove rows with NaN values
        df['Keyword'] = df['Keyword'].astype(str).str.strip().str.lower()
        df['Category'] = df['Category'].astype(str).str.strip()
        
        # Validate categories
        valid_categories_lower = [cat.lower() for cat in valid_categories]
        df['Category'] = df['Category'].apply(lambda x: validate_category(x, valid_categories))
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Keyword'])
        
        # Only keep keywords that were in the original list
        original_keywords_lower = [kw.lower() for kw in original_keywords]
        df = df[df['Keyword'].isin(original_keywords_lower)]
        
        if df.empty:
            logger.warning("No valid keywords found after validation")
            return None
            
        return df
        
    except Exception as e:
        logger.error(f"Error parsing CSV: {str(e)}")
        return None

def validate_category(category, valid_categories):
    """Validate and correct category names."""
    category = str(category).strip()
    
    # Check exact match (case insensitive)
    for valid_cat in valid_categories:
        if category.lower() == valid_cat.lower():
            return valid_cat
    
    # Check partial match
    for valid_cat in valid_categories:
        if category.lower() in valid_cat.lower() or valid_cat.lower() in category.lower():
            return valid_cat
    
    # Default to 'Other' if no match found
    return 'Other'

def create_fallback_categorization(keywords, valid_categories):
    """Create a fallback categorization if Ollama fails."""
    logger.info("Creating fallback categorization using simple rules")
    
    # Simple rule-based categorization
    category_keywords = {
        'Country': ['usa', 'india', 'china', 'japan', 'germany', 'france', 'britain', 'russia'],
        'City': ['london', 'paris', 'tokyo', 'newyork', 'delhi', 'mumbai', 'bangalore'],
        'Technology': ['computer', 'software', 'internet', 'digital', 'data', 'system', 'network'],
        'Health': ['health', 'medical', 'hospital', 'doctor', 'patient', 'treatment', 'disease'],
        'Transportation': ['car', 'train', 'bus', 'flight', 'transport', 'vehicle', 'travel'],
        'Weather': ['rain', 'sunny', 'cloud', 'temperature', 'climate', 'weather', 'storm'],
        'Environment': ['environment', 'pollution', 'forest', 'green', 'nature', 'climate']
    }
    
    categorized_data = []
    
    for keyword in keywords:
        category = 'Other'  # Default category
        
        for cat, cat_keywords in category_keywords.items():
            if any(ck in keyword.lower() for ck in cat_keywords):
                category = cat
                break
        
        categorized_data.append({'Keyword': keyword.lower(), 'Category': category})
    
    return pd.DataFrame(categorized_data)

# === Step 5: Main Pipeline ===
def process_pdf_to_categorized_csv(pdf_path, model="llama3.2", chunk_size=1500, max_keywords=50):
    """Main pipeline to process PDF and generate categorized keywords CSV."""
    try:
        logger.info(f"Starting PDF processing: {pdf_path}")
        
        # Check if Ollama model is available
        try:
            ollama.chat(model=model, messages=[{"role": "user", "content": "test"}])
        except Exception as e:
            raise Exception(f"Ollama model '{model}' not available. Please ensure Ollama is running and model is installed.")
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        logger.info(f"Extracted {len(text)} characters from PDF")
        
        # Split into chunks
        chunks = split_into_chunks(text, chunk_size)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Extract keywords
        keywords = extract_yake_keywords(chunks, max_keywords)
        logger.info(f"Extracted {len(keywords)} keywords: {keywords[:10]}...")
        
        if not keywords:
            raise ValueError("No keywords could be extracted from the PDF")
        
        # Categorize keywords
        categorized_df = categorize_keywords_with_ollama(keywords, model=model)
        
        if categorized_df.empty:
            raise ValueError("No keywords could be categorized")
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_csv = f"{base_name}_categorized_keywords.csv"
        
        # Save to CSV
        categorized_df.to_csv(output_csv, index=False)
        logger.info(f"Successfully saved {len(categorized_df)} categorized keywords to: {output_csv}")
        
        # Display summary
        print(f"\n=== Processing Summary ===")
        print(f"PDF processed: {pdf_path}")
        print(f"Keywords extracted: {len(keywords)}")
        print(f"Keywords categorized: {len(categorized_df)}")
        print(f"Output file: {output_csv}")
        print(f"\nCategory distribution:")
        print(categorized_df['Category'].value_counts())
        
        return output_csv
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise e

# === Entry Point ===
if __name__ == "__main__":
    # Configuration
    pdf_path = r"C:\Users\AbhishekKulkarni\Downloads\Article 1-main.pdf"
    model = "llama3.2"  # Make sure this model is installed in Ollama
    
    try:
        output_file = process_pdf_to_categorized_csv(
            pdf_path=pdf_path,
            model=model,
            chunk_size=1500,
            max_keywords=50
        )
        print(f"\n✅ Successfully processed PDF. Output saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure Ollama is running: `ollama serve`")
        print("2. Install required model: `ollama pull llama3.2`")
        print("3. Check if PDF file exists and is readable")
        print("4. Verify all dependencies are installed")
