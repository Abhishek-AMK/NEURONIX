import multiprocessing
import threading
import time
import datetime
import sys
import os
import re
from scraping import extract_and_navigate

UNIFORM_OUTPUT_DIR = "Justlit-dir"

#  Read URLs from file
def read_urls(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"URL file not found: {file_path}")
    urls = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                raise ValueError(f"Invalid line format: {line.strip()}")
            url = parts[0].strip()
            depth = int(parts[1].strip()) if parts[1].strip().isdigit() else None
            # Force all outputs to the uniform directory
            folder = UNIFORM_OUTPUT_DIR
            urls.append((url, depth, folder))
    return urls

# Logger
def log(folder, message):
    time_str = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_line = f"{time_str} {message}"
    log_file = os.path.join(folder, "scrape_log.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

def sanitize_filename(filename):
    # Handle Unicode characters properly
    try:
        # Try to encode as ASCII first, replacing problematic characters
        filename = filename.encode('ascii', 'ignore').decode('ascii')
    except UnicodeEncodeError:
        pass
    
    # Replace Unicode arrow and other problematic characters
    filename = filename.replace('\u2192', '->')  # Right arrow
    filename = filename.replace('\u2190', '<-')  # Left arrow
    filename = filename.replace('\u2191', '^')   # Up arrow
    filename = filename.replace('\u2193', 'v')   # Down arrow
    
    # Remove any remaining non-ASCII characters
    filename = re.sub(r'[^\x00-\x7F]+', '', filename)
    
    # Replace problematic filesystem characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    return filename

#  Scrape function
def scrape_link(url, max_depth, output_folder, user_agent, skip_flag):
    try:
        visited, files = extract_and_navigate(
            url=url,
            output_dir=output_folder,
            max_depth=max_depth if max_depth is not None else 100,
            interactive=False,
            headless=True,
            format='text',
            include_attrs=False,
            wait_time=1,
            infinite=(max_depth is None),
            respect_robots=True,
            user_agent=user_agent
        )
        if skip_flag.value:
            log(output_folder, f"Skipped by user during scrape: {url}")
            return False
        log(output_folder, f"Done: {url}")
        log(output_folder, f"Pages: {len(visited)}, Files: {len(files)}")
        for v in visited:
            log(output_folder, f"Extracted from: {v}")
        return True
    except Exception as e:
        log(output_folder, f"Error while scraping {url}: {type(e).__name__}: {e}")
        return False

# Background input thread
def wait_for_skip(flag, current_url):
    input(f"\nPress Enter to skip current URL: {current_url}...\n")
    flag.value = 1

# Main logic
if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows compatibility

    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/58.0.3029.110 Safari/537.3"
    file_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\AbhishekKulkarni\Desktop\Full-Pipeline\scripts\scraping\urls.txt"

    try:
        urls = read_urls(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to read URL file: {e}")
        sys.exit(1)

    # Ensure uniform output directory and subfolders exist
    subfolders = [
        "downloads/pdf", "downloads/csv", "downloads/excel",
        "downloads/json", "downloads/docs", "downloads/text", "downloads/others"
    ]
    for sub in subfolders:
        os.makedirs(os.path.join(UNIFORM_OUTPUT_DIR, sub), exist_ok=True)

    for url, depth, folder in urls:
        os.makedirs(folder, exist_ok=True)
        attempt = 0
        success = False

        while attempt < 3 and not success:
            log(folder, f"Attempt {attempt+1}/3 - Scraping: {url} | Depth: {'∞' if depth is None else depth}")

            skip_flag = multiprocessing.Value('i', 0)
            skip_thread = threading.Thread(target=wait_for_skip, args=(skip_flag, url), daemon=True)
            skip_thread.start()

            process = multiprocessing.Process(
                target=scrape_link,
                args=(url, depth, folder, user_agent, skip_flag)
            )
            process.start()

            while process.is_alive():
                if skip_flag.value:
                    process.terminate()
                    log(folder, f"Skip requested for: {url}")
                    break
                time.sleep(0.5)

            process.join()

            if skip_flag.value:
                log(folder, f"Retry due to user skip: {url}")
            else:
                # Check last log to see if scrape was successful
                log_file = os.path.join(folder, "scrape_log.txt")
                if os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8") as f:
                        last_lines = f.readlines()[-5:]
                        if any("Done" in line for line in last_lines):
                            success = True
                        elif any("Skipped" in line for line in last_lines):
                            # Keep retrying if it was skipped
                            pass

            attempt += 1

        if not success:
            log(folder, f"All attempts failed for: {url}")

        log(folder, "-" * 60)