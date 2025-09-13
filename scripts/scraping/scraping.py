import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotInteractableException, StaleElementReferenceException
import argparse
import re
import sys
import time
import json
import os
from urllib.parse import urlparse, urljoin, urlunparse
from webdriver_manager.chrome import ChromeDriverManager
import hashlib
from urllib.robotparser import RobotFileParser
import winsound
from plyer import notification

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def normalize_url(url):
    parsed = urlparse(url)
    normalized = parsed._replace(query="", fragment="")
    return urlunparse(normalized)

def hash_content(html):
    return hashlib.md5(html.encode('utf-8')).hexdigest()

def setup_driver(headless=True, user_agent=None):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--enable-unsafe-swiftshader")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-notifications")
    if user_agent:
        chrome_options.add_argument(f"--user-agent={user_agent}")
    chrome_options.add_argument("--plugins-enabled")
    chrome_options.add_argument("--enable-pdf-viewer")
    prefs = {
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": False,
    }
    chrome_options.add_experimental_option("prefs", prefs)
    try:
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=chrome_options)
    except Exception as e:
        print(f"Error setting up ChromeDriver: {e}")
        sys.exit(1)

def is_pdf_url(url, file_types=("pdf", "csv", "xlsx", "xls", "json", "docx", "doc")):
    if not url:
        return False
    parsed = urlparse(url)
    path = parsed.path.lower()
    query = parsed.query.lower()
    if any(path.endswith(f".{ext}") for ext in file_types) or any(ext in query for ext in file_types):
        return True
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        content_type = response.headers.get('Content-Type', '').lower()
        content_type_map = {
            'application/pdf': 'pdf',
            'text/csv': 'csv',
            'application/vnd.ms-excel': 'xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
            'application/json': 'json',
            'application/msword': 'doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx'
        }
        for ctype, ext in content_type_map.items():
            if ctype in content_type and ext in file_types:
                return True
    except requests.RequestException:
        pass
    return False

def download_pdf(url, file_type_dirs, file_counter, file_types=("pdf", "csv", "xlsx", "xls", "json", "docx", "doc", "txt"),
                 robot_parser=None, user_agent=None):
    if robot_parser and not robot_parser.can_fetch(user_agent, url):
        print(f"Skipping file (disallowed by robots.txt): {url}")
        return None
    try:
        headers = {'User-Agent': user_agent} if user_agent else {}
        response = requests.get(url, stream=True, timeout=30, headers=headers)
        if response.status_code != 200:
            print(f"Failed to download: {url} (Status code: {response.status_code})")
            return None
        content_type = response.headers.get('Content-Type', '').lower()
        ext_map = {
            "application/pdf": "pdf",
            "text/csv": "csv",
            "application/csv": "csv",
            "application/vnd.ms-excel": "xls",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
            "application/json": "json",
            "application/msword": "doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "text/plain": "txt"
        }
        ext = ext_map.get(content_type)
        if not ext:
            path = urlparse(url).path.lower()
            for ftype in file_types:
                if path.endswith(f".{ftype}"):
                    ext = ftype
                    break
        if not ext:
            print(f"No directory mapped for type: {content_type or 'unknown'}")
            ext = "others"
        save_dir = file_type_dirs.get(ext)
        if not save_dir:
            save_dir = os.path.join(os.path.dirname(list(file_type_dirs.values())[0]), "others")
            os.makedirs(save_dir, exist_ok=True)
            file_type_dirs[ext] = save_dir
        filename = os.path.basename(urlparse(url).path)
        if not filename or '.' not in filename:
            filename = f"downloaded_file_{file_counter}.{ext}"
        output_path = os.path.join(save_dir, filename)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {url} → {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading from {url}: {e}")
        return None

def extract_element_data(element, include_attrs=False):
    if element.name is None:
        text = clean_text(element.string or '')
        return {"type": "text", "content": text} if text else None
    element_info = {
        "type": "element",
        "tag": element.name,
        "children": []
    }
    if element.get('id'):
        element_info["id"] = element.get('id')
    if element.get('class'):
        element_info["classes"] = ' '.join(element.get('class'))
    if include_attrs:
        attrs = {
            k: ' '.join(v) if isinstance(v, list) else v
            for k, v in element.attrs.items() if k not in ['id', 'class']
        }
        if attrs:
            element_info["attributes"] = attrs
    direct_text = clean_text(''.join(element.strings))
    if direct_text:
        element_info["text"] = direct_text
    for child in element.children:
        child_data = extract_element_data(child, include_attrs)
        if child_data:
            element_info["children"].append(child_data)
    return element_info

def extract_structured_content(soup, url, format='text', include_attrs=False):
    title = soup.title.string if soup.title else "No title found"
    for tag in soup(['script', 'style', 'meta', 'link']):
        tag.decompose()
    if format in ['json', 'detailed']:
        body = soup.body or soup
        structure = extract_element_data(body, include_attrs)
        result = {
            "url": url,
            "domain": urlparse(url).netloc,
            "title": title,
            "structure": structure
        }
        return json.dumps(result, indent=2) if format == 'json' else "\n".join(format_detailed_text(result))
    output, seen_content = [f"URL: {url}", f"Title: {title}", ""], set()
    for el in soup.find_all(['div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article', 'section', 'main', 'aside', 'header', 'footer', 'nav', 'ul', 'ol', 'li', 'table']):
        text = clean_text(el.get_text())
        if text and text not in seen_content:
            seen_content.add(text)
            output.append("")
            output.append(text)
    final_output, prev_blank = [], False
    for line in output:
        if line.strip() == "":
            if not prev_blank:
                final_output.append("")
                prev_blank = True
        else:
            final_output.append(line)
            prev_blank = False
    return "\n".join(final_output)

def format_detailed_text(data, indent=0):
    lines = []
    prefix = " " * indent
    if indent == 0:
        lines.append(f"URL: {data['url']}")
        lines.append(f"Domain: {data['domain']}")
        lines.append(f"Title: {data['title']}")
        lines.append("")
        data = data['structure']
    tag_info = f"{prefix}<{data['tag']}"
    if 'id' in data:
        tag_info += f" id=\"{data['id']}\""
    if 'classes' in data:
        tag_info += f" class=\"{data['classes']}\""
    tag_info += ">"
    lines.append(tag_info)
    if 'text' in data and data['text'].strip():
        lines.append(f"{prefix}  TEXT: {data['text']}")
    for child in data.get('children', []):
        lines.extend(format_detailed_text(child, indent + 2))
    return lines

def find_clickable_elements(driver):
    selectors = [
        {"type": "Button", "selector": By.TAG_NAME, "value": "button"},
        {"type": "Link", "selector": By.TAG_NAME, "value": "a"},
        {"type": "Input", "selector": By.CSS_SELECTOR, "value": "input[type='submit'], input[type='button']"},
        {"type": "Clickable", "selector": By.CSS_SELECTOR, "value": "[onclick], [role='button'], [class*='btn'], [class*='button']"}
    ]
    return selectors

def collect_pdf_links_by_selector(driver, base_url, file_types=("pdf", "csv", "xlsx", "xls", "json", "docx", "doc")):
    file_links = []
    try:
        links = driver.find_elements(By.TAG_NAME, "a")
        for link in links:
            try:
                href = link.get_attribute("href")
                if href and is_pdf_url(href, file_types):
                    text = link.text or os.path.basename(urlparse(href).path) or "Download"
                    full_url = urljoin(base_url, href)
                    file_links.append((full_url, text))
            except StaleElementReferenceException:
                continue
    except Exception as e:
        print(f"Error collecting file links: {e}")
    return file_links

def safe_click_by_selector(driver, selector_type, selector_value, wait_time, tried_selectors):
    current_url = driver.current_url
    clicked_something = False
    try:
        elements = driver.find_elements(selector_type, selector_value)
        valid_elements = []
        for i, element in enumerate(elements):
            try:
                element_id = None
                if element.get_attribute("id"):
                    element_id = f"id:{element.get_attribute('id')}"
                elif element.get_attribute("name"):
                    element_id = f"name:{element.get_attribute('name')}"
                elif element.get_attribute("class"):
                    element_id = f"class:{element.get_attribute('class')}_{i}"
                else:
                    text = element.text if element.text else ""
                    element_id = f"text:{text[:20]}_{i}"
                if element_id in tried_selectors:
                    continue
                valid_elements.append((element, element_id))
            except StaleElementReferenceException:
                continue
            except Exception:
                continue
        for element, element_id in valid_elements:
            try:
                tried_selectors.add(element_id)
                driver.execute_script("arguments[0].scrollIntoView(true);", element)
                time.sleep(0.5)
                element.click()
                time.sleep(wait_time)
                new_url = driver.current_url
                if new_url != current_url:
                    print(f"Successfully clicked {element_id}, URL changed to: {new_url}")
                    clicked_something = True
                    return True, new_url
                print(f"Clicked {element_id}, but URL remained the same")
            except StaleElementReferenceException:
                print(f"Element became stale during click: {element_id}")
                continue
            except Exception as e:
                print(f"Error clicking {element_id}: {str(e)[:100]}")
                continue
    except Exception as e:
        print(f"Error finding or interacting with elements: {e}")
    return clicked_something, driver.current_url

def setup_robots_parser(base_url, user_agent):
    try:
        parsed_url = urlparse(base_url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        print(f"Reading robots.txt from: {robots_url}")
        rp.read()
        if rp.mtime() == 0:
            print("No robots.txt found or couldn't be parsed, proceeding with caution")
        else:
            print("Successfully loaded robots.txt")
        return rp
    except Exception as e:
        print(f"Error setting up robots parser: {e}")
        rp = RobotFileParser()
        return rp

def extract_and_navigate(url, output_dir, max_depth=2, interactive=False, headless=True, format='text', 
                         include_attrs=False, wait_time=1, infinite=False, respect_robots=True, user_agent=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_type_dirs = {
        "pdf": os.path.join(output_dir, "downloads", "pdf"),
        "csv": os.path.join(output_dir, "downloads", "csv"),
        "xlsx": os.path.join(output_dir, "downloads", "excel"),
        "xls": os.path.join(output_dir, "downloads", "excel"),
        "json": os.path.join(output_dir, "downloads", "json"),
        "doc": os.path.join(output_dir, "downloads", "docs"),
        "docx": os.path.join(output_dir, "downloads", "docs"),
        "txt": os.path.join(output_dir, "downloads", "text"),
        "others": os.path.join(output_dir, "downloads", "others")
    }
    for dir_path in file_type_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    if not user_agent:
        user_agent = "Mozilla/5.0 (compatible; MyWebCrawler/1.0; +https://example.com/bot)"
    robot_parser = None
    if respect_robots:
        robot_parser = setup_robots_parser(url, user_agent)
    driver = setup_driver(headless, user_agent)
    visited_urls = {}
    content_hashes = set()
    downloaded_files = set()
    normalized_to_url = {}
    state = {'pdf_counter': 1}
    def process_page(current_url, depth=0, tried_selectors=None):
        if tried_selectors is None:
            tried_selectors = set()
        normalized_url = normalize_url(current_url)
        if (not infinite and depth > max_depth) or normalized_url in visited_urls:
            return
        supported_types = ("pdf", "csv", "xlsx", "xls", "json", "docx", "doc", "txt")
        if is_pdf_url(current_url, supported_types):
            if current_url not in downloaded_files:
                file_path = download_pdf(current_url, file_type_dirs, state['pdf_counter'], supported_types, robot_parser, user_agent)
                if file_path:
                    downloaded_files.add(current_url)
                    state['pdf_counter'] += 1
            return
        try:
            print(f"Navigating to: {current_url}")
            driver.get(current_url)
            WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(wait_time)
            html = driver.execute_script("return document.documentElement.outerHTML")
            content_hash = hash_content(html)
            if content_hash in content_hashes:
                print(f"Skipping duplicate content: {current_url}")
                return
            content_hashes.add(content_hash)
            soup = BeautifulSoup(html, 'html.parser')
            links = collect_pdf_links_by_selector(driver, current_url, file_types=supported_types)
            for file_url, _ in links:
                if file_url not in downloaded_files:
                    if not respect_robots or robot_parser.can_fetch(user_agent, file_url):
                        file_path = download_pdf(file_url, file_type_dirs, state['pdf_counter'], supported_types, robot_parser, user_agent)
                        if file_path:
                            downloaded_files.add(file_url)
                            state['pdf_counter'] += 1
                    else:
                        print(f"Skipping (robots.txt disallowed): {file_url}")
            filename_base = re.sub(r'[^\w]', '_', urlparse(normalized_url).path or 'index')
            filename = f"{filename_base or 'index'}_{depth}"
            output_path = os.path.join(output_dir, f"{filename}.{format}")
            content = extract_structured_content(soup, current_url, format, include_attrs)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            text_download_path = os.path.join(file_type_dirs['txt'], f"{filename}.{format}")
            with open(text_download_path, 'w', encoding='utf-8') as f:
                f.write(content)
            visited_urls[normalized_url] = output_path
            normalized_to_url[normalized_url] = current_url
            for link in soup.find_all("a", href=True):
                href = link['href']
                joined_url = urljoin(current_url, href)
                norm_link = normalize_url(joined_url)
                if (urlparse(norm_link).netloc == urlparse(url).netloc and 
                    norm_link not in visited_urls and 
                    norm_link != normalized_url):
                    if not respect_robots or robot_parser.can_fetch(user_agent, joined_url):
                        process_page(joined_url, depth + 1)
                    else:
                        print(f"Skipping (disallowed by robots.txt): {joined_url}")
        except Exception as e:
            print(f"Error processing {current_url}: {e}")
    try:
        process_page(url)
    finally:
        driver.quit()
    print(f"Downloaded {len(downloaded_files)} files.")
    return visited_urls, downloaded_files

def main():
    parser = argparse.ArgumentParser(description='Dynamic website content scraper with navigation, PDF extraction, and robots.txt support.')
    parser.add_argument('url', help='Starting URL to scrape')
    parser.add_argument('-o', '--output', default='Justlit-dir', help='Uniform output directory (default: Justlit-dir)')
    parser.add_argument('-d', '--depth', type=int, default=2, help='Max navigation depth')
    parser.add_argument('-i', '--interactive', action='store_true', help='Enable interactive mode')
    parser.add_argument('-v', '--visible', action='store_true', help='Make browser visible')
    parser.add_argument('-f', '--format', choices=['text', 'json', 'detailed'], default='text', help='Output format')
    parser.add_argument('-a', '--attributes', action='store_true', help='Include HTML attributes')
    parser.add_argument('-w', '--wait', type=float, default=1, help='Wait time after load (seconds)')
    parser.add_argument('--infinite', action='store_true', help='Scrape until no new unique pages are found')
    parser.add_argument('-r', '--robots', action='store_true', default=False, help='Respect robots.txt rules')
    parser.add_argument('-u', '--user-agent', default=None, help='Set custom user agent')
    try:
        args = parser.parse_args()
        url = args.url if args.url.startswith(('http://', 'https://')) else 'https://' + args.url
        print(f"Extracting from: {url}")
        print(f"Respecting robots.txt: {'Yes' if args.robots else 'No'}")
        visited, pdfs = extract_and_navigate(
            url=url,
            output_dir=args.output,
            max_depth=args.depth,
            interactive=args.interactive,
            headless=not args.visible,
            format=args.format,
            include_attrs=args.attributes,
            wait_time=args.wait,
            infinite=args.infinite,
            respect_robots=args.robots,
            user_agent=args.user_agent
        )
        print(f"\nDone. {len(visited)} pages and {len(pdfs)} files saved in {args.output}/")
        print(json.dumps({
            "status": "success",
            "pages_saved": len(visited),
            "files_downloaded": len(pdfs),
            "output_dir": args.output
        }))
        winsound.Beep(1000, 500)
        notification.notify(
            title='Scraping Complete',
            message=f"{len(visited)} pages and {len(pdfs)} files were saved to {args.output}/",
            app_name='Web Scraper',
            timeout=10
        )
    except Exception as e:
        import traceback
        error_message = f"{type(e).__name__}: {str(e)}"
        print(traceback.format_exc())
        notification.notify(
            title='Scraping Failed',
            message=error_message,
            app_name='Web Scraper',
            timeout=600
        )

if __name__ == "__main__":
    main()