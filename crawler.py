import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

def safe_filename(url):
    """Generate a safe, unique filename for each URL."""
    return url.split('/')[-1].split('.')[0] + '.txt'

def get_text_and_links(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except Exception as e:
        print(f"Skipping {url}: {e}")
        return "", []

    html = response.text.replace("<br>", "<br/>")
    soup = BeautifulSoup(html, "html.parser")

    # Extract visible text only
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text()
    text = text.replace("Want to start a startup?  Get funded by\nY Combinator.", "")
    text = text.replace("Watch how this essay was\nwritten.", "")

    # Extract same-domain links
    links = []
    for a in soup.find_all("a", href=True):
        link = urljoin(url, a["href"])
        if urlparse(link).netloc == urlparse(url).netloc:
            links.append(link)

    return text, links

def crawl_one_hop(start_url, output_dir="essays"):
    os.makedirs(output_dir, exist_ok=True)

    # Crawl the start page
    print(f"Crawling root: {start_url}")
    _, links = get_text_and_links(start_url)

    # Crawl only direct children (1 hop)
    for link in sorted(links):
        print(f"Crawling child: {link}")
        child_text, _ = get_text_and_links(link)
        if child_text.strip():
            filename = safe_filename(link)
            if filename == 'index.txt':
                continue
            filename = os.path.join(output_dir, filename)
            with open(filename, "w", encoding="utf-8") as f:
                f.write(child_text)

if __name__ == "__main__":
    start_url = "https://www.paulgraham.com/articles.html"
    crawl_one_hop(start_url)
    print("Crawling finished. Clean text saved in 'crawled_pages/' folder.")