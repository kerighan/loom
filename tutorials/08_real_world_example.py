# Real World Example
# Web scraper with deduplication and persistent storage

from loom.database import DB
import os

# Clean up any existing database
try:
    if os.path.exists("scraper.loom"):
        os.remove("scraper.loom")
except PermissionError:
    pass

with DB("scraper.loom") as db:
    # Track seen URLs (fast deduplication)
    seen = db.create_bloomfilter("seen_urls", expected_items=100000)
    
    # Store scraped pages
    pages = db.create_list("pages", {
        "url": "U200",
        "title": "U200", 
        "status": "uint16",
        "timestamp": "uint64"
    })
    
    def process_url(url, title, status, timestamp):
        # Skip if already seen
        if url in seen:
            return False
        
        # Mark as seen
        seen.add(url)
        
        # Store result
        pages.append({
            "url": url,
            "title": title,
            "status": status,
            "timestamp": timestamp
        })
        return True
    
    # Simulate scraping
    urls = [
        ("https://example.com/1", "Page 1", 200, 1000),
        ("https://example.com/2", "Page 2", 200, 1001),
        ("https://example.com/1", "Page 1", 200, 1002),  # Duplicate!
        ("https://example.com/3", "Page 3", 404, 1003),
    ]
    
    for url, title, status, ts in urls:
        if process_url(url, title, status, ts):
            print(f"Scraped: {url}")
        else:
            print(f"Skipped (duplicate): {url}")
    
    print(f"\nTotal pages: {len(pages)}")
    print(f"URLs seen: {len(seen)}")
    
    # Get all successful pages
    successful = [p for p in pages if p["status"] == 200]
    print(f"Successful: {len(successful)}")

# Cleanup
try:
    os.remove("scraper.loom")
except PermissionError:
    pass  # File still in use on Windows
