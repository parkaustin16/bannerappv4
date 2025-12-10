"""
Banner Capture Method from LG.com Subsidiary Sites
This module implements the exact method used by the banner-inspection app
to capture web and mobile banners from LG subsidiary sites.
"""

import re
import asyncio
from typing import TypedDict, List, Optional
from urllib.parse import urljoin, urlparse
import aiohttp
from html.parser import HTMLParser

# Export main function
__all__ = ['crawl_url', 'BannerData', 'CrawlResult']


class BannerData(TypedDict):
    """Represents a single captured banner with its metadata and images."""
    dataTitle: str
    bannerHtml: str
    imageDesktop: str
    imageMobile: str


class CrawlResult(TypedDict):
    """Result of crawling a website for banners."""
    banners: List[BannerData]
    css: str


class HTMLPictureParser(HTMLParser):
    """Parser to extract source elements from picture tags."""
    
    def __init__(self):
        super().__init__()
        self.in_picture = False
        self.sources = []
    
    def handle_starttag(self, tag: str, attrs: List[tuple]):
        if tag == "picture":
            self.in_picture = True
        elif tag == "source" and self.in_picture:
            self.sources.append(dict(attrs))
    
    def handle_endtag(self, tag: str):
        if tag == "picture":
            self.in_picture = False


def resolve_url(url: str, base_url: str) -> str:
    """
    Resolve relative URLs to absolute URLs.
    
    Args:
        url: The URL to resolve (can be relative or absolute)
        base_url: The base URL for resolution
    
    Returns:
        Absolute URL as string
    """
    if not url:
        return ""
    
    # Already absolute
    if url.startswith("http://") or url.startswith("https://"):
        return url
    
    parsed_base = urlparse(base_url)
    
    # Protocol-relative URL
    if url.startswith("//"):
        return f"{parsed_base.scheme}:{url}"
    
    # Root-relative URL
    if url.startswith("/"):
        return f"{parsed_base.scheme}://{parsed_base.netloc}{url}"
    
    # Relative URL
    return urljoin(base_url, url)


def extract_css_urls(html: str, base_url: str) -> List[str]:
    """
    Extract all stylesheet URLs from HTML.
    
    This extracts CSS URLs from:
    1. <link rel="stylesheet" href="...">
    2. <link href="..." rel="stylesheet">
    
    Args:
        html: HTML content as string
        base_url: Base URL for resolving relative paths
    
    Returns:
        List of absolute CSS URLs
    """
    css_urls = []
    
    # Pattern 1: rel="stylesheet" comes before href
    link_regex_1 = r'<link[^>]*rel=["\']stylesheet["\'][^>]*href=["\']([^"\']+)["\'][^>]*>'
    
    # Pattern 2: href comes before rel="stylesheet"
    link_regex_2 = r'<link[^>]*href=["\']([^"\']+)["\'][^>]*rel=["\']stylesheet["\'][^>]*>'
    
    for pattern in [link_regex_1, link_regex_2]:
        for match in re.finditer(pattern, html, re.IGNORECASE):
            url = match.group(1)
            resolved = resolve_url(url, base_url)
            if resolved and resolved not in css_urls:
                css_urls.append(resolved)
    
    return css_urls


def extract_inline_styles(html: str) -> str:
    """
    Extract all inline <style> tags from HTML.
    
    Args:
        html: HTML content as string
    
    Returns:
        Combined inline CSS content
    """
    inline_styles = []
    
    # Find all <style>...</style> blocks
    style_regex = r'<style[^>]*>([\s\S]*?)<\/style>'
    
    for match in re.finditer(style_regex, html, re.IGNORECASE):
        content = match.group(1).strip()
        if content:
            inline_styles.append(content)
    
    if not inline_styles:
        return ""
    
    return "/* Inline Styles */\n" + "\n\n".join(inline_styles)


async def collect_css(css_urls: List[str], timeout: int = 10) -> str:
    """
    Fetch and combine all CSS files.
    
    Args:
        css_urls: List of CSS file URLs to fetch
        timeout: Timeout in seconds for each request (default: 10)
    
    Returns:
        Combined CSS content from all successfully fetched files
    """
    css_contents = []
    max_retries = 2
    
    async with aiohttp.ClientSession() as session:
        for url in css_urls:
            retries = 0
            success = False
            
            while retries < max_retries and not success:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                    
                    async with session.get(url, headers=headers, timeout=timeout) as response:
                        if response.status == 200:
                            css = await response.text()
                            css_contents.append(f"/* Source: {url} */\n{css}\n")
                            success = True
                            print(f"✓ Successfully fetched CSS from {url}")
                        else:
                            print(f"✗ Failed to fetch CSS from {url}: {response.status} {response.reason}")
                            retries += 1
                
                except asyncio.TimeoutError:
                    print(f"✗ Timeout fetching CSS from {url}")
                    retries += 1
                except Exception as error:
                    print(f"✗ Error fetching CSS from {url}: {str(error)}")
                    retries += 1
                    if retries < max_retries:
                        await asyncio.sleep(1)
    
    if not css_contents:
        print("⚠ No CSS files were successfully collected")
        return ""
    
    print(f"✓ Successfully collected {len(css_contents)} CSS files from {len(css_urls)} URLs")
    return "\n".join(css_contents)


def extract_banner_html_and_images(html: str, base_url: str) -> List[BannerData]:
    """
    Extract banner HTML and associated images from carousel items.
    
    LG.com structure uses:
    - div.cmp-carousel__item: Container for each banner
    - data-title attribute: Banner identifier
    - div.cmp-container: Banner content wrapper
    - picture > source: Responsive images with media queries
    
    For each banner, extracts:
    - Desktop image: source with media="(min-width: 769px)"
    - Mobile image: source with media="(max-width: 768px)"
    
    Args:
        html: Complete HTML content of the page
        base_url: Base URL for resolving relative image paths
    
    Returns:
        List of BannerData dictionaries
    """
    banners = []
    
    # Find all carousel items
    carousel_item_pattern = r'<div[^>]*class="cmp-carousel__item"[^>]*>(.*?)</div>'
    
    for item_match in re.finditer(carousel_item_pattern, html, re.IGNORECASE | re.DOTALL):
        item_html = item_match.group(0)
        
        # Extract data-title attribute
        title_match = re.search(r'data-title=["\']([^"\']*)["\']', item_html, re.IGNORECASE)
        data_title = title_match.group(1) if title_match else ""
        
        # Check for cmp-container (banner content wrapper)
        if "cmp-container" not in item_html:
            continue
        
        # Initialize image URLs
        image_desktop = ""
        image_mobile = ""
        
        # Extract picture element and its source tags
        picture_match = re.search(
            r'<picture[^>]*>(.*?)</picture>',
            item_html,
            re.IGNORECASE | re.DOTALL
        )
        
        if picture_match:
            picture_html = picture_match.group(1)
            
            # Find all source tags with media queries
            source_pattern = r'<source[^>]*media="([^"]*)"[^>]*srcset="([^"]*)"[^>]*>'
            
            for source_match in re.finditer(source_pattern, picture_html, re.IGNORECASE):
                media = source_match.group(1)
                srcset = source_match.group(2)
                
                if "min-width: 769px" in media:
                    image_desktop = resolve_url(srcset, base_url)
                elif "max-width: 768px" in media:
                    image_mobile = resolve_url(srcset, base_url)
        
        banners.append(BannerData(
            dataTitle=data_title,
            bannerHtml=item_html,
            imageDesktop=image_desktop,
            imageMobile=image_mobile
        ))
    
    return banners


async def crawl_url(url: str) -> CrawlResult:
    """
    Main banner crawling method - the core of the banner capture process.
    
    This method:
    1. Fetches the webpage HTML
    2. Parses the DOM to find carousel banner items
    3. Extracts banner HTML, desktop images, and mobile images
    4. Collects all CSS (external stylesheets and inline styles)
    
    The LG.com site structure uses:
    - Carousel items with responsive image sources
    - Separate desktop/mobile breakpoints (769px threshold)
    - External CSS files and inline styles
    
    Args:
        url: The website URL to crawl
    
    Returns:
        CrawlResult containing:
        - banners: List of extracted BannerData objects
        - css: Combined CSS content (external + inline)
    
    Raises:
        Exception: If fetching or parsing fails
    """
    print(f"Starting banner capture from: {url}")
    
    async with aiohttp.ClientSession() as session:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        async with session.get(url, headers=headers) as response:
            if not response.ok:
                raise Exception(f"Failed to fetch {url}: {response.status} {response.reason}")
            
            html = await response.text()
    
    print(f"✓ Fetched HTML ({len(html)} bytes)")
    
    # Extract banners from DOM structure
    banners = extract_banner_html_and_images(html, url)
    print(f"✓ Found {len(banners)} banners")
    
    # Extract CSS URLs
    css_urls = extract_css_urls(html, url)
    print(f"✓ Found {len(css_urls)} external CSS files")
    
    # Collect external CSS
    external_css = await collect_css(css_urls)
    
    # Extract inline styles
    inline_css = extract_inline_styles(html)
    if inline_css:
        print(f"✓ Found inline styles")
    
    # Combine all CSS
    css = "\n\n".join([c for c in [external_css, inline_css] if c])
    
    result = CrawlResult(
        banners=banners,
        css=css
    )
    
    print(f"\n✓ Banner capture complete!")
    print(f"  - Banners: {len(result['banners'])}")
    print(f"  - CSS size: {len(result['css'])} bytes")
    
    return result


# Example usage
if __name__ == "__main__":
    # Example: Crawl an LG subsidiary site
    example_url = "https://www.lg.com"  # Replace with actual LG subsidiary URL
    
    result = asyncio.run(crawl_url(example_url))
    
    print("\n" + "="*60)
    print("CRAWL RESULTS")
    print("="*60)
    
    for i, banner in enumerate(result['banners'], 1):
        print(f"\nBanner {i}: {banner['dataTitle']}")
        print(f"  Desktop Image: {banner['imageDesktop']}")
        print(f"  Mobile Image:  {banner['imageMobile']}")
        print(f"  HTML Size:     {len(banner['bannerHtml'])} bytes")
    
    print(f"\nTotal CSS: {len(result['css'])} bytes")
    print(f"CSS Preview: {result['css'][:200]}...")
