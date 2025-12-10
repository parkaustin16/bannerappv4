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
    
    Supports multiple banner structures:
    1. LG.com structure: div.cmp-carousel__item with picture > source
    2. Generic carousel: Any div with picture > source tags
    3. Picture tags: Any responsive image with media queries
    
    Args:
        html: Complete HTML content of the page
        base_url: Base URL for resolving relative image paths
    
    Returns:
        List of BannerData dictionaries
    """
    banners = []
    
    # Try multiple extraction strategies
    
    # Strategy 1: LG.com carousel items (original)
    carousel_item_pattern = r'<div[^>]*class="[^"]*cmp-carousel__item[^"]*"[^>]*>(.*?)</div>'
    carousel_matches = list(re.finditer(carousel_item_pattern, html, re.IGNORECASE | re.DOTALL))
    
    # Strategy 2: Generic carousel items with data-title
    if not carousel_matches:
        carousel_item_pattern = r'<div[^>]*data-title=["\']([^"\']*)["\'][^>]*>(.*?)</div>'
        carousel_matches = list(re.finditer(carousel_item_pattern, html, re.IGNORECASE | re.DOTALL))
    
    # Strategy 3: Any picture tags with media queries
    if not carousel_matches:
        picture_pattern = r'<picture[^>]*>(.*?)</picture>'
        picture_matches = list(re.finditer(picture_pattern, html, re.IGNORECASE | re.DOTALL))
        if picture_matches:
            for idx, picture_match in enumerate(picture_matches):
                picture_html = picture_match.group(0)
                source_pattern = r'<source[^>]*media="([^"]*)"[^>]*srcset="([^"]*)"[^>]*>'
                source_matches = list(re.finditer(source_pattern, picture_html, re.IGNORECASE))
                
                if source_matches:
                    image_desktop = ""
                    image_mobile = ""
                    
                    for source_match in source_matches:
                        media = source_match.group(1)
                        srcset = source_match.group(2)
                        
                        if "min-width" in media or "desktop" in media.lower():
                            image_desktop = resolve_url(srcset, base_url)
                        elif "max-width" in media or "mobile" in media.lower():
                            image_mobile = resolve_url(srcset, base_url)
                    
                    # If we found at least one image, add it as a banner
                    if image_desktop or image_mobile:
                        banners.append(BannerData(
                            dataTitle=f"Banner {idx + 1}",
                            bannerHtml=picture_html,
                            imageDesktop=image_desktop,
                            imageMobile=image_mobile
                        ))
            
            if banners:
                return banners
    
    # Process carousel matches using original logic
    for item_match in carousel_matches:
        if len(item_match.groups()) >= 1:
            # Check if this is the data-title pattern
            if "data-title=" in carousel_item_pattern:
                data_title = item_match.group(1)
                item_html = item_match.group(2)
                # Reconstruct full HTML
                full_match = item_match.group(0)
            else:
                item_html = item_match.group(1)
                full_match = item_match.group(0)
                
                # Extract data-title attribute if present
                title_match = re.search(r'data-title=["\']([^"\']*)["\']', full_match, re.IGNORECASE)
                data_title = title_match.group(1) if title_match else ""
        else:
            continue
        
        # Check for cmp-container or picture tags (banner content wrapper)
        if "cmp-container" not in full_match and "picture" not in full_match.lower():
            continue
        
        # Initialize image URLs
        image_desktop = ""
        image_mobile = ""
        
        # Extract picture element and its source tags
        picture_match = re.search(
            r'<picture[^>]*>(.*?)</picture>',
            full_match,
            re.IGNORECASE | re.DOTALL
        )
        
        if picture_match:
            picture_html = picture_match.group(1)
            
            # Find all source tags with media queries
            source_pattern = r'<source[^>]*media="([^"]*)"[^>]*srcset="([^"]*)"[^>]*>'
            source_matches = list(re.finditer(source_pattern, picture_html, re.IGNORECASE))
            
            for source_match in source_matches:
                media = source_match.group(1)
                srcset = source_match.group(2)
                
                if "min-width: 769px" in media or "min-width" in media:
                    if not image_desktop:  # Only set if not already set
                        image_desktop = resolve_url(srcset, base_url)
                elif "max-width: 768px" in media or "max-width" in media:
                    if not image_mobile:  # Only set if not already set
                        image_mobile = resolve_url(srcset, base_url)
        
        # Only add if we found at least one image or have an explicit title
        if (image_desktop or image_mobile) or data_title:
            banners.append(BannerData(
                dataTitle=data_title or f"Banner {len(banners) + 1}",
                bannerHtml=full_match,
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
    
    Supports multiple HTML structures for flexibility across different sites.
    
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
    
    # Debug: Count potential picture tags
    picture_count = len(re.findall(r'<picture[^>]*>', html, re.IGNORECASE))
    print(f"  - Found {picture_count} picture tags in HTML")
    
    # Debug: Count carousel items
    carousel_count = len(re.findall(r'<div[^>]*class="[^"]*cmp-carousel__item[^"]*"', html, re.IGNORECASE))
    print(f"  - Found {carousel_count} carousel items in HTML")
    
    # Extract banners from DOM structure
    banners = extract_banner_html_and_images(html, url)
    print(f"✓ Found {len(banners)} banners")
    
    if banners:
        for i, banner in enumerate(banners, 1):
            has_desktop = "Yes" if banner['imageDesktop'] else "No"
            has_mobile = "Yes" if banner['imageMobile'] else "No"
            print(f"  - Banner {i}: {banner['dataTitle']} (Desktop: {has_desktop}, Mobile: {has_mobile})")
    
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
