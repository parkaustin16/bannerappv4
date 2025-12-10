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

try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

# Export main function
__all__ = ['crawl_url', 'BannerData', 'CrawlResult']

# Target aspect ratio for banners (8:3 = 2.667)
TARGET_ASPECT_RATIO = 8 / 3
ASPECT_RATIO_TOLERANCE = 0.1  # Allow 10% tolerance

def is_valid_aspect_ratio(width: int, height: int) -> bool:
    """
    Check if banner dimensions match the target aspect ratio (8:3).
    
    Args:
        width: Banner width in pixels
        height: Banner height in pixels
    
    Returns:
        True if aspect ratio is within tolerance of 8:3
    """
    if width <= 0 or height <= 0:
        return False
    
    aspect_ratio = width / height
    min_ratio = TARGET_ASPECT_RATIO * (1 - ASPECT_RATIO_TOLERANCE)
    max_ratio = TARGET_ASPECT_RATIO * (1 + ASPECT_RATIO_TOLERANCE)
    
    return min_ratio <= aspect_ratio <= max_ratio


class BannerData(TypedDict):
    """Represents a single captured banner with dimensions and screenshot."""
    dataTitle: str
    bannerHtml: str
    imageDesktop: str
    imageMobile: str
    extractedText: dict
    width: int  # Banner width in pixels
    height: int  # Banner height in pixels
    screenshotBytes: Optional[bytes]  # Screenshot of banner area


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


def extract_text_from_banner(html: str) -> dict:
    """
    Extract text from banner HTML using regex patterns.
    Mimics textinspect.extract_text_from_html but works standalone.
    
    Args:
        html: Banner HTML code
    
    Returns:
        Dictionary with eyebrow, head_copy, body_copy
    """
    eyebrow = ""
    head_copy = ""
    body_copy = ""
    
    # Extract eyebrow text (class containing 'eyebrow')
    eyebrow_pattern = r'<[^>]*class="[^"]*eyebrow[^"]*"[^>]*>([^<]*)</[^>]*>'
    eyebrow_match = re.search(eyebrow_pattern, html, re.IGNORECASE)
    if eyebrow_match:
        eyebrow = eyebrow_match.group(1).strip()
    
    # Extract head copy (h1, h2, or class containing 'head')
    head_patterns = [
        r'<h[1-2][^>]*>([^<]*)</h[1-2]>',
        r'<[^>]*class="[^"]*head[^"]*"[^>]*>([^<]*)</[^>]*>'
    ]
    for pattern in head_patterns:
        head_match = re.search(pattern, html, re.IGNORECASE)
        if head_match:
            head_copy = head_match.group(1).strip()
            break
    
    # Extract body copy (p, div, or class containing 'body')
    body_patterns = [
        r'<p[^>]*>([^<]*)</p>',
        r'<[^>]*class="[^"]*body[^"]*"[^>]*>([^<]*)</[^>]*>'
    ]
    for pattern in body_patterns:
        body_match = re.search(pattern, html, re.IGNORECASE)
        if body_match:
            body_copy = body_match.group(1).strip()
            break
    
    return {
        "eyebrow": eyebrow,
        "head_copy": head_copy,
        "body_copy": body_copy
    }


async def capture_banner_screenshot(url: str, banner_selector: str = ".cmp-carousel__item", banner_index: int = 0) -> tuple:
    """
    Capture a screenshot of a specific banner element with aspect ratio validation.
    
    Only captures desktop banners with aspect ratio of 8:3 (±10%).
    Uses Playwright to render the page and capture the banner area including all text.
    
    Args:
        url: The website URL
        banner_selector: CSS selector for the banner element
        banner_index: Index of the banner to capture (0 for first, 1 for second, etc.)
    
    Returns:
        Tuple of (width, height, screenshot_bytes, valid) or (0, 0, None, False) if capture fails
    """
    if not HAS_PLAYWRIGHT:
        print("⚠ Playwright not installed. Skipping screenshot capture.")
        return (0, 0, None, False)
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Set user agent
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            
            try:
                # Navigate to URL with timeout
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Wait for banner selector to be visible
                try:
                    await page.wait_for_selector(banner_selector, timeout=5000)
                except:
                    print(f"⚠ Banner selector '{banner_selector}' not found")
                
                # Get all banner elements matching selector
                banner_elements = await page.query_selector_all(banner_selector)
                if not banner_elements or banner_index >= len(banner_elements):
                    print(f"⚠ Banner index {banner_index} not found (total banners: {len(banner_elements)})")
                    await browser.close()
                    return (0, 0, None, False)
                
                # Get the specific banner element at the given index
                banner_element = banner_elements[banner_index]
                
                # Get banner bounding box
                bbox = await banner_element.bounding_box()
                if not bbox:
                    print("⚠ Could not get banner bounding box")
                    await browser.close()
                    return (0, 0, None, False)
                
                width = int(bbox['width'])
                height = int(bbox['height'])
                
                # Validate aspect ratio (8:3 desktop banner)
                valid_aspect = is_valid_aspect_ratio(width, height)
                aspect_ratio = width / height if height > 0 else 0
                
                if not valid_aspect:
                    print(f"✗ Invalid aspect ratio: {aspect_ratio:.2f} (target: {TARGET_ASPECT_RATIO:.2f} ±{ASPECT_RATIO_TOLERANCE*100:.0f}%)")
                    await browser.close()
                    return (width, height, None, False)
                
                print(f"✓ Banner dimensions: {width}x{height}px (aspect ratio: {aspect_ratio:.2f})")
                
                # Set viewport to match banner dimensions (with some padding)
                await page.set_viewport_size({"width": width + 20, "height": height + 20})
                
                # Scroll element into view and take screenshot
                await banner_element.scroll_into_view_if_needed()
                screenshot = await banner_element.screenshot()
                
                print(f"✓ Captured banner screenshot ({len(screenshot)} bytes)")
                
                await browser.close()
                return (width, height, screenshot, True)
                
            except Exception as e:
                print(f"✗ Error capturing screenshot: {e}")
                await browser.close()
                return (0, 0, None, False)
                
    except Exception as e:
        print(f"✗ Error launching browser: {e}")
        return (0, 0, None, False)


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
                            imageMobile=image_mobile,
                            extractedText={},
                            width=0,
                            height=0,
                            screenshotBytes=None
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
            # Extract text from banner HTML
            extracted_text = extract_text_from_banner(full_match)
            
            banners.append(BannerData(
                dataTitle=data_title or f"Banner {len(banners) + 1}",
                bannerHtml=full_match,
                imageDesktop=image_desktop,
                imageMobile=image_mobile,
                extractedText=extracted_text,
                width=0,
                height=0,
                screenshotBytes=None
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
        
        # Capture screenshots of banners with their actual dimensions
        print(f"\n📸 Capturing banner screenshots...")
        valid_banners = []
        
        for i, banner in enumerate(banners):
            try:
                # Pass the banner index to capture the correct banner on the page
                width, height, screenshot, valid = await capture_banner_screenshot(url, ".cmp-carousel__item", banner_index=i)
                
                if not valid:
                    print(f"  ⊘ Skipped {banner['dataTitle']}: invalid aspect ratio or dimensions")
                    continue
                
                if screenshot:
                    updated_banner = BannerData(
                        dataTitle=banner['dataTitle'],
                        bannerHtml=banner['bannerHtml'],
                        imageDesktop=banner['imageDesktop'],
                        imageMobile=banner['imageMobile'],
                        extractedText=banner['extractedText'],
                        width=width,
                        height=height,
                        screenshotBytes=screenshot
                    )
                    valid_banners.append(updated_banner)
                    print(f"  ✓ Captured {banner['dataTitle']} ({width}x{height}px)")
            except Exception as e:
                print(f"  ✗ Failed to capture {banner['dataTitle']}: {e}")
        
        # Replace banners list with only valid ones
        banners = valid_banners
        print(f"\n✓ {len(banners)} banners passed aspect ratio validation")
    
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
