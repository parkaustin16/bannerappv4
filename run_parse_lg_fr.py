import asyncio
import aiohttp
import traceback
import re
from capturev2 import extract_banner_html_and_images

URL = 'https://lg.com/fr'
LOG_PATH = 'lg_debug.log'

async def fetch_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, timeout=30) as resp:
            resp.raise_for_status()
            return await resp.text()

async def main():
    with open(LOG_PATH, 'w', encoding='utf-8') as lf:
        lf.write(f"RUN LOG for {URL}\n")
        lf.write('='*80 + '\n')
        try:
            lf.write('Fetching HTML...\n')
            html = await fetch_html(URL)
            lf.write(f'Fetched HTML length: {len(html)} bytes\n')
            # Save a copy of the HTML for offline inspection
            with open('lg_fr.html', 'w', encoding='utf-8') as hf:
                hf.write(html)
            lf.write('Wrote lg_fr.html\n')

            # Basic counts
            pictures = len(re.findall(r'<picture[^>]*>', html, re.IGNORECASE))
            carousels = len(re.findall(r'<div[^>]*class="[^\"]*cmp-carousel__item[^\"]*"', html, re.IGNORECASE))
            imgs = re.findall(r'<img[^>]*>', html, re.IGNORECASE)
            lf.write(f'picture tags: {pictures}\n')
            lf.write(f'cmp-carousel__item divs: {carousels}\n')
            lf.write(f'<img> tags: {len(imgs)} (showing up to 50)\n')

            for i, tag in enumerate(imgs[:50], 1):
                src = re.search(r'src=["\']([^"\']+)["\']', tag, re.IGNORECASE)
                srcset = re.search(r'srcset=["\']([^"\']+)["\']', tag, re.IGNORECASE)
                lf.write(f'  {i}. src={src.group(1) if src else ""} srcset={srcset.group(1) if srcset else ""}\n')

            lf.write('\nRunning extract_banner_html_and_images() with debug=False (parser output):\n')
            banners = extract_banner_html_and_images(html, URL, debug=False)
            lf.write(f'Parser returned {len(banners)} banners\n')
            for i, b in enumerate(banners, 1):
                lf.write(f'--- Banner {i} ---\n')
                lf.write(f"title: {b.get('dataTitle')}\n")
                lf.write(f"imageDesktop: {b.get('imageDesktop')}\n")
                lf.write(f"imageMobile: {b.get('imageMobile')}\n")
                lf.write(f"html_len: {len(b.get('bannerHtml',''))}\n")

        except Exception as e:
            lf.write('EXCEPTION\n')
            lf.write(traceback.format_exc())
            print('Wrote debug log to', LOG_PATH)
            return

        lf.write('\nComplete.\n')
    print('Wrote debug log to', LOG_PATH, 'and lg_fr.html')

if __name__ == '__main__':
    asyncio.run(main())
