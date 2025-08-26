import json
import os
import io
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image, ImageDraw
import streamlit as st
import pandas as pd
from datetime import datetime
import base64

# --- File Operations ---
def load_json_cached(file_path: str, default_value: Any) -> Any:
    """Load JSON file with caching to reduce I/O operations."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                content = f.read().strip()
                if not content:  # Handle empty files
                    return default_value
                return json.loads(content)
        except (json.JSONDecodeError, IOError) as e:
            st.error(f"Error loading {file_path}: {e}")
            # Try to backup and recreate the file if it's corrupted
            try:
                backup_path = f"{file_path}.backup"
                if os.path.exists(file_path):
                    os.rename(file_path, backup_path)
                return default_value
            except:
                return default_value
    return default_value

def save_json(file_path: str, data: Any) -> bool:
    """Save JSON file with error handling."""
    try:
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except IOError as e:
        st.error(f"Error saving {file_path}: {e}")
        return False

def get_image_hash(img: Image.Image) -> str:
    """Generate a hash for the image to use as cache key."""
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return hashlib.md5(img_bytes.getvalue()).hexdigest()

# --- Image Processing ---
def validate_image_aspect_ratio(img: Image.Image, target_ratio: float = 8/3, tolerance: float = 0.01) -> Tuple[bool, float]:
    """Validate image aspect ratio."""
    w, h = img.size
    aspect_ratio = w / h
    is_valid = abs(aspect_ratio - target_ratio) <= tolerance
    return is_valid, aspect_ratio

def resize_image_for_display(img: Image.Image, max_width: int = 800) -> Image.Image:
    """Resize image for display while maintaining aspect ratio."""
    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img

def create_zone_preview_image(img: Image.Image, zones: List[Dict], ignore_zones: List[Dict]) -> Image.Image:
    """Create a preview image with zones drawn."""
    preview_img = img.copy()
    draw = ImageDraw.Draw(preview_img)
    w, h = img.size
    
    # Draw text zones (green)
    for zone in zones:
        name = zone.get("name", "Unnamed")
        nx, ny, nw, nh = zone.get("zone", (0, 0, 0, 0))
        x, y, w_abs, h_abs = int(nx * w), int(ny * h), int(nw * w), int(nh * h)
        draw.rectangle([x, y, x + w_abs, y + h_abs], outline="green", width=2)
        draw.text((x + 5, y + 5), name, fill="green")
    
    # Draw ignore zones (blue)
    for zone in ignore_zones:
        name = zone.get("name", "Unnamed")
        nx, ny, nw, nh = zone.get("zone", (0, 0, 0, 0))
        x, y, w_abs, h_abs = int(nx * w), int(ny * h), int(nw * w), int(nh * h)
        draw.rectangle([x, y, x + w_abs, y + h_abs], outline="blue", width=2)
        draw.text((x + 5, y + 5), name, fill="blue")
    
    return preview_img

# --- Geometry Utilities ---
def overlap_ratio(text_box: Tuple[int, int, int, int], zone_box: Tuple[int, int, int, int]) -> float:
    """Calculate overlap ratio between text box and zone box."""
    tx, ty, tw, th = text_box
    zx, zy, zw, zh = zone_box
    
    inter_x1 = max(tx, zx)
    inter_y1 = max(ty, zy)
    inter_x2 = min(tx + tw, zx + zw)
    inter_y2 = min(ty + th, zy + zh)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    text_area = tw * th
    return inter_area / text_area if text_area > 0 else 0.0

def convert_ocr_bbox_to_rect(bbox) -> Tuple[int, int, int, int]:
    """Convert OCR bounding box to rectangle format."""
    xs = [int(p[0]) for p in bbox]
    ys = [int(p[1]) for p in bbox]
    tx, ty, tw, th = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    return tx, ty, tw, th

def normalize_coordinates(x: float, y: float, w: float, h: float, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """Convert absolute coordinates to normalized (0-1) coordinates."""
    return x / img_width, y / img_height, w / img_width, h / img_height

def denormalize_coordinates(nx: float, ny: float, nw: float, nh: float, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """Convert normalized (0-1) coordinates to absolute coordinates."""
    return int(nx * img_width), int(ny * img_height), int(nw * img_width), int(nh * img_height)

# --- Export Functions ---
def generate_csv_report(results: List[Dict]) -> str:
    """Generate CSV report from batch processing results."""
    df_data = []
    for result in results:
        df_data.append({
            'Filename': result.get('filename', 'Unknown'),
            'Score': result.get('score', 0),
            'Infractions': len(result.get('penalties', [])),
            'Aspect Ratio Valid': result.get('aspect_ratio_valid', False),
            'Processing Time': result.get('processing_time', 0),
            'Timestamp': result.get('timestamp', '')
        })
    
    df = pd.DataFrame(df_data)
    csv = df.to_csv(index=False)
    return csv

def generate_pdf_report(results: List[Dict]) -> bytes:
    """Generate PDF report from batch processing results."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph("Banner QA Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Summary
        total_images = len(results)
        avg_score = sum(r.get('score', 0) for r in results) / total_images if total_images > 0 else 0
        total_infractions = sum(len(r.get('penalties', [])) for r in results)
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Images Processed', str(total_images)],
            ['Average Score', f"{avg_score:.1f}%"],
            ['Total Infractions', str(total_infractions)],
            ['Report Generated', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 12))
        
        # Detailed results
        if results:
            detail_data = [['Filename', 'Score', 'Infractions', 'Aspect Ratio']]
            for result in results:
                detail_data.append([
                    result.get('filename', 'Unknown'),
                    f"{result.get('score', 0)}%",
                    str(len(result.get('penalties', []))),
                    '✓' if result.get('aspect_ratio_valid', False) else '✗'
                ])
            
            detail_table = Table(detail_data)
            detail_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(detail_table)
        
        doc.build(elements)
        return buffer.getvalue()
    except ImportError:
        st.error("ReportLab not installed. Install with: pip install reportlab")
        return b""

def get_download_link(data: bytes, filename: str, text: str) -> str:
    """Generate a download link for files."""
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:file/{filename.split(".")[-1]};base64,{b64}" download="{filename}">{text}</a>'

# --- Template Management ---
def load_templates() -> Dict[str, Dict]:
    """Load predefined templates."""
    templates = {
        "Standard Banner": {
            "description": "Standard 8:3 banner with common text zones",
            "text_zones": [
                {"name": "Headline", "zone": (0.125, 0.1458, 0.3047, 0.1458)},
                {"name": "Body", "zone": (0.125, 0.3027, 0.3047, 0.05)},
                {"name": "Eyebrow", "zone": (0.125, 0.1042, 0.3047, 0.05)},
                {"name": "CTA", "zone": (0.125, 0.375, 0.0676, 0.083)}
            ],
            "ignore_zones": [
                {"name": "Disclaimer", "zone": (0.1149, 0.8958, 0.8041, 0.1959)},
                {"name": "Icons", "zone": (0.125, 0.6875, 0.3097, 0.0875)}
            ],
            "ignore_terms": ["2024", "ai", "lg", "miniled", "no", "oled", "oled tv", "por 12 anos", "world"]
        },
        "Minimal Banner": {
            "description": "Minimal banner with just headline and CTA",
            "text_zones": [
                {"name": "Headline", "zone": (0.1, 0.2, 0.8, 0.3)},
                {"name": "CTA", "zone": (0.1, 0.6, 0.3, 0.1)}
            ],
            "ignore_zones": [],
            "ignore_terms": []
        },
        "Product Banner": {
            "description": "Product-focused banner with multiple text areas",
            "text_zones": [
                {"name": "Product Name", "zone": (0.1, 0.1, 0.4, 0.15)},
                {"name": "Price", "zone": (0.1, 0.3, 0.2, 0.1)},
                {"name": "Features", "zone": (0.1, 0.45, 0.6, 0.2)},
                {"name": "CTA", "zone": (0.1, 0.7, 0.25, 0.1)}
            ],
            "ignore_zones": [
                {"name": "Legal", "zone": (0.05, 0.85, 0.9, 0.15)}
            ],
            "ignore_terms": ["limited time", "offer", "sale", "discount"]
        }
    }
    return templates

def apply_template(template_name: str) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Apply a template configuration."""
    templates = load_templates()
    if template_name in templates:
        template = templates[template_name]
        return (
            template.get("text_zones", []),
            template.get("ignore_zones", []),
            template.get("ignore_terms", [])
        )
    return [], [], []

# --- Analytics ---
def save_analytics_data(result: Dict):
    """Save analytics data for tracking performance over time."""
    analytics_file = "analytics.json"
    analytics = load_json_cached(analytics_file, [])
    
    analytics_entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": result.get("filename", "Unknown"),
        "score": result.get("score", 0),
        "infractions": len(result.get("penalties", [])),
        "aspect_ratio_valid": result.get("aspect_ratio_valid", False),
        "processing_time": result.get("processing_time", 0),
        "image_size": result.get("image_size", (0, 0)),
        "zones_used": result.get("zones_used", 0)
    }
    
    analytics.append(analytics_entry)
    
    # Keep only last 1000 entries to prevent file from growing too large
    if len(analytics) > 1000:
        analytics = analytics[-1000:]
    
    save_json(analytics_file, analytics)

def get_analytics_summary() -> Dict:
    """Get analytics summary for dashboard."""
    analytics_file = "analytics.json"
    analytics = load_json_cached(analytics_file, [])
    
    if not analytics:
        return {
            "total_processed": 0,
            "avg_score": 0,
            "total_infractions": 0,
            "success_rate": 0,
            "avg_processing_time": 0
        }
    
    total_processed = len(analytics)
    avg_score = sum(entry.get("score", 0) for entry in analytics) / total_processed
    total_infractions = sum(entry.get("infractions", 0) for entry in analytics)  # Use infractions field instead of penalties
    success_rate = sum(1 for entry in analytics if entry.get("score", 0) == 100) / total_processed * 100
    avg_processing_time = sum(entry.get("processing_time", 0) for entry in analytics) / total_processed
    
    return {
        "total_processed": total_processed,
        "avg_score": round(avg_score, 1),
        "total_infractions": total_infractions,
        "success_rate": round(success_rate, 1),
        "avg_processing_time": round(avg_processing_time, 2)
    }
