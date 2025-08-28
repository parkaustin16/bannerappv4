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
from reportlab.lib.utils import ImageReader


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
def validate_image_aspect_ratio(img: Image.Image, target_ratio: float = 8 / 3, tolerance: float = 0.01) -> Tuple[
    bool, float]:
    """Validate image aspect ratio."""
    w, h = img.size
    aspect_ratio = w / h
    is_valid = abs(aspect_ratio - target_ratio) <= tolerance
    return is_valid, aspect_ratio


def resize_image_for_display(img: Image.Image, max_width: int = 800) -> bytes:
    """Resize image for display while maintaining aspect ratio."""
    img_bytes = io.BytesIO()
    try:
        # Calculate new dimensions
        w, h = img.size
        if w > max_width:
            new_w = max_width
            new_h = int(h * (max_width / w))
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            img_resized = img
        
        # Save to bytes
        img_resized.save(img_bytes, format="PNG", optimize=True)
        return img_bytes.getvalue()
    finally:
        img_bytes.close()


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


def normalize_coordinates(x: float, y: float, w: float, h: float, img_width: int, img_height: int) -> Tuple[
    float, float, float, float]:
    """Convert absolute coordinates to normalized (0-1) coordinates."""
    return x / img_width, y / img_height, w / img_width, h / img_height


def denormalize_coordinates(nx: float, ny: float, nw: float, nh: float, img_width: int, img_height: int) -> Tuple[
    int, int, int, int]:
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


def generate_pdf_report(results: List[Dict]) -> Optional[bytes]:
    """Generate a PDF report for the QA results."""
    try:
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import (
            SimpleDocTemplate,
            Table,
            TableStyle,
            Paragraph,
            Spacer,
            Image as RLImage,
            PageBreak,
            Flowable,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        buffer = io.BytesIO()
        # Add today's date to the page title
        today = datetime.now().strftime("%B %d, %Y")
        doc = SimpleDocTemplate(
            buffer,
            pagesize=landscape(letter),
            leftMargin=36,
            rightMargin=36,
            top=28,
            bottom=28,
            title=f"PDF Report - {today}"
        )
        elements = []
        styles = getSampleStyleSheet()

        # Custom styles
        tag_style = ParagraphStyle(
            name='Tag', parent=styles['BodyText'], textColor=colors.white, alignment=0,
            fontSize=10, leading=12
        )
        rule_style = ParagraphStyle(
            name='Rule', parent=styles['BodyText'], textColor=colors.white, fontSize=9, leading=12
        )
        small_style = ParagraphStyle(
            name='Small', parent=styles['BodyText'], fontSize=9, leading=11
        )

        # Calculate summary statistics
        total_images = len(results)
        avg_score = sum(r.get('score', 0) for r in results) / total_images if total_images > 0 else 0
        total_infractions = sum(len(r.get('penalties', [])) for r in results)
        success_count = sum(1 for r in results if r.get('score', 0) == 100)
        avg_processing_time = sum(r.get('processing_time', 0) for r in results) / total_images if total_images > 0 else 0
        
        # Intro page: Color Key with disclaimer
        key_data = [
            ['Green', 'Defined text zone or allowed text (valid - inside zone)'],
            ['Blue', 'Defined ignored zones or ignored text'],
            ['Red', 'Unallowed text (infraction - outside zone)'],
        ]
        key_table = Table(key_data, colWidths=[120, 500])
        key_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('TEXTCOLOR', (0, 0), (0, 0), colors.green),
            ('TEXTCOLOR', (0, 1), (0, 1), colors.blue),
            ('TEXTCOLOR', (0, 2), (0, 2), colors.red),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
        ]))
        
        # Summary statistics table
        summary_data = [
            ['Total Images', str(total_images)],
            ['Average Score', f"{avg_score:.1f}%"],
            ['Total Infractions', str(total_infractions)],
            ['Perfect Scores', str(success_count)],
            ['Average Processing Time', f"{avg_processing_time:.2f}s"],
        ]
        summary_table = Table(summary_data, colWidths=[200, 100])
        summary_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(Paragraph(f'PDF Report - {today}', styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph('Summary Statistics', styles['Heading2']))
        elements.append(Spacer(1, 8))
        elements.append(summary_table)
        elements.append(Spacer(1, 16))
        elements.append(Paragraph('Color Key', styles['Heading2']))
        elements.append(Spacer(1, 8))
        elements.append(key_table)
        
        # AI Disclaimer on title page only
        disclaimer_style = ParagraphStyle(
            name='Disclaimer', parent=styles['BodyText'], 
            fontSize=8, leading=10, textColor=colors.grey,
            alignment=1  # Center alignment
        )
        disclaimer_text = "DISCLAIMER: All results are analyzed by AI and may not always be 100% accurate. This report is for guidance purposes only and should be reviewed by human operators for final validation."
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(disclaimer_text, disclaimer_style))
        elements.append(PageBreak())

        # Rounded hero banner title Flowable
        class HeroBannerTitle(Flowable):
            def __init__(self, text):
                super().__init__()
                self.text = text
                self.padding = 8
                self.line_height = 16
                self.radius = 8
                self.bg = colors.HexColor('#343a40')  # Same as infractions panel
                self.text_color = colors.white

            def wrap(self, availWidth, availHeight):
                # Calculate text width to fit the content
                c = self.canv
                c.saveState()
                c.setFont('Helvetica-Bold', 11)
                text_width = c.stringWidth(self.text, 'Helvetica-Bold', 11)
                c.restoreState()
                
                # Set width to text width + padding, with minimum width
                self.width = max(240, text_width + self.padding * 2)
                self._height = self.padding * 2 + self.line_height
                return self.width, self._height

            def draw(self):
                c = self.canv
                w = self.width
                h = getattr(self, '_height', 32)
                c.saveState()
                c.setFillColor(self.bg)
                c.setStrokeColor(self.bg)
                try:
                    c.roundRect(0, 0, w, h, self.radius, stroke=0, fill=1)
                except Exception:
                    c.rect(0, 0, w, h, stroke=0, fill=1)
                c.setFillColor(self.text_color)
                c.setFont('Helvetica-Bold', 11)
                c.drawString(self.padding, self.padding + 2, self.text)
                c.restoreState()

        # Rounded infractions panel Flowable
        class InfractionsPanel(Flowable):
            def __init__(self, lines, width):
                super().__init__()
                self.lines = lines
                self.width = width
                self.padding = 10
                self.line_height = 13
                self.radius = 10
                self.bg = colors.HexColor('#343a40')
                self.text_color = colors.white

            def wrap(self, availWidth, availHeight):
                # Ensure enough room for title + all lines (no clipping on last item)
                total_lines = (len(self.lines) + 1)  # title + lines
                self._height = self.padding * 2 + self.line_height * total_lines + 2  # safety padding
                return self.width, self._height

            def draw(self):
                c = self.canv
                w = self.width
                h = getattr(self, '_height', 60)
                c.saveState()
                c.setFillColor(self.bg)
                c.setStrokeColor(self.bg)
                try:
                    c.roundRect(0, 0, w, h, self.radius, stroke=0, fill=1)
                except Exception:
                    c.rect(0, 0, w, h, stroke=0, fill=1)
                c.setFillColor(self.text_color)
                y = h - self.padding - self.line_height
                c.setFont('Helvetica-Bold', 12)
                c.drawString(self.padding, y, 'Infractions')
                c.setFont('Helvetica', 10)
                for line in self.lines:
                    y -= self.line_height
                    c.drawString(self.padding, y, line)
                c.restoreState()

        for idx, result in enumerate(results, start=1):
            if idx > 1:
                elements.append(PageBreak())

            # Header with rounded rectangle styling
            filename = result.get('filename', 'Unknown')
            tag_text = f"Hero Banner {idx:02d} - {filename}"
            elements.append(HeroBannerTitle(tag_text))
            elements.append(Spacer(1, 8))

            # Build left image cell
            left_flowables = []
            try:
                img = result.get('annotated_image')
                if img is not None:
                    img_buf = io.BytesIO()
                    img.save(img_buf, format='PNG')
                    img_buf.seek(0)
                    # Target width for left column (roughly 65% of content width)
                    target_w = (doc.width * 0.65)
                    iw, ih = img.size
                    scale = min(1.0, target_w / float(iw))
                    rl_image = RLImage(img_buf, width=iw * scale, height=ih * scale)
                    left_flowables.append(rl_image)
            except Exception:
                pass

            # Right panel with metrics + infractions
            metrics_data = [
                ['Score', f"{result.get('score', 0)}%"],
                ['Infractions', str(len(result.get('penalties', [])))],
                ['Aspect Ratio', f"{float(result.get('aspect_ratio', 0.0)):.2f}"],
                ['Processing Time', f"{float(result.get('processing_time', 0.0)):.2f}s"],
            ]
            metrics_tbl = Table(metrics_data, colWidths=[110, 120])
            metrics_tbl.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
                ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))

            # Right panel now ONLY metrics (no infractions on the right)
            right_panel = [metrics_tbl]
            right_tbl = Table([[v] for v in right_panel], colWidths=[doc.width * 0.30])
            right_tbl.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))

            # Two-column layout: image left, panel right
            body_tbl = Table(
                [[left_flowables, right_tbl]],
                colWidths=[doc.width * 0.65, doc.width * 0.30],
            )
            body_tbl.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            elements.append(body_tbl)
            elements.append(Spacer(1, 12))

            # Full-width infractions panel with dark gray rounded rectangle and white text
            detailed_penalties = result.get('penalties', [])
            lines = []
            if detailed_penalties:
                for i, p in enumerate(detailed_penalties, 1):
                    try:
                        if len(p) >= 4:
                            msg, txt, pts = p[0], p[1], p[2]
                        elif len(p) == 3:
                            msg, txt, pts = p
                        else:
                            msg, pts = p
                            txt = ''
                        line = f"{i}. {msg}"
                        if txt:
                            line += f": '{txt}'"
                        line += f" ({pts})"
                        lines.append(line)
                    except Exception:
                        continue
            else:
                lines.append('No infractions - perfect score.')

            elements.append(InfractionsPanel(lines, doc.width))

        doc.build(elements)
        return buffer.getvalue()
    except ImportError:
        st.error("ReportLab not installed. Install with: pip install reportlab")
        return b""


def generate_excel_report(results: List[Dict]) -> Optional[bytes]:
    """Generate an Excel report with embedded annotated images."""
    try:
        import xlsxwriter
    except ImportError:
        st.error("XlsxWriter not installed. Install with: pip install XlsxWriter")
        return None

    try:
        output = io.BytesIO()
        try:
            wb = xlsxwriter.Workbook(output)
            ws = wb.add_worksheet("QA Results")
            
            # Set up formats
            header_format = wb.add_format({
                'bold': True,
                'bg_color': '#4F81BD',
                'font_color': 'white',
                'border': 1
            })
            
            # Write headers
            headers = ["Filename", "Score", "Infractions", "Processing Time", "Aspect Ratio"]
            for col, header in enumerate(headers):
                ws.write(0, col, header, header_format)
            
            # Write data
            for row, result in enumerate(results, start=1):
                ws.write(row, 0, result["filename"])
                ws.write(row, 1, f"{result['score']}%")
                ws.write(row, 2, len(result["penalties"]))
                ws.write(row, 3, f"{result['processing_time']:.2f}s")
                ws.write(row, 4, f"{result['aspect_ratio']:.2f}")
            
            # Add annotated images
            ws_img = wb.add_worksheet("Annotated Images")
            row_cursor = 0
            
            for result in results:
                try:
                    # Get the annotated image
                    annotated_img = result.get("annotated_image")
                    if annotated_img:
                        # Convert PIL Image to bytes
                        img_buf = io.BytesIO()
                        try:
                            annotated_img.save(img_buf, format="PNG", optimize=True)
                            img_buf.seek(0)
                            
                            # Insert image with proper scaling
                            max_width_px = 800  # reasonable width for Excel
                            iw, ih = annotated_img.size
                            scale = 1.0
                            if iw > max_width_px:
                                scale = max_width_px / float(iw)
                            
                            # Insert image with proper options - pass BytesIO object directly
                            ws_img.insert_image(row_cursor + 1, 0, img_buf, {
                                'x_scale': scale,
                                'y_scale': scale,
                                'x_offset': 5,
                                'y_offset': 5
                            })
                            
                            # Add some metadata about the image
                            ws_img.write(row_cursor + 1, 1, f"Score: {result.get('score', 0)}%")
                            ws_img.write(row_cursor + 1, 2, f"Infractions: {len(result.get('penalties', []))}")
                            
                            # Advance cursor by image height in rows (~20 px/row) plus some spacing
                            row_cursor += int((ih * scale) / 20) + 6
                        finally:
                            img_buf.close()
                    else:
                        row_cursor += 2
                except Exception as e:
                    # If image insertion fails, still add the filename and error info
                    ws_img.write(row_cursor + 1, 1, f"Error inserting image: {str(e)}")
                    row_cursor += 4

            wb.close()
            output.seek(0)
            return output.getvalue()
        finally:
            output.close()
    except Exception as e:
        st.error(f"Error generating Excel report: {e}")
        return None


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
    total_infractions = sum(
        entry.get("infractions", 0) for entry in analytics)  # Use infractions field instead of penalties
    success_rate = sum(1 for entry in analytics if entry.get("score", 0) == 100) / total_processed * 100
    avg_processing_time = sum(entry.get("processing_time", 0) for entry in analytics) / total_processed

    return {
        "total_processed": total_processed,
        "avg_score": round(avg_score, 1),
        "total_infractions": total_infractions,
        "success_rate": round(success_rate, 1),
        "avg_processing_time": round(avg_processing_time, 2)
    }
