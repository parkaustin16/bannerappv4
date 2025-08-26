import streamlit as st
from PIL import Image, ImageDraw
import json
import os
import easyocr
import io
import hashlib
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import pandas as pd

# Import our utility modules
from utils import (
    load_json_cached, save_json, get_image_hash, validate_image_aspect_ratio,
    overlap_ratio, convert_ocr_bbox_to_rect, normalize_coordinates, denormalize_coordinates,
    generate_csv_report, generate_pdf_report, get_download_link,
    save_analytics_data, get_analytics_summary,
    create_zone_preview_image, resize_image_for_display
)

# Page configuration
st.set_page_config(
    layout="wide",
    page_title="Banner QA - Text Zones",
    page_icon="🎯"
)

# --- Configuration Constants ---
IGNORE_FILE = "ignore_terms.json"
IGNORE_ZONES_FILE = "ignore_zones.json"
TEXT_ZONES_FILE = "text_zones.json"


# --- Cached OCR Reader ---
@st.cache_resource
def load_reader():
    """Load and cache the OCR reader to avoid reloading on every use."""
    return easyocr.Reader(["en"])


# --- Optimized OCR with Image Hash Caching ---
@st.cache_data
def run_ocr_cached(_reader, img_bytes: bytes, image_hash: str, contrast_ths=0.05, adjust_contrast=0.7,
                   text_threshold=0.7, decoder="beamsearch"):
    """Run OCR with caching based on image hash to avoid reprocessing identical images."""
    return _reader.readtext(
        img_bytes,
        contrast_ths=contrast_ths,
        adjust_contrast=adjust_contrast,
        text_threshold=text_threshold,
        decoder=decoder,
    )


# (auto-adjust helpers removed)

# --- Session State Management ---
def initialize_session_state():
    """Initialize all session state variables."""
    try:
        # Load defaults and initialize session copy
        if "text_zones_default" not in st.session_state:
            st.session_state.text_zones_default = load_json_cached(TEXT_ZONES_FILE, [])
        if "text_zones" not in st.session_state:
            st.session_state.text_zones = list(st.session_state.text_zones_default)

        if "persistent_ignore_terms" not in st.session_state:
            st.session_state.persistent_ignore_terms = load_json_cached(IGNORE_FILE, [])

        if "ignore_zones" not in st.session_state:
            st.session_state.ignore_zones = load_json_cached(IGNORE_ZONES_FILE, [])

        if "overlap_threshold" not in st.session_state:
            st.session_state.overlap_threshold = 0.75

        if "batch_results" not in st.session_state:
            st.session_state.batch_results = []

        # Track uploads to reset zones per image upload
        if "_last_upload_signatures" not in st.session_state:
            st.session_state._last_upload_signatures = None
        if "_is_single_upload" not in st.session_state:
            st.session_state._is_single_upload = True
        if "_cached_img_bytes" not in st.session_state:
            st.session_state._cached_img_bytes = None
        if "_cached_img_name" not in st.session_state:
            st.session_state._cached_img_name = None
        if "_hidden_zone_indices" not in st.session_state:
            st.session_state._hidden_zone_indices = set()
        if "_default_zone_signatures" not in st.session_state:
            try:
                st.session_state._default_zone_signatures = set(
                    (z.get("name", ""), tuple(z.get("zone", (0,0,0,0)))) for z in st.session_state.text_zones_default
                )
            except Exception:
                st.session_state._default_zone_signatures = set()

        if "current_mode" not in st.session_state:
            st.session_state.current_mode = "process"

        if "show_analytics" not in st.session_state:
            st.session_state.show_analytics = False

        # Ensure all zone lists are actually lists
        if not isinstance(st.session_state.text_zones, list):
            st.session_state.text_zones = []
        if not isinstance(st.session_state.ignore_zones, list):
            st.session_state.ignore_zones = []
        if not isinstance(st.session_state.persistent_ignore_terms, list):
            st.session_state.persistent_ignore_terms = []

    except Exception as e:
        st.error(f"Error initializing session state: {e}")
        # Set default values if initialization fails
        st.session_state.text_zones = []
        st.session_state.ignore_zones = []
        st.session_state.persistent_ignore_terms = []
        st.session_state.overlap_threshold = 0.75
        st.session_state.batch_results = []
        st.session_state.current_mode = "process"
        st.session_state.show_analytics = False


# --- Zone Management Functions ---
def add_text_zone(name: str, x: float, y: float, w: float, h: float) -> bool:
    """Add a new text zone."""
    try:
        zones_list = st.session_state.text_zones.copy()
        # Use float() to ensure we have valid numbers, avoid rounding issues
        zones_list.append({
            "name": name,
            "zone": (float(x), float(y), float(w), float(h))
        })
        st.session_state.text_zones = zones_list
        try:
            _reprocess_from_cache()
        except Exception:
            pass
        return True
    except Exception as e:
        st.error(f"Error adding text zone: {e}")
        return False


def delete_text_zone(index: int) -> bool:
    """Delete a text zone by index."""
    try:
        zones_list = st.session_state.text_zones.copy()
        if 0 <= index < len(zones_list):
            zones_list.pop(index)
            st.session_state.text_zones = zones_list
            try:
                _reprocess_from_cache()
            except Exception:
                pass
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting text zone: {e}")
        return False


def update_text_zone(index: int, x: float, y: float, w: float, h: float) -> bool:
    """Update an existing text zone by index with clamped normalized values."""
    try:
        zones_list = st.session_state.text_zones.copy()
        if 0 <= index < len(zones_list):
            # clamp values
            x = float(max(0.0, min(1.0, x)))
            w = float(max(0.0, min(1.0, w)))
            h = float(max(0.0, min(1.0, h)))
            y = float(max(0.0, min(1.0 - h, y)))
            if x + w > 1.0:
                w = max(0.0, 1.0 - x)
            zones_list[index]["zone"] = (x, y, w, h)
            st.session_state.text_zones = zones_list
            try:
                _reprocess_from_cache()
            except Exception:
                pass
            return True
        return False
    except Exception as e:
        st.error(f"Error updating text zone: {e}")
        return False


def add_ignore_zone(name: str, x: float, y: float, w: float, h: float) -> bool:
    """Add a new ignore zone."""
    try:
        zones_list = st.session_state.ignore_zones.copy()
        # Use float() to ensure we have valid numbers, avoid rounding issues
        zones_list.append({
            "name": name,
            "zone": (float(x), float(y), float(w), float(h))
        })
        st.session_state.ignore_zones = zones_list
        try:
            _reprocess_from_cache()
        except Exception:
            pass
        return True
    except Exception as e:
        st.error(f"Error adding ignore zone: {e}")
        return False


def delete_ignore_zone(index: int) -> bool:
    """Delete an ignore zone by index."""
    try:
        zones_list = st.session_state.ignore_zones.copy()
        if 0 <= index < len(zones_list):
            zones_list.pop(index)
            st.session_state.ignore_zones = zones_list
            try:
                _reprocess_from_cache()
            except Exception:
                pass
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting ignore zone: {e}")
        return False


def update_ignore_zone(index: int, x: float, y: float, w: float, h: float) -> bool:
    """Update an existing ignore zone by index with clamped normalized values."""
    try:
        zones_list = st.session_state.ignore_zones.copy()
        if 0 <= index < len(zones_list):
            x = float(max(0.0, min(1.0, x)))
            w = float(max(0.0, min(1.0, w)))
            h = float(max(0.0, min(1.0, h)))
            y = float(max(0.0, min(1.0 - h, y)))
            if x + w > 1.0:
                w = max(0.0, 1.0 - x)
            zones_list[index]["zone"] = (x, y, w, h)
            st.session_state.ignore_zones = zones_list
            try:
                _reprocess_from_cache()
            except Exception:
                pass
            return True
        return False
    except Exception as e:
        st.error(f"Error updating ignore zone: {e}")
        return False


def add_ignore_terms(terms: List[str]) -> bool:
    """Add new ignore terms."""
    try:
        new_terms = [t.strip().lower() for t in terms if t.strip()]
        st.session_state.persistent_ignore_terms.extend(new_terms)
        st.session_state.persistent_ignore_terms = sorted(set(st.session_state.persistent_ignore_terms))
        return save_json(IGNORE_FILE, st.session_state.persistent_ignore_terms)
    except Exception as e:
        st.error(f"Error adding ignore terms: {e}")
        return False


# --- Image Processing ---
def process_image(img: Image.Image, ocr_reader, overlap_threshold: float, filename: str = "Unknown", log_analytics: bool = True) -> Dict:
    """Process image with OCR and return results."""
    start_time = time.time()
    w, h = img.size
    # Draw on a copy to preserve original
    annotated_img = img.copy()
    draw = ImageDraw.Draw(annotated_img)

    # Validate aspect ratio
    aspect_ratio_valid, aspect_ratio = validate_image_aspect_ratio(img)

    # Convert zones to absolute coordinates
    abs_ignore_zones = []
    for item in st.session_state.ignore_zones:
        try:
            name = item.get("name", "Ignore Zone")
            zone_data = item.get("zone", (0, 0, 0, 0))
            # Ensure we have valid numbers
            nx, ny, nw, nh = float(zone_data[0]), float(zone_data[1]), float(zone_data[2]), float(zone_data[3])
            ix, iy, iw, ih = int(nx * w), int(ny * h), int(nw * w), int(nh * h)
            abs_ignore_zones.append((ix, iy, iw, ih))
            draw.rectangle([ix, iy, ix + iw, iy + ih], outline="blue", width=3)
            draw.text((ix + 5, iy + 5), name, fill="blue")
        except Exception as e:
            st.error(f"Error processing ignore zone {name}: {e}")
            continue

    # Draw text zones (respect hidden flags)
    for item in st.session_state.text_zones:
        try:
            name = item.get("name", "Zone")
            # Skip hidden zones
            if id(item) in st.session_state._hidden_zone_indices:
                continue
            zone_data = item.get("zone", (0, 0, 0, 0))
            # Ensure we have valid numbers
            nx, ny, nw, nh = float(zone_data[0]), float(zone_data[1]), float(zone_data[2]), float(zone_data[3])
            zx, zy, zw, zh = int(nx * w), int(ny * h), int(nw * w), int(nh * h)
            draw.rectangle([zx, zy, zx + zw, zy + zh], outline="green", width=3)
        except Exception as e:
            st.error(f"Error processing text zone {name}: {e}")
            continue

    # Prepare image for OCR (use original image, not annotated)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    # Run OCR with caching
    image_hash = get_image_hash(img)
    results = run_ocr_cached(ocr_reader, img_bytes, image_hash)

    penalties = []
    score = 100
    used_zones = {item["name"]: False for item in st.session_state.text_zones}

    # Process OCR results
    for (bbox, text, prob) in results:
        detected_text = text.strip()
        tx, ty, tw, th = convert_ocr_bbox_to_rect(bbox)

        # Clamp to image bounds
        tx = max(0, min(tx, w - 1))
        ty = max(0, min(ty, h - 1))
        tw = max(1, min(tw, w - tx))
        th = max(1, min(th, h - ty))

        # Check ignore zones first
        in_ignore_zone = False
        for izx, izy, izw, izh in abs_ignore_zones:
            if tx >= izx and ty >= izy and (tx + tw) <= (izx + izw) and (ty + th) <= (izy + izh):
                draw.rectangle([tx, ty, tx + tw, ty + th], outline="blue", width=3)
                in_ignore_zone = True
                break
        if in_ignore_zone:
            continue

        # Check overlap with text zones
        inside_any = False
        best_ratio = 0.0
        best_zone = None

        for item in st.session_state.text_zones:
            try:
                zone_name = item["name"]
                zone_data = item["zone"]
                # Ensure we have valid numbers
                nx, ny, nw, nh = float(zone_data[0]), float(zone_data[1]), float(zone_data[2]), float(zone_data[3])
                zx, zy, zw, zh = int(nx * w), int(ny * h), int(nw * w), int(nh * h)

                ratio = overlap_ratio((tx, ty, tw, th), (zx, zy, zw, zh))
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_zone = zone_name
                if ratio >= overlap_threshold:
                    inside_any = True
                    used_zones[zone_name] = True
                    break
            except Exception as e:
                st.error(f"Error processing text zone {zone_name} in OCR: {e}")
                continue

        # Check ignore terms
        if any(term in detected_text.lower() for term in st.session_state.persistent_ignore_terms):
            draw.rectangle([tx, ty, tx + tw, ty + th], outline="blue", width=2)
            continue

        # Mark text based on zone placement
        if inside_any:
            draw.rectangle([tx, ty, tx + tw, ty + th], outline="green", width=2)
            continue

        # Text outside zones - penalty
        draw.rectangle([tx, ty, tx + tw, ty + th], outline="red", width=2)
        if best_ratio > 0:
            msg = f"Text outside allowed zones (best overlap {best_ratio * 100:.1f}% with {best_zone})"
        else:
            msg = "Text outside allowed zones"
        penalties.append((msg, detected_text, -5))
        score = max(0, score - 5)  # Ensure score never goes below 0

    processing_time = time.time() - start_time

    result = {
        "filename": filename,
        "score": score,
        "penalties": penalties,
        "aspect_ratio_valid": aspect_ratio_valid,
        "aspect_ratio": aspect_ratio,
        "processing_time": processing_time,
        "image_size": (w, h),
        "zones_used": sum(used_zones.values()),
        "timestamp": datetime.now().isoformat(),
        "annotated_image": annotated_img
    }

    # Save analytics data (skip during redraws)
    if log_analytics:
        save_analytics_data(result)

    return result


def _reprocess_from_cache():
    """Redraw current image with updated zones without re-uploading.
    Uses cached reader and image bytes if available.
    """
    reader = st.session_state.get("_cached_reader") or load_reader()
    # Single image: rebuild list with one updated result
    if st.session_state.get("_is_single_upload", True):
        raw = st.session_state.get("_cached_img_bytes")
        if not raw:
            return
        try:
            img = Image.open(io.BytesIO(raw))
            result = process_image(img.convert("RGB"), reader, st.session_state.overlap_threshold,
                                   st.session_state.get("_cached_img_name", "Cached"), log_analytics=False)
            st.session_state.batch_results = [result]
        except Exception:
            pass
        return

    # Batch: rebuild entire results list to keep ordering and avoid duplication
    batch = st.session_state.get("_cached_batch") or []
    if not batch:
        return
    try:
        new_results = []
        for entry in batch:
            img = Image.open(io.BytesIO(entry["bytes"]))
            res = process_image(img.convert("RGB"), reader, st.session_state.overlap_threshold,
                                entry.get("name", "Cached"), log_analytics=False)
            new_results.append(res)
        st.session_state.batch_results = new_results
    except Exception:
        pass


# --- UI Components ---
def render_header():
    """Render the main header with mode selection."""
    st.title("🎯 Banner QA - Text Zones")

    # Mode selection
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📄 Process Images", use_container_width=True):
            st.session_state.current_mode = "process"
            st.session_state.show_analytics = False  # Hide analytics when switching to process mode
    with col2:
        if st.button("📊 Analytics Dashboard", use_container_width=True):
            st.session_state.show_analytics = not st.session_state.show_analytics

    st.markdown("---")


def render_analytics_dashboard():
    """Render the analytics dashboard."""
    if not st.session_state.show_analytics:
        return

    st.header("📊 Analytics Dashboard")

    # Get analytics summary
    analytics = get_analytics_summary()

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images Processed", analytics["total_processed"])
    with col2:
        st.metric("Average Score", f"{analytics['avg_score']}%")
    with col3:
        st.metric("Success Rate", f"{analytics['success_rate']}%")
    with col4:
        st.metric("Avg Processing Time", f"{analytics['avg_processing_time']}s")

    # Load detailed analytics
    analytics_file = "analytics.json"
    detailed_analytics = load_json_cached(analytics_file, [])

    if detailed_analytics:
        # Convert to DataFrame for better visualization
        df = pd.DataFrame(detailed_analytics)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Recent activity chart
        st.subheader("Recent Activity")
        recent_df = df.tail(20)
        st.line_chart(recent_df.set_index('timestamp')['score'])

        # Score distribution
        st.subheader("Score Distribution")
        score_counts = df['score'].value_counts().sort_index()
        st.bar_chart(score_counts)

        # Processing time trends
        st.subheader("Processing Time Trends")
        st.line_chart(recent_df.set_index('timestamp')['processing_time'])

    st.markdown("---")

    # Add wipe analytics button at the bottom with warning
    st.subheader("🗑️ Data Management")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if st.button("🗑️ Wipe Analytics", type="secondary"):
            st.session_state.show_wipe_warning = True

    # Show warning and confirmation
    if st.session_state.get("show_wipe_warning", False):
        st.warning("⚠️ **WARNING: This action cannot be undone!**")
        st.error(
            "This will permanently delete all analytics data including processing history, scores, and performance metrics.")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("✅ Yes, Delete All Data", type="primary"):
                try:
                    # Delete analytics file
                    if os.path.exists("analytics.json"):
                        os.remove("analytics.json")
                    st.session_state.show_wipe_warning = False
                    st.success("✅ Analytics data wiped successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error wiping analytics: {e}")

        with col2:
            if st.button("❌ Cancel", type="secondary"):
                st.session_state.show_wipe_warning = False
                st.rerun()


def render_sidebar():
    """Render the sidebar with all controls."""
    try:
        st.sidebar.title("⚙️ Settings")

        # Detection Settings
        with st.sidebar.expander("🔎 Detection Settings", expanded=False):
            st.session_state.overlap_threshold = st.slider(
                "Minimum overlap (%) for text to count as inside a zone",
                min_value=0.0, max_value=1.0, value=st.session_state.overlap_threshold,
                step=0.01, format="%.4f"
            )

        # Text Zones Management
        with st.sidebar.expander("📐 Text Zones", expanded=False):
            # (auto-adjust button removed)

            st.markdown("### Add Text Zone")
            zone_name = st.text_input("Zone Name", value="Zone 1", key="text_zone_name")

            col1, col2 = st.columns(2)
            with col1:
                tz_x = st.number_input("X", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.4f",
                                       key="tz_x")
                tz_y = st.number_input("Y", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.4f",
                                       key="tz_y")
            with col2:
                tz_w = st.number_input("Width", min_value=0.0, max_value=1.0, value=0.3, step=0.01, format="%.4f",
                                       key="tz_w")
                tz_h = st.number_input("Height", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.4f",
                                       key="tz_h")

            if st.button("💾 Add Text Zone"):
                if add_text_zone(zone_name, tz_x, tz_y, tz_w, tz_h):
                    st.success(f"✅ Text zone '{zone_name}' added!")
                    st.rerun()

            # Display saved zones
            if st.session_state.text_zones:
                st.markdown("**Current Text Zones:**")
                for i, item in enumerate(st.session_state.text_zones):
                    try:
                        name = item.get("name", f"Zone {i + 1}")
                        zone_data = item.get("zone", (0, 0, 0, 0))
                        # Ensure we have valid numbers
                        zx, zy, zw, zh = float(zone_data[0]), float(zone_data[1]), float(zone_data[2]), float(
                            zone_data[3])
                        st.write(f"{i + 1}. **{name}**  (x={zx:.3f}, y={zy:.3f}, w={zw:.3f}, h={zh:.3f})")

                        is_default = (name, (zx, zy, zw, zh)) in st.session_state._default_zone_signatures

                        col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
                        with col1:
                            hidden = id(item) in st.session_state._hidden_zone_indices
                            if st.button("👁️ Hide" if not hidden else "👁️ Show", key=f"toggle_hide_tz_{i}"):
                                if hidden:
                                    st.session_state._hidden_zone_indices.discard(id(item))
                                else:
                                    st.session_state._hidden_zone_indices.add(id(item))
                                _reprocess_from_cache()
                        with col2:
                            if st.button("✏️ Edit", key=f"edit_text_zone_{i}"):
                                st.session_state[f"_edit_text_zone_{i}"] = True
                        with col3:
                            if st.button("⬆️ Y -0.02", key=f"move_up_text_zone_{i}"):
                                update_text_zone(i, zx, max(0.0, zy - 0.02), zw, zh)
                        with col4:
                            if st.button("⬇️ Y +0.02", key=f"move_down_text_zone_{i}"):
                                update_text_zone(i, zx, min(1.0 - zh, zy + 0.02), zw, zh)
                        with col5:
                            if (not is_default) and st.button("❌ Delete", key=f"del_text_zone_{i}"):
                                delete_text_zone(i)

                        # Y-axis expand/shrink controls
                        col_e, col_f = st.columns(2)
                        with col_e:
                            if st.button("⬆️ H +0.02", key=f"expand_text_zone_{i}"):
                                new_h = min(1.0 - zy, zh + 0.02)
                                update_text_zone(i, zx, zy, zw, new_h)
                        with col_f:
                            if st.button("⬇️ H -0.02", key=f"shrink_text_zone_{i}"):
                                new_h = max(0.0, zh - 0.02)
                                update_text_zone(i, zx, zy, zw, new_h)

                        # Inline edit form
                        if st.session_state.get(f"_edit_text_zone_{i}"):
                            st.markdown("Edit fields:")
                            ec1, ec2, ec3, ec4 = st.columns(4)
                            with ec1:
                                ex = st.number_input("X", min_value=0.0, max_value=1.0, value=zx, step=0.01,
                                                     format="%.4f", key=f"edit_tz_x_{i}")
                            with ec2:
                                ey = st.number_input("Y", min_value=0.0, max_value=1.0, value=zy, step=0.01,
                                                     format="%.4f", key=f"edit_tz_y_{i}")
                            with ec3:
                                ew = st.number_input("W", min_value=0.0, max_value=1.0, value=zw, step=0.01,
                                                     format="%.4f", key=f"edit_tz_w_{i}")
                            with ec4:
                                eh = st.number_input("H", min_value=0.0, max_value=1.0, value=zh, step=0.01,
                                                     format="%.4f", key=f"edit_tz_h_{i}")
                            ec5, ec6 = st.columns(2)
                            with ec5:
                                if st.button("Save", key=f"save_text_zone_{i}"):
                                    if update_text_zone(i, ex, ey, ew, eh):
                                        st.session_state.pop(f"_edit_text_zone_{i}", None)
                            with ec6:
                                if st.button("Cancel", key=f"cancel_text_zone_{i}"):
                                    st.session_state.pop(f"_edit_text_zone_{i}", None)
                    except Exception as e:
                        st.error(f"Error displaying text zone {i}: {e}")

        # Ignore Settings
        with st.sidebar.expander("🛑 Ignore Settings", expanded=False):
            ignore_input = st.text_area("Enter words/phrases to ignore (comma separated):", key="ignore_input")
            if st.button("Add Ignore Terms"):
                new_terms = [t.strip() for t in ignore_input.split(",") if t.strip()]
                if add_ignore_terms(new_terms):
                    # clear input and reprocess current image to update score/infractions immediately
                    st.session_state["ignore_input"] = ""
                    try:
                        _reprocess_from_cache()
                    except Exception:
                        pass

            if st.session_state.persistent_ignore_terms:
                st.markdown("**Ignored Terms:**")
                for term in st.session_state.persistent_ignore_terms:
                    st.write(f"- {term}")

            st.markdown("### Add Ignore Zone")
            zone_name = st.text_input("Zone Name", value="Ignore Zone 1", key="ignore_zone_name")

            col1, col2 = st.columns(2)
            with col1:
                iz_x = st.number_input("X", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.4f",
                                       key="iz_x")
                iz_y = st.number_input("Y", min_value=0.0, max_value=1.0, value=0.9, step=0.01, format="%.4f",
                                       key="iz_y")
            with col2:
                iz_w = st.number_input("Width", min_value=0.0, max_value=1.0, value=0.8, step=0.01, format="%.4f",
                                       key="iz_w")
                iz_h = st.number_input("Height", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.4f",
                                       key="iz_h")

            if st.button("Add Ignore Zone"):
                if add_ignore_zone(zone_name, iz_x, iz_y, iz_w, iz_h):
                    st.success(f"✅ Ignore zone '{zone_name}' added!")
                    st.rerun()

            # Display saved ignore zones
            if st.session_state.ignore_zones:
                st.markdown("**Current Ignore Zones:**")
                for i, item in enumerate(st.session_state.ignore_zones):
                    try:
                        name = item.get("name", f"Zone {i + 1}")
                        zone_data = item.get("zone", (0, 0, 0, 0))
                        # Ensure we have valid numbers
                        zx, zy, zw, zh = float(zone_data[0]), float(zone_data[1]), float(zone_data[2]), float(
                            zone_data[3])
                        st.write(f"{i + 1}: **{name}** → (x={zx:.4f}, y={zy:.4f}, w={zw:.4f}, h={zh:.4f})")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button(f"❌ Delete", key=f"del_ignore_zone_{i}"):
                                delete_ignore_zone(i)
                        with col_b:
                            if st.button(f"✏️ Edit fields", key=f"edit_ignore_zone_{i}"):
                                st.session_state[f"_edit_ignore_zone_{i}"] = True

                        if st.session_state.get(f"_edit_ignore_zone_{i}"):
                            st.markdown("Edit fields:")
                            ec1, ec2, ec3, ec4 = st.columns(4)
                            with ec1:
                                ex = st.number_input("X", min_value=0.0, max_value=1.0, value=zx, step=0.01,
                                                     format="%.4f", key=f"edit_iz_x_{i}")
                            with ec2:
                                ey = st.number_input("Y", min_value=0.0, max_value=1.0, value=zy, step=0.01,
                                                     format="%.4f", key=f"edit_iz_y_{i}")
                            with ec3:
                                ew = st.number_input("W", min_value=0.0, max_value=1.0, value=zw, step=0.01,
                                                     format="%.4f", key=f"edit_iz_w_{i}")
                            with ec4:
                                eh = st.number_input("H", min_value=0.0, max_value=1.0, value=zh, step=0.01,
                                                     format="%.4f", key=f"edit_iz_h_{i}")
                            ec5, ec6 = st.columns(2)
                            with ec5:
                                if st.button("Save", key=f"save_ignore_zone_{i}"):
                                    if update_ignore_zone(i, ex, ey, ew, eh):
                                        st.session_state.pop(f"_edit_ignore_zone_{i}", None)
                                        st.success("Saved")
                            with ec6:
                                if st.button("Cancel", key=f"cancel_ignore_zone_{i}"):
                                    st.session_state.pop(f"_edit_ignore_zone_{i}", None)
                    except Exception as e:
                        st.error(f"Error displaying ignore zone {i}: {e}")

    except Exception as e:
        st.sidebar.error(f"Error rendering sidebar: {e}")
        # Try to show basic controls if sidebar fails
        st.sidebar.title("⚙️ Settings (Limited)")
        st.sidebar.info("Some settings may be unavailable due to an error.")


def render_process_mode():
    """Render the unified image processing mode (single or multiple images)."""
    st.header("📄 Image Processing")

    st.info("💡 **Tip:** You can upload one image or multiple images at once. The system will process them accordingly.")

    uploaded_files = st.file_uploader(
        "Upload banner(s)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="process_upload"
    )

    if uploaded_files:
        # Reset zones to defaults per new upload set (but keep in-session edits until the new upload occurs)
        try:
            current_sig = tuple(sorted(f.name for f in uploaded_files))
        except Exception:
            current_sig = None
        if current_sig and st.session_state._last_upload_signatures != current_sig:
            st.session_state.text_zones = list(st.session_state.text_zones_default)
            st.session_state.ignore_zones = load_json_cached(IGNORE_ZONES_FILE, [])
            st.session_state._last_upload_signatures = current_sig
            st.session_state._is_single_upload = (len(current_sig) == 1)
            # clear previous cached image when a new set is uploaded
            st.session_state._cached_img_bytes = None
            st.session_state._cached_img_name = None
        # Determine if single or multiple images
        is_single = len(uploaded_files) == 1

        # Auto-process images when uploaded
        if len(uploaded_files) != len(st.session_state.batch_results) or not st.session_state.batch_results:
            st.session_state.batch_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Load OCR reader once
            ocr_reader = load_reader()
            # cache reader for quick redraws
            st.session_state["_cached_reader"] = ocr_reader
            # prepare cache list for batch redraws
            cached_batch = []

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")

                try:
                    img = Image.open(uploaded_file).convert("RGB")
                    # cache original bytes for redraws
                    raw_buf = io.BytesIO()
                    img.save(raw_buf, format="PNG")
                    raw_bytes = raw_buf.getvalue()
                    st.session_state["_cached_img_bytes"] = raw_bytes
                    st.session_state["_cached_img_name"] = uploaded_file.name
                    cached_batch.append({"bytes": raw_bytes, "name": uploaded_file.name})
                    result = process_image(img, ocr_reader, st.session_state.overlap_threshold, uploaded_file.name)
                    st.session_state.batch_results.append(result)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            # store batch cache for redraws
            st.session_state["_cached_batch"] = cached_batch
            status_text.text("✅ Processing complete!")
            st.success(f"Processed {len(st.session_state.batch_results)} image(s)")

    # Display results
    if st.session_state.batch_results:
        # Determine display mode based on number of images
        is_single = len(st.session_state.batch_results) == 1

        if is_single:
            # Single image display
            result = st.session_state.batch_results[0]

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(result["annotated_image"], caption=f"QA Result – Score: {result['score']}%",
                         use_container_width=True)

            with col2:
                st.metric("Score", f"{result['score']}%")
                st.metric("Infractions", len(result["penalties"]))
                st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                st.metric("Aspect Ratio", f"{result['aspect_ratio']:.2f}")

            # Display penalties
            if result["penalties"]:
                st.error("Infractions:")
                for pen in result["penalties"]:
                    if len(pen) == 3:
                        msg, txt, pts = pen
                        st.write(f"{msg}: '{txt}' ({pts})")
                    else:
                        msg, pts = pen
                        st.write(f"{msg} ({pts})")
            else:
                st.success("Perfect score! ✅ All text inside zones.")

            # Export options for single image
            st.subheader("📤 Export Results")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Export as CSV"):
                    csv_data = generate_csv_report([result])
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"banner_qa_{result['filename']}.csv",
                        mime="text/csv"
                    )

            with col2:
                if st.button("📄 Export as PDF"):
                    pdf_data = generate_pdf_report([result])
                    if pdf_data:
                        st.download_button(
                            label="Download PDF",
                            data=pdf_data,
                            file_name=f"banner_qa_{result['filename']}.pdf",
                            mime="application/pdf"
                        )

        else:
            # Multiple images display (batch mode)
            st.subheader("📊 Batch Results")

            # Summary metrics
            total_images = len(st.session_state.batch_results)
            avg_score = sum(r["score"] for r in st.session_state.batch_results) / total_images
            total_infractions = sum(len(r["penalties"]) for r in st.session_state.batch_results)
            success_count = sum(1 for r in st.session_state.batch_results if r["score"] == 100)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Images", total_images)
            with col2:
                st.metric("Average Score", f"{avg_score:.1f}%")
            with col3:
                st.metric("Total Infractions", total_infractions)
            with col4:
                st.metric("Perfect Scores", success_count)

            # Results table
            st.subheader("Detailed Results")
            results_data = []
            for result in st.session_state.batch_results:
                results_data.append({
                    "Filename": result["filename"],
                    "Score": result["score"],
                    "Infractions": len(result["penalties"]),
                    "Aspect Ratio": f"{result['aspect_ratio']:.2f}",
                    "Processing Time": f"{result['processing_time']:.2f}s"
                })

            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)

            # Display annotated images with boxes
            st.subheader("📸 Annotated Images")

            # Create tabs for better organization
            tab_names = [f"{result['filename']} ({result['score']}%)" for result in st.session_state.batch_results]
            if len(tab_names) > 10:  # If too many images, show first 10
                tab_names = tab_names[:10]
                st.info(f"Showing first 10 images. Total processed: {len(st.session_state.batch_results)}")

            tabs = st.tabs(tab_names)

            for i, (tab, result) in enumerate(zip(tabs, st.session_state.batch_results[:10])):
                with tab:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Display annotated image
                        st.image(
                            result["annotated_image"],
                            caption=f"Score: {result['score']}% | Infractions: {len(result['penalties'])}",
                            use_container_width=True
                        )

                    with col2:
                        # Display metrics
                        st.metric("Score", f"{result['score']}%")
                        st.metric("Infractions", len(result["penalties"]))
                        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                        st.metric("Aspect Ratio", f"{result['aspect_ratio']:.2f}")

                        # Display penalties if any
                        if result["penalties"]:
                            st.error("Infractions:")
                            for pen in result["penalties"][:5]:  # Show first 5 penalties
                                if len(pen) == 3:
                                    msg, txt, pts = pen
                                    st.write(f"• {msg}: '{txt}' ({pts})")
                                else:
                                    msg, pts = pen
                                    st.write(f"• {msg} ({pts})")

                            if len(result["penalties"]) > 5:
                                st.write(f"... and {len(result['penalties']) - 5} more")
                        else:
                            st.success("✅ Perfect score!")

                    # Show aspect ratio status
                    if result["aspect_ratio_valid"]:
                        st.info("✅ Aspect ratio is valid (8:3)")
                    else:
                        st.warning(f"⚠️ Aspect ratio {result['aspect_ratio']:.2f} is not 8:3")

            # Export options for batch
            st.subheader("📤 Export Batch Results")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Export as CSV"):
                    csv_data = generate_csv_report(st.session_state.batch_results)
                    st.download_button(
                        label="Download CSV Report",
                        data=csv_data,
                        file_name=f"batch_qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

            with col2:
                if st.button("📄 Export as PDF"):
                    pdf_data = generate_pdf_report(st.session_state.batch_results)
                    if pdf_data:
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_data,
                            file_name=f"batch_qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )


# --- Main Application Logic ---
def main():
    """Main application function."""
    try:
        # Initialize session state
        initialize_session_state()

        # Render header
        render_header()

        # Render analytics dashboard if requested
        render_analytics_dashboard()

        # Render sidebar
        render_sidebar()

        # Render main content based on mode
        if st.session_state.current_mode == "process":
            render_process_mode()

    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page to restart the application.")
        # Show a simple restart button
        if st.button("🔄 Restart Application"):
            st.rerun()


if __name__ == "__main__":
    main()
