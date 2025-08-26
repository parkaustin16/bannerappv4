import streamlit as st
import streamlit.components.v1 as components
from typing import List, Dict, Tuple, Optional
from PIL import Image
import json

def create_zone_editor_html(img_width: int, img_height: int, zones: List[Dict], ignore_zones: List[Dict]) -> str:
    """Create HTML for the drag-and-drop zone editor."""
    
    # Convert zones to JavaScript format
    text_zones_js = []
    for zone in zones:
        name = zone.get("name", "Unnamed")
        nx, ny, nw, nh = zone.get("zone", (0, 0, 0, 0))
        x, y, w, h = int(nx * img_width), int(ny * img_height), int(nw * img_width), int(nh * img_height)
        text_zones_js.append({
            "id": f"text_{len(text_zones_js)}",
            "name": name,
            "x": x, "y": y, "width": w, "height": h,
            "type": "text"
        })
    
    ignore_zones_js = []
    for zone in ignore_zones:
        name = zone.get("name", "Unnamed")
        nx, ny, nw, nh = zone.get("zone", (0, 0, 0, 0))
        x, y, w, h = int(nx * img_width), int(ny * img_height), int(nw * img_width), int(nh * img_height)
        ignore_zones_js.append({
            "id": f"ignore_{len(ignore_zones_js)}",
            "name": name,
            "x": x, "y": y, "width": w, "height": h,
            "type": "ignore"
        })
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .zone-editor {{
                position: relative;
                display: inline-block;
                border: 2px solid #ccc;
                margin: 10px 0;
            }}
            .zone {{
                position: absolute;
                border: 2px solid;
                cursor: move;
                background-color: rgba(0, 255, 0, 0.2);
                min-width: 20px;
                min-height: 20px;
            }}
            .zone.text {{
                border-color: green;
                background-color: rgba(0, 255, 0, 0.2);
            }}
            .zone.ignore {{
                border-color: blue;
                background-color: rgba(0, 0, 255, 0.2);
            }}
            .zone-label {{
                position: absolute;
                top: -20px;
                left: 0;
                background: white;
                padding: 2px 4px;
                font-size: 10px;
                border: 1px solid #ccc;
                white-space: nowrap;
            }}
            .zone-handle {{
                position: absolute;
                width: 8px;
                height: 8px;
                background: white;
                border: 1px solid #000;
                cursor: se-resize;
            }}
            .zone-handle.nw {{ top: -4px; left: -4px; cursor: nw-resize; }}
            .zone-handle.ne {{ top: -4px; right: -4px; cursor: ne-resize; }}
            .zone-handle.sw {{ bottom: -4px; left: -4px; cursor: sw-resize; }}
            .zone-handle.se {{ bottom: -4px; right: -4px; cursor: se-resize; }}
            .controls {{
                margin: 10px 0;
                padding: 10px;
                background: #f0f0f0;
                border-radius: 5px;
            }}
            .zone-list {{
                margin: 10px 0;
                max-height: 200px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
            }}
            .zone-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 5px;
                margin: 2px 0;
                background: white;
                border: 1px solid #ddd;
            }}
            .zone-item button {{
                background: #ff4444;
                color: white;
                border: none;
                padding: 2px 6px;
                cursor: pointer;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="controls">
            <button onclick="addTextZone()">Add Text Zone</button>
            <button onclick="addIgnoreZone()">Add Ignore Zone</button>
            <button onclick="saveZones()">Save Zones</button>
            <button onclick="clearZones()">Clear All</button>
        </div>
        
        <div class="zone-editor" id="zoneEditor" style="width: {img_width}px; height: {img_height}px;">
            <img id="previewImage" src="data:image/png;base64,{get_image_base64()}" 
                 style="width: 100%; height: 100%; object-fit: contain;">
        </div>
        
        <div class="zone-list">
            <h4>Text Zones:</h4>
            <div id="textZoneList"></div>
            <h4>Ignore Zones:</h4>
            <div id="ignoreZoneList"></div>
        </div>
        
        <script>
            let zones = {json.dumps(text_zones_js + ignore_zones_js)};
            let selectedZone = null;
            let isDragging = false;
            let isResizing = false;
            let resizeHandle = null;
            let startX, startY;
            
            const editor = document.getElementById('zoneEditor');
            const imgWidth = {img_width};
            const imgHeight = {img_height};
            
            function addTextZone() {{
                const zone = {{
                    id: 'text_' + Date.now(),
                    name: 'Text Zone ' + (zones.filter(z => z.type === 'text').length + 1),
                    x: 50,
                    y: 50,
                    width: 100,
                    height: 50,
                    type: 'text'
                }};
                zones.push(zone);
                createZoneElement(zone);
                updateZoneLists();
            }}
            
            function addIgnoreZone() {{
                const zone = {{
                    id: 'ignore_' + Date.now(),
                    name: 'Ignore Zone ' + (zones.filter(z => z.type === 'ignore').length + 1),
                    x: 50,
                    y: 50,
                    width: 100,
                    height: 50,
                    type: 'ignore'
                }};
                zones.push(zone);
                createZoneElement(zone);
                updateZoneLists();
            }}
            
            function createZoneElement(zone) {{
                const zoneEl = document.createElement('div');
                zoneEl.className = `zone ${{zone.type}}`;
                zoneEl.id = zone.id;
                zoneEl.style.left = zone.x + 'px';
                zoneEl.style.top = zone.y + 'px';
                zoneEl.style.width = zone.width + 'px';
                zoneEl.style.height = zone.height + 'px';
                
                const label = document.createElement('div');
                label.className = 'zone-label';
                label.textContent = zone.name;
                zoneEl.appendChild(label);
                
                // Add resize handles
                const handles = ['nw', 'ne', 'sw', 'se'];
                handles.forEach(pos => {{
                    const handle = document.createElement('div');
                    handle.className = `zone-handle ${{pos}}`;
                    handle.dataset.handle = pos;
                    zoneEl.appendChild(handle);
                }});
                
                zoneEl.addEventListener('mousedown', startDrag);
                zoneEl.addEventListener('click', selectZone);
                
                editor.appendChild(zoneEl);
            }}
            
            function startDrag(e) {{
                if (e.target.classList.contains('zone-handle')) {{
                    isResizing = true;
                    resizeHandle = e.target.dataset.handle;
                }} else {{
                    isDragging = true;
                }}
                selectedZone = e.currentTarget;
                startX = e.clientX - selectedZone.offsetLeft;
                startY = e.clientY - selectedZone.offsetTop;
                e.preventDefault();
            }}
            
            function selectZone(e) {{
                if (selectedZone) {{
                    selectedZone.style.borderWidth = '2px';
                }}
                selectedZone = e.currentTarget;
                selectedZone.style.borderWidth = '3px';
                
                // Prompt for name change
                const newName = prompt('Enter zone name:', selectedZone.querySelector('.zone-label').textContent);
                if (newName) {{
                    selectedZone.querySelector('.zone-label').textContent = newName;
                    const zone = zones.find(z => z.id === selectedZone.id);
                    if (zone) zone.name = newName;
                }}
            }}
            
            function onMouseMove(e) {{
                if (!isDragging && !isResizing) return;
                
                const rect = editor.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                if (isDragging) {{
                    const newX = Math.max(0, Math.min(x - startX, imgWidth - selectedZone.offsetWidth));
                    const newY = Math.max(0, Math.min(y - startY, imgHeight - selectedZone.offsetHeight));
                    selectedZone.style.left = newX + 'px';
                    selectedZone.style.top = newY + 'px';
                    
                    const zone = zones.find(z => z.id === selectedZone.id);
                    if (zone) {{
                        zone.x = newX;
                        zone.y = newY;
                    }}
                }} else if (isResizing) {{
                    const zone = zones.find(z => z.id === selectedZone.id);
                    if (!zone) return;
                    
                    let newWidth = zone.width;
                    let newHeight = zone.height;
                    let newX = zone.x;
                    let newY = zone.y;
                    
                    switch (resizeHandle) {{
                        case 'se':
                            newWidth = Math.max(20, x - zone.x);
                            newHeight = Math.max(20, y - zone.y);
                            break;
                        case 'sw':
                            newWidth = Math.max(20, zone.x + zone.width - x);
                            newHeight = Math.max(20, y - zone.y);
                            newX = x;
                            break;
                        case 'ne':
                            newWidth = Math.max(20, x - zone.x);
                            newHeight = Math.max(20, zone.y + zone.height - y);
                            newY = y;
                            break;
                        case 'nw':
                            newWidth = Math.max(20, zone.x + zone.width - x);
                            newHeight = Math.max(20, zone.y + zone.height - y);
                            newX = x;
                            newY = y;
                            break;
                    }}
                    
                    selectedZone.style.left = newX + 'px';
                    selectedZone.style.top = newY + 'px';
                    selectedZone.style.width = newWidth + 'px';
                    selectedZone.style.height = newHeight + 'px';
                    
                    zone.x = newX;
                    zone.y = newY;
                    zone.width = newWidth;
                    zone.height = newHeight;
                }}
            }}
            
            function onMouseUp() {{
                isDragging = false;
                isResizing = false;
                resizeHandle = null;
            }}
            
            function deleteZone(zoneId) {{
                const zoneEl = document.getElementById(zoneId);
                if (zoneEl) {{
                    zoneEl.remove();
                }}
                zones = zones.filter(z => z.id !== zoneId);
                updateZoneLists();
            }}
            
            function updateZoneLists() {{
                const textList = document.getElementById('textZoneList');
                const ignoreList = document.getElementById('ignoreZoneList');
                
                textList.innerHTML = '';
                ignoreList.innerHTML = '';
                
                zones.forEach(zone => {{
                    const item = document.createElement('div');
                    item.className = 'zone-item';
                    item.innerHTML = `
                        <span>${{zone.name}} ({{zone.x}}, {{zone.y}}, {{zone.width}}, {{zone.height}})</span>
                        <button onclick="deleteZone('${{zone.id}}')">Delete</button>
                    `;
                    
                    if (zone.type === 'text') {{
                        textList.appendChild(item);
                    }} else {{
                        ignoreList.appendChild(item);
                    }}
                }});
            }}
            
            function saveZones() {{
                const normalizedZones = zones.map(zone => ({{
                    name: zone.name,
                    zone: [
                        zone.x / imgWidth,
                        zone.y / imgHeight,
                        zone.width / imgWidth,
                        zone.height / imgHeight
                    ]
                }}));
                
                const textZones = normalizedZones.filter(z => 
                    zones.find(oz => oz.name === z.name)?.type === 'text'
                );
                const ignoreZones = normalizedZones.filter(z => 
                    zones.find(oz => oz.name === z.name)?.type === 'ignore'
                );
                
                // Send data to Streamlit
                window.parent.postMessage({{
                    type: 'ZONE_EDITOR_SAVE',
                    textZones: textZones,
                    ignoreZones: ignoreZones
                }}, '*');
            }}
            
            function clearZones() {{
                if (confirm('Are you sure you want to clear all zones?')) {{
                    zones = [];
                    editor.querySelectorAll('.zone').forEach(el => el.remove());
                    updateZoneLists();
                }}
            }}
            
            // Initialize
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
            
            zones.forEach(zone => createZoneElement(zone));
            updateZoneLists();
        </script>
    </body>
    </html>
    """
    
    return html

def get_image_base64() -> str:
    """Get base64 encoded image for the editor."""
    # This would be replaced with actual image data
    # For now, return a placeholder
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

def render_zone_editor(img: Image.Image, zones: List[Dict], ignore_zones: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Render the zone editor and return updated zones."""
    
    # Resize image for editor
    max_width = 800
    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        editor_width = max_width
        editor_height = int(h * ratio)
    else:
        editor_width = w
        editor_height = h
    
    # Create HTML for the editor
    html_content = create_zone_editor_html(editor_width, editor_height, zones, ignore_zones)
    
    # Render the component
    components.html(html_content, height=editor_height + 300, scrolling=True)
    
    # Handle messages from the editor
    if 'zone_editor_data' in st.session_state:
        data = st.session_state.zone_editor_data
        st.session_state.pop('zone_editor_data')
        return data.get('textZones', []), data.get('ignoreZones', [])
    
    return zones, ignore_zones
