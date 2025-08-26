# 🎯 Banner QA – Advanced Text Zone Validation

A powerful Streamlit application for automated banner quality assurance with advanced text zone validation, batch processing, and analytics.

## ✨ Features

### 🚀 Core Functionality
- **OCR Text Detection**: Advanced text recognition using EasyOCR
- **Zone-based Validation**: Define text zones and ignore zones for precise control
- **Real-time Processing**: Instant feedback with annotated results
- **Aspect Ratio Validation**: Automatic 8:3 banner ratio checking

### 📁 Batch Processing
- **Multiple Image Upload**: Process dozens of images simultaneously
- **Progress Tracking**: Real-time progress indicators
- **Batch Analytics**: Comprehensive reporting across all processed images
- **Export Options**: CSV and PDF reports for batch results

### 🎨 Template System
- **Pre-built Templates**: Standard Banner, Minimal Banner, Product Banner
- **One-click Application**: Apply complete zone configurations instantly
- **Customizable**: Modify templates or create your own

### 📊 Analytics Dashboard
- **Performance Metrics**: Track scores, processing times, and success rates
- **Historical Data**: View trends over time
- **Visual Charts**: Score distribution and processing time trends
- **Success Rate Tracking**: Monitor improvement over time

### 📤 Export & Reporting
- **CSV Reports**: Detailed data export for analysis
- **PDF Reports**: Professional reports with summaries and charts
- **Individual Results**: Export single image results
- **Batch Reports**: Comprehensive batch processing summaries

### ⚡ Performance Optimizations
- **Smart Caching**: OCR results cached by image hash
- **File I/O Optimization**: Reduced disk operations with caching
- **Session State Management**: Persistent settings across interactions
- **Memory Efficient**: Optimized for large batch processing

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd bannerappv3
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📋 Requirements

- Python 3.8+
- Streamlit
- Pillow (PIL)
- EasyOCR
- Pandas
- ReportLab (for PDF generation)

## 🎮 Usage Guide

### Single Image Processing
1. Click "📄 Single Image" mode
2. Upload a banner image (PNG, JPG, JPEG)
3. Configure zones and ignore terms in the sidebar
4. View results with annotated image and metrics
5. Export results as CSV or PDF

### Batch Processing
1. Click "📁 Batch Processing" mode
2. Upload multiple banner images
3. Click "🚀 Process All Images"
4. Monitor progress with real-time indicators
5. View comprehensive batch results
6. Export batch report as CSV or PDF

### Template Management
1. Open the "🎨 Templates" section in the sidebar
2. Select a template (Standard, Minimal, or Product Banner)
3. Click "Apply Template" to load pre-configured zones
4. Customize zones as needed

### Analytics Dashboard
1. Click "📊 Analytics Dashboard" to view performance metrics
2. Monitor trends in scores and processing times
3. Track success rates over time
4. View detailed charts and visualizations

## ⚙️ Configuration

### Zone Management
- **Text Zones**: Define areas where text is allowed (green)
- **Ignore Zones**: Define areas to ignore (blue)
- **Ignore Terms**: Specific words/phrases to ignore

### Detection Settings
- **Overlap Threshold**: Minimum overlap percentage for text to count as inside a zone
- **OCR Parameters**: Adjust contrast, text threshold, and decoder settings

### Templates
- **Standard Banner**: Common 8:3 banner with headline, body, eyebrow, and CTA zones
- **Minimal Banner**: Simple banner with just headline and CTA
- **Product Banner**: Product-focused with multiple text areas

## 📁 File Structure

```
bannerappv3/
├── app.py                 # Main application
├── utils.py              # Utility functions
├── zone_editor.py        # Drag-and-drop zone editor
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── text_zones.json      # Saved text zones
├── ignore_zones.json    # Saved ignore zones
├── ignore_terms.json    # Saved ignore terms
└── analytics.json       # Analytics data
```

## 🔧 Advanced Features

### Custom Templates
Create custom templates by modifying the `load_templates()` function in `utils.py`:

```python
def load_templates() -> Dict[str, Dict]:
    templates = {
        "My Custom Template": {
            "description": "Description of my template",
            "text_zones": [
                {"name": "Zone Name", "zone": (x, y, width, height)}
            ],
            "ignore_zones": [...],
            "ignore_terms": [...]
        }
    }
    return templates
```

### Analytics Customization
Modify analytics tracking by editing the `save_analytics_data()` function in `utils.py`.

### Export Customization
Customize report formats by modifying the `generate_csv_report()` and `generate_pdf_report()` functions.

## 🚀 Performance Tips

1. **Use Templates**: Apply pre-configured templates for faster setup
2. **Batch Processing**: Process multiple images at once for efficiency
3. **Caching**: Results are automatically cached for identical images
4. **Zone Optimization**: Use fewer, larger zones when possible

## 🐛 Troubleshooting

### Common Issues

**OCR Not Working**
- Ensure EasyOCR is properly installed
- Check image format (PNG, JPG, JPEG)
- Verify image quality and text clarity

**Slow Processing**
- Use batch processing for multiple images
- Reduce image resolution if possible
- Check available system memory

**Zone Issues**
- Verify zone coordinates are between 0 and 1
- Ensure zones don't overlap unnecessarily
- Use templates for common configurations

### Error Messages

- **"Error loading file"**: Check file permissions and format
- **"OCR processing failed"**: Verify image quality and OCR installation
- **"Invalid zone coordinates"**: Ensure coordinates are normalized (0-1)

## 📈 Future Enhancements

- [ ] Drag-and-drop zone editor
- [ ] Multi-language OCR support
- [ ] API integration for external tools
- [ ] Cloud deployment options
- [ ] Advanced image preprocessing
- [ ] Machine learning-based zone suggestions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the analytics dashboard for performance insights

---

**Made with ❤️ for better banner quality assurance**
