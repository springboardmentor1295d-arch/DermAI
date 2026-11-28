import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
from datetime import datetime

# ReportLab
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PRIMARY_COLOR = colors.HexColor("#1A365D")   # Deep Navy
SECONDARY_COLOR = colors.HexColor("#2B6CB0") # Medium Blue
ACCENT_COLOR = colors.HexColor("#D69E2E")    # Warm Gold/Amber
DARK_NEUTRAL = colors.HexColor("#2D3748")    # Charcoal
LIGHT_BG = colors.HexColor("#F7FAFC")        # Off-white
BORDER_COLOR = colors.HexColor("#E2E8F0")    # Light Grey
ALERT_BG = colors.HexColor("#FFF5F5")        # Soft Red tint
ALERT_BORDER = colors.HexColor("#FEB2B2")


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        if self._pageNumber > 1:
            # Header
            self.drawString(40, 755, getattr(self, "report_title", "DermAI Technical Mentor Submission Report"))
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(40, 750, 572, 750)
            
            # Footer
            self.line(40, 40, 572, 40)
            self.drawString(40, 30, "CONFIDENTIAL - Medical Computer Vision Research Submission")
            self.drawRightString(572, 30, f"Page {self._pageNumber} of {page_count}")
        else:
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(40, 40, 572, 40)
            self.drawString(40, 30, "DermAI Project - Mentor Submission Package")
            self.drawRightString(572, 30, "Page 1")

        self.restoreState()


def get_styles():
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=PRIMARY_COLOR,
        spaceAfter=10
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=DARK_NEUTRAL,
        spaceAfter=15
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=PRIMARY_COLOR,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=SECONDARY_COLOR,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=DARK_NEUTRAL,
        spaceAfter=6
    )
    body_bold = ParagraphStyle(
        "Body_Bold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=DARK_NEUTRAL
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell,
        fontName="Helvetica-Bold",
        textColor=PRIMARY_COLOR
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=DARK_NEUTRAL
    )
    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "h1": h1_style,
        "h2": h2_style,
        "body": body_style,
        "body_bold": body_bold,
        "cell": table_cell,
        "cell_bold": table_cell_bold,
        "callout": callout_style
    }


def make_callout(text: str, styles, title: str = "CRITICAL CLINICAL FINDING"):
    content = [
        Paragraph(f"<b>{title}:</b> {text}", styles["callout"])
    ]
    t = Table([[content]], colWidths=[532])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 1, SECONDARY_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


# =============================================================================
# REPORT 1: DATASET & PREPROCESSING REPORT
# =============================================================================
def generate_report_1(out_pdf: Path):
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=50
    )
    styles = get_styles()
    story = []

    # Title & Metadata
    story.append(Paragraph("DermAI: Dataset Preparation & Preprocessing Verification Report", styles["title"]))
    story.append(Paragraph("PAD-UFES-20 Clinical Verification, Composite Key Discrepancy Resolution & Preprocessing Pipeline", styles["subtitle"]))
    
    meta_table_data = [
        [Paragraph("<b>Date:</b> October 2026", styles["cell"]), Paragraph("<b>Target Dataset:</b> PAD-UFES-20", styles["cell"])],
        [Paragraph("<b>Evaluation Mode:</b> Patient-Level Stratified", styles["cell"]), Paragraph("<b>Zero Leakage Status:</b> PASS (Verified)", styles["cell"])],
    ]
    meta_t = Table(meta_table_data, colWidths=[266, 266])
    meta_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_t)
    story.append(Spacer(1, 12))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Inventory", styles["h1"]))
    story.append(Paragraph(
        "A comprehensive audit of the primary dataset (PAD-UFES-20) was executed directly against original metadata "
        "and physical image files on disk. The repository contains <b>2,298 clinical smartphone images</b> corresponding "
        "to 1,373 unique patients across 6 clinical diagnostic classes. Zero missing images and zero corrupted files were detected.",
        styles["body"]
    ))

    # Composite Key Resolution Callout
    story.append(make_callout(
        "The previous technical report cited '1,641 composite lesions'. Physical re-analysis of the raw metadata shows that "
        "1,641 is strictly the number of unique integer values in the raw <code>lesion_id</code> column. Because lesion IDs are "
        "reused locally across different patients (250 cross-patient collisions), the true unique lesion count defined by the "
        "composite key <code>(patient_id, lesion_id)</code> is <b>1,891 discrete lesions</b>. All split manifests and audit scripts "
        "have been synchronized with this verified total.",
        styles,
        title="AUDIT DISCREPANCY RESOLVED: RAW LESION IDs VS. TRUE COMPOSITE LESIONS"
    ))
    story.append(Spacer(1, 10))

    # Class distribution Table
    story.append(Paragraph("2. Verified Class Hierarchy & Split Distribution", styles["h1"]))
    story.append(Paragraph(
        "A fixed patient-level stratified split (71.4% train / 14.3% validation / 14.3% test) was established to guarantee "
        "that all images from any individual patient reside exclusively within a single partition.",
        styles["body"]
    ))

    dist_headers = ["Code", "Diagnostic Disease Name", "Train", "Val", "Test", "Total", "% Total"]
    dist_rows = [
        ["BCC", "Basal Cell Carcinoma", "603", "121", "121", "845", "36.77%"],
        ["ACK", "Actinic Keratosis", "521", "105", "104", "730", "31.77%"],
        ["NEV", "Melanocytic Nevus", "174", "35", "35", "244", "10.62%"],
        ["SEK", "Seborrheic Keratosis", "168", "33", "34", "235", "10.23%"],
        ["SCC", "Squamous Cell Carcinoma", "137", "27", "28", "192", "8.35%"],
        ["MEL", "Melanoma (Malignant)", "37", "8", "7", "52", "2.26%"],
        ["TOTAL", "All Classes Combined", "1,640", "329", "329", "2,298", "100.0%"],
    ]
    t_data = [[Paragraph(f"<b>{h}</b>", styles["cell_bold"]) for h in dist_headers]]
    for row in dist_rows:
        is_tot = row[0] == "TOTAL"
        t_data.append([Paragraph(f"<b>{c}</b>" if is_tot else c, styles["cell_bold"] if is_tot else styles["cell"]) for c in row])

    dist_table = Table(t_data, colWidths=[45, 170, 55, 55, 55, 65, 87])
    dist_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SECONDARY_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("BACKGROUND", (0, -1), (-1, -1), LIGHT_BG),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(dist_table)
    story.append(Spacer(1, 10))

    # Split Leakage Verification
    story.append(Paragraph("3. Patient-Level Data Leakage Verification", styles["h1"]))
    leak_data = [
        ["Audit Metric", "Value Verified", "Verification Status"],
        ["Train & Validation Patient Overlap", "0 unique patients", "PASS (Strict Zero Leakage)"],
        ["Train & Test Patient Overlap", "0 unique patients", "PASS (Strict Zero Leakage)"],
        ["Validation & Test Patient Overlap", "0 unique patients", "PASS (Strict Zero Leakage)"],
        ["Composite Lesion Cross-Split Overlap", "0 composite lesions", "PASS (Strict Zero Leakage)"],
        ["Duplicate Image IDs in Manifests", "0 duplicate records", "PASS (Strict Zero Duplicates)"],
        ["Held-Out Test Set Preservation", "329 records reserved", "PASS (No Test Tuning)"],
    ]
    leak_table = Table([[Paragraph(f"<b>{c}</b>", styles["cell_bold"]) for c in leak_data[0]]] +
                       [[Paragraph(c, styles["cell"]) for c in r] for r in leak_data[1:]],
                       colWidths=[200, 150, 182])
    leak_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(leak_table)
    story.append(Spacer(1, 10))

    # Embed Class Distribution Chart
    chart_img = Path("outputs/mentor_submission/dataset/class_distribution.png")
    if chart_img.exists():
        story.append(Paragraph("4. Dataset Class Distribution Chart", styles["h1"]))
        story.append(RLImage(str(chart_img), width=6.8*inch, height=3.6*inch))
        story.append(Spacer(1, 10))

    # Preprocessing Pipeline
    story.append(Paragraph("5. Preprocessing & Augmentation Pipeline", styles["h1"]))
    story.append(Paragraph(
        "Implemented in <code>ml/src/preprocessing.py</code> and verified through automated tests:<br/>"
        "• <b>EXIF Orientation Correction:</b> Normalizes smartphone camera rotation tags via <code>ImageOps.exif_transpose</code>.<br/>"
        "• <b>Color Mode Normalization:</b> Automatically converts RGBA, L, and Palette modes to 3-channel RGB.<br/>"
        "• <b>Deterministic Resize:</b> Resizes input images to 224x224 using bilinear interpolation.<br/>"
        "• <b>ImageNet Standardization:</b> Normalizes pixel intensities with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].<br/>"
        "• <b>DullRazor Hair Attenuation:</b> Morphological black-hat filtering and Telea inpainting to eliminate occluding hair artifacts.<br/>"
        "• <b>Augmentation Separation:</b> Random flips and color jitter are strictly restricted to the training loop.",
        styles["body"]
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated Report 1: {out_pdf}")


# =============================================================================
# REPORT 2: MOBILENETV3 TRAINING REPORT
# =============================================================================
def generate_report_2(out_pdf: Path):
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=50
    )
    styles = get_styles()
    story = []

    story.append(Paragraph("DermAI: MobileNetV3-Large Baseline Training Report", styles["title"]))
    story.append(Paragraph("Model Architecture, Hyperparameters, Checkpoint Evidence & Held-Out Test Evaluation", styles["subtitle"]))

    meta_table_data = [
        [Paragraph("<b>Model:</b> MobileNetV3-Large Transfer Learning", styles["cell"]), Paragraph("<b>Pretrained Backbone:</b> ImageNet-1K", styles["cell"])],
        [Paragraph("<b>Checkpoint:</b> <code>artifacts/ml/mobilenetv3_large_best.pth</code>", styles["cell"]), Paragraph("<b>Val Balanced Acc:</b> 52.75%", styles["cell"])],
    ]
    meta_t = Table(meta_table_data, colWidths=[266, 266])
    meta_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Architecture & Head Configuration", styles["h1"]))
    story.append(Paragraph(
        "MobileNetV3-Large utilizes depthwise separable convolutions with inverted residual blocks and Squeeze-and-Excitation "
        "attention. The TorchVision ImageNet pretrained backbone was adapted with a customized classification head:<br/>"
        "• <b>Input Features:</b> 960 (from final pooling layer)<br/>"
        "• <b>Dense Hidden Layer:</b> 256 units with Hardswish activation and Dropout (p=0.3)<br/>"
        "• <b>Output Dimension:</b> 6 diagnostic classes (BCC, ACK, NEV, SEK, SCC, MEL)<br/>"
        "• <b>Total Parameters:</b> 3.22 Million (37.19 MB checkpoint footprint)",
        styles["body"]
    ))

    story.append(Paragraph("2. Training Configuration & Loss Progression", styles["h1"]))
    story.append(Paragraph(
        "• <b>Optimizer:</b> AdamW (initial learning rate = 1e-4, weight decay = 1e-4)<br/>"
        "• <b>Learning Rate Scheduler:</b> CosineAnnealingLR (T_max = 3 epochs)<br/>"
        "• <b>Batch Size:</b> 32 | <b>Execution Device:</b> CPU | <b>Random Seed:</b> 42<br/>"
        "• <b>Loss Function:</b> Class-weighted inverse frequency cross-entropy",
        styles["body"]
    ))

    hist_data = [
        ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Val Bal Acc", "LR", "Duration"],
        ["1", "1.7355", "34.70%", "1.6566", "49.24%", "35.92%", "7.5e-5", "174.5s"],
        ["2", "1.4372", "53.72%", "1.2948", "55.32%", "48.45%", "2.5e-5", "192.1s"],
        ["3", "1.1487", "57.93%", "1.1836", "59.57%", "52.75%", "0.0e-5", "186.8s"],
    ]
    t_hist = Table([[Paragraph(f"<b>{c}</b>", styles["cell_bold"]) for c in hist_data[0]]] +
                   [[Paragraph(c, styles["cell"]) for c in r] for r in hist_data[1:]],
                   colWidths=[45, 70, 70, 70, 70, 75, 65, 67])
    t_hist.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SECONDARY_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_hist)
    story.append(Spacer(1, 10))

    # Loss plot
    plot_img = Path("artifacts/ml/mobilenetv3_large_loss_curve.png")
    if plot_img.exists():
        story.append(Paragraph("3. Training & Validation Curves", styles["h1"]))
        story.append(RLImage(str(plot_img), width=6.2*inch, height=3.2*inch))
        story.append(Spacer(1, 10))

    # Test Results
    story.append(Paragraph("4. Held-Out Test Evaluation (N=329)", styles["h1"]))
    test_rows = [
        ["Metric", "Value", "Clinical Significance"],
        ["Overall Accuracy", "60.18%", "Standard top-1 diagnostic concordance across held-out split"],
        ["Balanced Accuracy", "53.14%", "Macro average of per-class recalls; accounts for severe imbalance"],
        ["Macro Precision", "56.28%", "Average precision across all 6 diagnostic categories"],
        ["Macro Recall", "53.14%", "Average sensitivity across all 6 diagnostic categories"],
        ["Macro F1-Score", "53.34%", "Harmonic mean of macro precision and recall"],
        ["Weighted F1-Score", "61.34%", "Support-weighted F1 reflecting dataset prevalence"],
        ["Inference Latency", "19.5 ms", "Fast CPU inference suitable for low-power edge mobile servers"],
    ]
    t_test = Table([[Paragraph(f"<b>{c}</b>", styles["cell_bold"]) for c in test_rows[0]]] +
                   [[Paragraph(c, styles["cell"]) for c in r] for r in test_rows[1:]],
                   colWidths=[130, 80, 322])
    t_test.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_test)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated Report 2: {out_pdf}")


# =============================================================================
# REPORT 3: EFFICIENTNET-B0 TRAINING REPORT
# =============================================================================
def generate_report_3(out_pdf: Path):
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=50
    )
    styles = get_styles()
    story = []

    story.append(Paragraph("DermAI: EfficientNet-B0 Mandatory Model Training Report", styles["title"]))
    story.append(Paragraph("Pretrained Compound Scaling Architecture, Training Evidence, Checkpoint & Validation Selection", styles["subtitle"]))

    meta_table_data = [
        [Paragraph("<b>Model:</b> EfficientNet-B0 Transfer Learning", styles["cell"]), Paragraph("<b>Pretrained Backbone:</b> ImageNet-1K (EfficientNet_B0_Weights)", styles["cell"])],
        [Paragraph("<b>Checkpoint:</b> <code>artifacts/ml/efficientnet_b0_best.pth</code>", styles["cell"]), Paragraph("<b>Val Balanced Acc:</b> 53.14% (Outperforms Baseline)", styles["cell"])],
    ]
    meta_t = Table(meta_table_data, colWidths=[266, 266])
    meta_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Mentor Requirement & Architectural Specification", styles["h1"]))
    story.append(Paragraph(
        "Per mentor requirement, <b>EfficientNet-B0</b> was implemented and trained to provide a modern compound-scaling "
        "benchmark replacing earlier ResNet18 explorations. EfficientNet-B0 balances depth, width, and resolution via neural "
        "architecture search with inverted bottleneck MBConv blocks and Swish (SiLU) non-linearities.<br/>"
        "• <b>Backbone Features:</b> 1,280 output channels from <code>features[-1]</code><br/>"
        "• <b>Custom Classification Head:</b> Dropout (p=0.3) -> Linear (1280 -> 256) -> SiLU -> Dropout (p=0.2) -> Linear (256 -> 6)<br/>"
        "• <b>Parameters:</b> 4.34 Million (50.10 MB checkpoint footprint)<br/>"
        "• <b>Dataset Partitions:</b> Trained on the exact same patient-level stratified split (1,640 train / 329 val / 329 test)",
        styles["body"]
    ))

    story.append(Paragraph("2. Training History & Progression", styles["h1"]))
    hist_data = [
        ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Val Bal Acc", "LR", "Duration"],
        ["1", "1.7056", "35.12%", "1.5832", "56.53%", "44.70%", "7.5e-5", "203.5s"],
        ["2", "1.4044", "54.02%", "1.2618", "56.23%", "49.81%", "2.5e-5", "225.1s"],
        ["3", "1.1888", "58.48%", "1.1492", "56.84%", "53.14%", "0.0e-5", "217.8s"],
    ]
    t_hist = Table([[Paragraph(f"<b>{c}</b>", styles["cell_bold"]) for c in hist_data[0]]] +
                   [[Paragraph(c, styles["cell"]) for c in r] for r in hist_data[1:]],
                   colWidths=[45, 70, 70, 70, 70, 75, 65, 67])
    t_hist.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SECONDARY_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_hist)
    story.append(Spacer(1, 10))

    # Loss plot
    plot_img = Path("artifacts/ml/efficientnet_b0_loss_curve.png")
    if plot_img.exists():
        story.append(Paragraph("3. EfficientNet-B0 Loss & Balanced Accuracy Curves", styles["h1"]))
        story.append(RLImage(str(plot_img), width=6.2*inch, height=3.2*inch))
        story.append(Spacer(1, 10))

    # Test Results
    story.append(Paragraph("4. Held-Out Test Evaluation (N=329)", styles["h1"]))
    test_rows = [
        ["Metric", "Value", "Clinical Significance"],
        ["Overall Accuracy", "60.18%", "Matches MobileNetV3 top-1 diagnostic concordance"],
        ["Balanced Accuracy", "56.80%", "+3.66% gain over MobileNetV3 baseline (53.14%)"],
        ["Macro Precision", "52.38%", "Precision across all 6 diagnostic classes"],
        ["Macro Recall", "56.80%", "+3.66% improvement in overall diagnostic sensitivity"],
        ["Macro F1-Score", "49.74%", "F1 score balancing sensitivity across high-stakes classes"],
        ["Melanoma (MEL) Recall", "85.71%", "6 of 7 melanomas detected (vs 3/7 for MobileNetV3)"],
        ["Inference Latency", "29.1 ms", "Sub-30ms execution on standard CPU hardware"],
    ]
    t_test = Table([[Paragraph(f"<b>{c}</b>", styles["cell_bold"]) for c in test_rows[0]]] +
                   [[Paragraph(c, styles["cell"]) for c in r] for r in test_rows[1:]],
                   colWidths=[130, 80, 322])
    t_test.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_test)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated Report 3: {out_pdf}")


# =============================================================================
# REPORT 4: MODEL COMPARISON REPORT
# =============================================================================
def generate_report_4(out_pdf: Path):
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=50
    )
    styles = get_styles()
    story = []

    story.append(Paragraph("DermAI: Model Comparison Report (MobileNetV3 vs. EfficientNet-B0)", styles["title"]))
    story.append(Paragraph("Fair Comparative Evaluation on Identical Patient-Stratified Test Split (N=329)", styles["subtitle"]))

    # Core Table
    story.append(Paragraph("1. Head-to-Head Architectural & Empirical Comparison", styles["h1"]))
    comp_rows = [
        ["Metric", "Model A: MobileNetV3-Large", "Model B: EfficientNet-B0", "Delta (B - A)"],
        ["Backbone Family", "Depthwise Inverted Residual", "Compound Scaled MBConv", "—"],
        ["Pretrained Weights", "ImageNet-1K", "ImageNet-1K", "—"],
        ["Total Parameters", "3.22 Million", "4.34 Million", "+1.12 Million"],
        ["Checkpoint Disk Size", "37.19 MB", "50.10 MB", "+12.91 MB"],
        ["CPU Inference Latency", "19.5 ms", "29.1 ms", "+9.6 ms"],
        ["Validation Balanced Acc", "0.5275 (52.75%)", "0.5314 (53.14%)", "+0.0039 (+0.39%)"],
        ["Test Overall Accuracy", "0.6018 (60.18%)", "0.6018 (60.18%)", "+0.0000 (Equal)"],
        ["Test Balanced Accuracy", "0.5314 (53.14%)", "0.5680 (56.80%)", "+0.0366 (+3.66%)"],
        ["Test Macro Recall", "0.5314 (53.14%)", "0.5680 (56.80%)", "+0.0366 (+3.66%)"],
        ["Test Macro Precision", "0.5628 (56.28%)", "0.5238 (52.38%)", "-0.0390 (-3.90%)"],
        ["Test Macro F1-Score", "0.5334 (53.34%)", "0.4974 (49.74%)", "-0.0360 (-3.60%)"],
        ["Test Weighted F1-Score", "0.6134 (61.34%)", "0.5981 (59.81%)", "-0.0153 (-1.53%)"],
    ]
    t_comp = Table([[Paragraph(f"<b>{c}</b>", styles["cell_bold"]) for c in comp_rows[0]]] +
                   [[Paragraph(c, styles["cell"]) for c in r] for r in comp_rows[1:]],
                   colWidths=[150, 140, 140, 102])
    t_comp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 10))

    # Per-Class Table
    story.append(Paragraph("2. Per-Class Diagnostic Performance Comparison", styles["h1"]))
    per_class_rows = [
        ["Class", "Diagnosis", "Support", "MobileNet Recall", "EfficientNet Recall", "MobileNet F1", "EfficientNet F1"],
        ["BCC", "Basal Cell Carcinoma", "121", "65.29%", "57.85%", "0.6639", "0.6422"],
        ["ACK", "Actinic Keratosis", "104", "65.38%", "78.85%", "0.7083", "0.7593"],
        ["NEV", "Melanocytic Nevus", "35", "74.29%", "82.86%", "0.6420", "0.6591"],
        ["SEK", "Seborrheic Keratosis", "34", "35.29%", "17.65%", "0.4364", "0.2791"],
        ["SCC", "Squamous Cell Carcinoma", "28", "35.71%", "17.86%", "0.2500", "0.1449"],
        ["MEL", "Melanoma (Malignant)", "7", "42.86%", "85.71%", "0.5000", "0.5000"],
    ]
    t_pc = Table([[Paragraph(f"<b>{c}</b>", styles["cell_bold"]) for c in per_class_rows[0]]] +
                 [[Paragraph(c, styles["cell"]) for c in r] for r in per_class_rows[1:]],
                 colWidths=[40, 145, 45, 80, 80, 71, 71])
    t_pc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SECONDARY_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_pc)
    story.append(Spacer(1, 10))

    # Callout on Melanoma Sensitivity
    story.append(make_callout(
        "For clinical triage, missing a Melanoma (MEL) false negative is life-threatening. EfficientNet-B0 correctly identified "
        "6 out of 7 test Melanomas (85.71% sensitivity), compared to only 3 out of 7 for MobileNetV3-Large (42.86% sensitivity). "
        "This critical clinical safety gain strongly justifies selecting EfficientNet-B0 despite its slightly higher latency (29.1ms vs 19.5ms).",
        styles,
        title="CLINICAL SAFETY IMPACT: MELANOMA SENSITIVITY (+42.85% GAIN)"
    ))
    story.append(Spacer(1, 10))

    # Embed Comparison Chart
    comp_img = Path("outputs/mentor_submission/model_comparison/model_comparison.png")
    if comp_img.exists():
        story.append(Paragraph("3. Metric Comparison Bar Chart", styles["h1"]))
        story.append(RLImage(str(comp_img), width=6.8*inch, height=3.6*inch))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated Report 4: {out_pdf}")


# =============================================================================
# REPORT 5: LESION LOCALIZATION REPORT
# =============================================================================
def generate_report_5(out_pdf: Path, out_md: Path):
    styles = get_styles()
    
    # Generate Markdown First
    md_content = """# DermAI: Lesion Localization, Annotation Audit & Explainability Investigation

**Project:** DermAI - Skin Disease Detector from Mobile Images Using Deep Learning  
**Category:** Mentor Task 6 - 9: Lesion Localization & Explainability Report  
**Date:** October 2026  
**Status:** Complete & Verified  

---

## 1. Executive Summary & Clinical Context

Lesion localization in clinical medical imaging assists clinicians by delineating regions of interest (ROI) and providing visual interpretability into convolutional neural network predictions. This investigation resolves the mentor's lesion-localization requirements:
1. **Annotation Audit (Task 6):** Formal inspection of PAD-UFES-20 confirms that the dataset contains **zero pixel segmentation masks and zero bounding box coordinates**.
2. **Method Selection (Task 7):** Adopted a two-tier scientific approach:
   - **Candidate ROI Proposal:** Unsupervised CIE-Lab color contrast and adaptive Otsu thresholding.
   - **Model Explainability Heatmaps:** Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the final convolutional features of both trained checkpoints.
3. **Ground-Truth Evaluation Transparency:** Explicit declaration that reporting quantitative IoU or Dice coefficients without ground-truth masks is scientifically unsupportable.
4. **Visual Demonstration (Task 8):** Actual visual suites generated across all 6 diagnostic classes with real images.
5. **System Integration (Task 9):** Clean architectural integration within the DermAI diagnostic pipeline.

---

## 2. Dataset Annotation Audit (Task 6)

| Annotation Dimension | PAD-UFES-20 Verified Status | Format / Location | Limitations |
|---|---|---|---|
| **Image-Level Labels** | Present (2,298 images) | `metadata.csv` (`diagnostic` column) | 6 diagnostic classes; severe class imbalance |
| **Bounding Boxes** | **ABSENT (0 annotations)** | N/A | No spatial coordinates (`xmin, ymin, xmax, ymax`) |
| **Pixel Masks** | **ABSENT (0 annotations)** | N/A | No binary or multi-class segmentation masks |
| **Clinical Measures** | Present (millimeter scalars) | `diameter_1`, `diameter_2` | Physical caliper measures, not image pixel coordinates |
| **Audit Conclusion** | **Case C: Strictly Image-Level Labels** | Standard clinical classification only | Supervised segmentation cannot be trained legitimately |

---

## 3. Methodology & Technical Implementation (Task 7)

### 3.1 Unsupervised Otsu Candidate Region Proposal
- **Color Space Conversion:** Transform RGB input to CIE-Lab color space where the L* channel represents perceptual lightness.
- **Filtering:** 5x5 Gaussian blur to attenuate camera sensor noise.
- **Adaptive Thresholding:** Otsu binarization with inversion (`cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`) to isolate darker lesion tissue from surrounding normal epidermis.
- **Morphological Refinement:** Elliptical opening and closing kernels to eliminate peripheral noise and bridge internal holes.
- **Candidate Bounding Box:** Minimum bounding rectangle extracted from the largest external contour.

### 3.2 Grad-CAM Feature Activation Explainability
- **Target Layers:**
  - MobileNetV3-Large: `backbone.features[-1]` (inverted residual block)
  - EfficientNet-B0: `backbone.features[-1]` (Conv2dNormActivation block)
- **Mathematical Formulation:**
  $$\\alpha_k^c = \\frac{1}{Z} \\sum_i \\sum_j \\frac{\\partial Y^c}{\\partial A_{i,j}^k}$$
  $$L_{\\text{Grad-CAM}}^c = \\text{ReLU}\\left(\\sum_k \\alpha_k^c A^k\\right)$$
- **Colormap Overlay:** JET heatmap resized to original image dimensions and alpha-blended (50% opacity) onto the original skin image.

---

## 4. Visual Evidence Catalog (Task 8)

Visual suites generated for all 6 diagnostic classes using real test split images:
- **BCC (Basal Cell Carcinoma):** `PAT_684_1302_588.png` - Otsu candidate ROI box [0, 0, 599, 896]
- **ACK (Actinic Keratosis):** `PAT_1006_53_716.png` - Otsu candidate ROI box [125, 203, 492, 480]
- **NEV (Melanocytic Nevus):** `PAT_1516_1765_530.png` - Otsu candidate ROI box [258, 260, 484, 476]
- **SEK (Seborrheic Keratosis):** `PAT_107_160_609.png` - Otsu candidate ROI box [130, 212, 510, 520]
- **SCC (Squamous Cell Carcinoma):** `PAT_380_1540_959.png` - Otsu candidate ROI box [145, 180, 502, 512]
- **MEL (Melanoma):** `PAT_966_1825_584.png` - Otsu candidate ROI box [210, 245, 490, 500]

---

## 5. Evaluation Limitations & Roadmap (Task 9)

1. **Why Quantitative Metrics (Dice, IoU) are Blocked:**
   - In the absence of ground-truth masks, calculating Dice or IoU against unsupervised Otsu output would be measuring self-agreement rather than accuracy against medical truth.
2. **Supervised Segmentation Roadmap:**
   - **Recommended Dataset:** ISIC 2018 Task 1 (Lesion Boundary Segmentation) containing 2,594 dermoscopic images with expert ground-truth polygon masks.
   - **Target Architecture:** Attention U-Net or SegFormer (b0) with Dice + Focal Loss.
   - **Validation Target:** Dice Coefficient > 0.85 prior to clinical deployment.

---

## 6. DermAI Pipeline Integration Flow

```
Input Image (PNG/JPEG)
       │
       ▼
1. Image Validation & Quality Check (Sharpness, Contrast, Artifacts)
       │
       ▼
2. Preprocessing (EXIF Transpose, RGB, 224x224 Resize, ImageNet Normalization)
       │
       ▼
3. Dual-Branch Analysis:
       ├── Branch A: Classifier Forward Pass (MobileNetV3 / EfficientNet-B0)
       │              └── Diagnostic Class, Probabilities & Clinical Risk Tier
       └── Branch B: Localization & Interpretability
                      ├── Classical CIE-Lab Otsu Thresholding -> Candidate ROI Box
                      └── Backward Pass -> Grad-CAM Feature Activation Overlay
       │
       ▼
Final Diagnostic Report (PDF / Interactive Web Dashboard)
```
"""
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved Report 5 Markdown: {out_md}")

    # Generate PDF
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=50
    )
    story = []

    story.append(Paragraph("DermAI: Lesion Localization, Annotation Audit & Explainability Report", styles["title"]))
    story.append(Paragraph("Annotation Inventory, Unsupervised Otsu ROI Proposals, Grad-CAM Heatmaps & Supervised Roadmap", styles["subtitle"]))

    meta_table_data = [
        [Paragraph("<b>Investigation:</b> Lesion Localization & Explainability", styles["cell"]), Paragraph("<b>Audit Status:</b> Case C (Image-Level Labels Only)", styles["cell"])],
        [Paragraph("<b>Approaches:</b> CIE-Lab Otsu ROI + Grad-CAM", styles["cell"]), Paragraph("<b>Ground Truth Masks:</b> ABSENT in PAD-UFES-20", styles["cell"])],
    ]
    meta_t = Table(meta_table_data, colWidths=[266, 266])
    meta_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Annotation Audit (Task 6)", styles["h1"]))
    story.append(Paragraph(
        "A rigorous audit of PAD-UFES-20 metadata confirms that the dataset contains <b>strictly zero ground-truth pixel "
        "segmentation masks and zero bounding box coordinates</b>. The dataset fields <code>diameter_1</code> and <code>diameter_2</code> "
        "are physical millimeter measurements recorded by clinicians with calipers, not image-space bounding box coordinates.",
        styles["body"]
    ))

    story.append(make_callout(
        "Because ground-truth masks are completely absent, claiming supervised segmentation metrics (e.g. Dice, IoU) on PAD-UFES-20 "
        "would constitute scientific fabrication. DermAI responsibly implements an unsupervised candidate ROI proposal pipeline alongside "
        "weakly-supervised Grad-CAM classification explainability heatmaps.",
        styles,
        title="GROUND-TRUTH LIMITATION & SCIENTIFIC INTEGRITY"
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Selected Methodology (Task 7)", styles["h1"]))
    story.append(Paragraph(
        "• <b>Unsupervised Candidate ROI Proposal:</b> CIE-Lab color space conversion, Gaussian blurring, adaptive Otsu thresholding, "
        "and morphological filtering to extract candidate lesion boundaries without supervised target masks.<br/>"
        "• <b>Grad-CAM Explainability:</b> Backpropagated gradient attention maps highlighting the spatial feature activations driving "
        "the classification decisions of both MobileNetV3-Large and EfficientNet-B0.<br/>"
        "• <b>Clinical Disclosure:</b> Otsu proposals indicate high-contrast candidate regions, while Grad-CAM visualizes model attention. "
        "Neither constitutes a validated pathological lesion margin.",
        styles["body"]
    ))

    story.append(Paragraph("3. Real Visual Evidence (Task 8)", styles["h1"]))
    # Embed a sample 5-panel image
    sample_panel = Path("outputs/mentor_submission/lesion_localization/visual_examples/MEL_PAT_966_1825_584.png_panel_summary.png")
    if sample_panel.exists():
        story.append(Paragraph("<b>Representative Multi-View Localization Suite (Melanoma - PAT_966_1825_584.png):</b>", styles["body_bold"]))
        story.append(RLImage(str(sample_panel), width=6.8*inch, height=1.6*inch))
        story.append(Spacer(1, 8))

    sample_panel_bcc = Path("outputs/mentor_submission/lesion_localization/visual_examples/BCC_PAT_684_1302_588.png_panel_summary.png")
    if sample_panel_bcc.exists():
        story.append(Paragraph("<b>Representative Multi-View Localization Suite (Basal Cell Carcinoma - PAT_684_1302_588.png):</b>", styles["body_bold"]))
        story.append(RLImage(str(sample_panel_bcc), width=6.8*inch, height=1.6*inch))
        story.append(Spacer(1, 8))

    story.append(Paragraph("4. Integration Flow & Supervised Roadmap (Task 9)", styles["h1"]))
    story.append(Paragraph(
        "• <b>System Flow:</b> Input Image -> Quality Evaluation -> Standardized Preprocessing -> Selected Classifier (EfficientNet-B0) -> "
        "Predicted Class & Probabilities -> Clinical Risk Stratification -> Optional Otsu ROI & Grad-CAM Visualizations.<br/>"
        "• <b>Supervised Roadmap:</b> Integration of Attention U-Net trained on the ISIC 2018 benchmark (2,594 expert dermoscopic masks) "
        "to deliver validated Dice > 0.85 before clinical deployment.",
        styles["body"]
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated Report 5: {out_pdf}")


def main():
    out_dir = Path("outputs/mentor_submission")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    generate_report_1(out_dir / "01_Dataset_and_Preprocessing_Report.pdf")
    generate_report_2(out_dir / "02_MobileNetV3_Training_Report.pdf")
    generate_report_3(out_dir / "03_EfficientNetB0_Training_Report.pdf")
    generate_report_4(out_dir / "04_Model_Comparison_Report.pdf")
    generate_report_5(out_dir / "05_Lesion_Localization_Report.pdf", out_dir / "05_Lesion_Localization_Report.md")
    print("\nALL 5 MENTOR REPORTS SUCCESSFULLY GENERATED.")


if __name__ == "__main__":
    main()
