import csv
import io
import json
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..config import DISCLAIMER, settings
from .risk import CLASS_NAMES


def report_dict(a, image, case):
    return dict(case_id=case.id, patient_code=case.patient_code, image_id=image.id, analysis_id=a.id,
                analysis_date=a.created_at.isoformat(), body_location=case.body_location,
                image_quality=image.quality_label, quality_details=image.quality,
                detected_lesion=dict(area_ratio=a.lesion_area_ratio, method=a.seg_method),
                predicted_condition=CLASS_NAMES.get(a.predicted_class, a.predicted_class), predicted_code=a.predicted_class,
                confidence=a.confidence, top3=a.top3, risk_category=a.risk, clinical_review_recommended=a.needs_review,
                explainability=a.explain_method, model=f"{a.model_version.name} v{a.model_version.version}" if a.model_version else None,
                summary=f"The model ranks {CLASS_NAMES.get(a.predicted_class)} most likely ({a.confidence:.0%}). "
                        f"Risk category (from dataset class labels): {a.risk}. Professional evaluation is recommended.",
                disclaimer=DISCLAIMER)


def to_json(d): return json.dumps(d, indent=2, default=str)


def to_csv(d):
    s = io.StringIO(); w = csv.writer(s); w.writerow(["field", "value"])
    for k, v in d.items():
        w.writerow([k, json.dumps(v, default=str) if isinstance(v, (dict, list)) else v])
    return s.getvalue()


def to_pdf(d, a):
    buf = io.BytesIO(); st = getSampleStyleSheet()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
    el = [Paragraph("DermAI Analysis Report", st["Title"]),
          Paragraph(f"<b>{DISCLAIMER}</b>", st["Normal"]), Spacer(1, 12)]
    rows = [["Case ID", d["case_id"]], ["Analysis ID", d["analysis_id"]], ["Date", d["analysis_date"][:19]],
            ["Body location", d["body_location"] or "-"], ["Image quality", d["image_quality"]],
            ["Predicted condition", f'{d["predicted_condition"]} ({d["predicted_code"]})'],
            ["Confidence", f'{d["confidence"]:.1%}'], ["Risk category", d["risk_category"]],
            ["Clinical review recommended", "Yes" if d["clinical_review_recommended"] else "Not flagged"],
            ["Explainability", d["explainability"]], ["Model", d["model"]]]
    t = Table(rows, colWidths=[5.5 * cm, 11 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .3, colors.grey), ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke)]))
    el += [t, Spacer(1, 10), Paragraph("Top predictions", st["Heading3"])]
    el.append(Table([["Class", "Probability"]] + [[CLASS_NAMES.get(c, c), f"{p:.1%}"] for c, p in d["top3"]]))
    imgs = []
    for p in (a.preprocessed_path, a.mask_path, a.cam_path):
        fp = os.path.join(settings.storage_dir, p) if p else None
        if fp and os.path.exists(fp): imgs.append(RLImage(fp, 5.2 * cm, 5.2 * cm))
    if imgs:
        el += [Spacer(1, 10), Paragraph("Preprocessed | Lesion mask | Explanation map", st["Heading3"]), Table([imgs])]
    el += [Spacer(1, 10), Paragraph(d["summary"], st["Normal"]), Spacer(1, 8), Paragraph(f"<i>{DISCLAIMER}</i>", st["Normal"])]
    doc.build(el); return buf.getvalue()
