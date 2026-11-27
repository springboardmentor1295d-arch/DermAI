import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from .. import models
from ..config import settings
from ..database import get_db
from ..deps import STAFF, audit, get_case_or_404, get_current_user, notify
from ..services import explain, inference, preprocess, reporting, risk, segmentation, storage

router = APIRouter(tags=["analysis"])


def analysis_out(a):
    return dict(id=a.id, image_id=a.image_id, case_id=a.image.case_id, status=a.status, predicted_class=a.predicted_class,
                predicted_name=risk.CLASS_NAMES.get(a.predicted_class), confidence=a.confidence, top3=a.top3,
                probs=a.probs, risk=a.risk, needs_review=a.needs_review, lesion_area_ratio=a.lesion_area_ratio,
                seg_method=a.seg_method, explain_method=a.explain_method,
                original_url=f"/media/{a.image.path}", preprocessed_url=f"/media/{a.preprocessed_path}",
                mask_url=f"/media/{a.mask_path}", cam_url=f"/media/{a.cam_path}",
                quality_label=a.image.quality_label,
                model=f"{a.model_version.name} v{a.model_version.version}" if a.model_version else None,
                duration_ms=a.duration_ms, created_at=a.created_at)


def _get(db, analysis_id, user):
    a = db.get(models.Analysis, analysis_id)
    if not a:
        raise HTTPException(404, "Analysis not found")
    get_case_or_404(db, a.image.case_id, user)
    return a


@router.post("/images/{image_id}/analyze", status_code=201)
def analyze(image_id: int, force: bool = False, db: Session = Depends(get_db), user=Depends(get_current_user)):
    img = db.get(models.Image, image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    case = get_case_or_404(db, img.case_id, user)
    mv = db.query(models.ModelVersion).filter_by(is_active=True).first()
    if not mv:
        raise HTTPException(503, "No active model. Train one (ml/) and place best.pt in backend/models/, then restart.")
    if img.quality_label == "Poor" and not force:
        raise HTTPException(422, "Poor-quality image: recapture or retry with force=true")
    t0 = time.time()
    rgb = storage.load_rgb(img.path)
    mask, bbox, ratio, method = segmentation.segment_lesion(rgb)
    roi = preprocess.preprocess(rgb, bbox)
    try:
        res = inference.predict(mv.path, roi)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    cam, cam_method = explain.explain(res["primary"], res["x"], res["pred_idx"], roi, res["arch"])
    rk, review = risk.assess(res["pred"], res["confidence"], img.quality_label)
    base = f"a{int(time.time()*1000)}"
    a = models.Analysis(image_id=img.id, model_version_id=mv.id, predicted_class=res["pred"], confidence=res["confidence"],
                        top3=res["top3"], probs=res["probs"], risk=rk, needs_review=review, lesion_area_ratio=ratio,
                        seg_method=method, explain_method=cam_method,
                        preprocessed_path=storage.save_rgb(roi, "derived", base + "_pre"),
                        mask_path=storage.save_rgb(segmentation.overlay_mask(rgb, mask), "derived", base + "_mask"),
                        cam_path=storage.save_rgb(cam, "derived", base + "_cam"),
                        duration_ms=int((time.time() - t0) * 1000))
    db.add(a); db.commit(); audit(db, user, "analyze", str(a.id))
    uid = case.owner_id
    notify(db, uid, "info", "Analysis completed", f"Analysis #{a.id} for case #{case.id} is ready.")
    if img.quality_label == "Poor":
        notify(db, uid, "low", "Poor image quality", "Result may be unreliable; consider recapturing the image.")
    if res["confidence"] < settings.low_confidence_threshold:
        notify(db, uid, "medium", "Low-confidence prediction", "Clinical review recommended.")
    if res["pred"] in ("MEL", "BCC", "SCC") and not review:
        notify(db, uid, "high", "Higher-risk category predicted",
               "The AI output falls in a higher-risk dataset category. This is not a diagnosis - please seek professional evaluation.")
    return analysis_out(a)


@router.get("/analyses")
def history(case_id: int | None = None, date_from: str | None = None, date_to: str | None = None,
            db: Session = Depends(get_db), user=Depends(get_current_user)):
    q = db.query(models.Analysis).join(models.Image).join(models.Case)
    if user.role not in STAFF:
        q = q.filter(models.Case.owner_id == user.id)
    if case_id:
        q = q.filter(models.Image.case_id == case_id)
    if date_from:
        q = q.filter(models.Analysis.created_at >= date_from)
    if date_to:
        q = q.filter(models.Analysis.created_at <= date_to + " 23:59:59")
    return [analysis_out(a) for a in q.order_by(models.Analysis.id.desc()).limit(200).all()]


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    return analysis_out(_get(db, analysis_id, user))


@router.get("/analyses/{a_id}/compare/{b_id}")
def compare(a_id: int, b_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    a, b = _get(db, a_id, user), _get(db, b_id, user)
    return dict(previous=analysis_out(a), current=analysis_out(b), changes=dict(
        class_changed=a.predicted_class != b.predicted_class, confidence_delta=b.confidence - a.confidence,
        lesion_area_ratio_delta=(b.lesion_area_ratio or 0) - (a.lesion_area_ratio or 0),
        risk_changed=a.risk != b.risk,
        note="Area change is a rough indicator (different distance/angle affects it); not a clinical measurement."))


@router.get("/reports/{analysis_id}.{fmt}")
def report(analysis_id: int, fmt: str, db: Session = Depends(get_db), user=Depends(get_current_user)):
    a = _get(db, analysis_id, user)
    d = reporting.report_dict(a, a.image, a.image.case)
    audit(db, user, "report", f"{analysis_id}.{fmt}")
    if fmt == "json":
        return Response(reporting.to_json(d), media_type="application/json")
    if fmt == "csv":
        return Response(reporting.to_csv(d), media_type="text/csv")
    if fmt == "pdf":
        return Response(reporting.to_pdf(d, a), media_type="application/pdf",
                        headers={"Content-Disposition": f"attachment; filename=dermai_{analysis_id}.pdf"})
    raise HTTPException(400, "Format must be pdf, json or csv")


@router.post("/images/{image_id}/research-analyze")
def research_analyze(
    image_id: int,
    metadata: dict,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    import base64
    import io
    from PIL import Image
    from ..services import fusion

    if user.role not in ("admin", "researcher", "ml_engineer"):
        raise HTTPException(403, "Research access required")

    img = db.get(models.Image, image_id)

    if not img:
        raise HTTPException(404, "Image not found")

    get_case_or_404(db, img.case_id, user)

    if img.quality_label == "Poor":
        raise HTTPException(
            422, "Please upload a better-quality image"
        )

    mv = db.query(models.ModelVersion).filter_by(
        name="efficientnet_b0_fusion_v2"
    ).first()

    if not mv or not (mv.metrics or {}).get("research_only"):
        raise HTTPException(
            503, "Fusion V2 model is not registered"
        )

    rgb = storage.load_rgb(img.path)

    try:
        res = fusion.predict(mv.path, rgb, metadata)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception:
        raise HTTPException(500, "Fusion inference failed")

    cam, method = explain.explain(
        res["primary"],
        res["x"],
        res["pred_idx"],
        rgb,
        res["arch"]
    )

    buffer = io.BytesIO()
    Image.fromarray(cam).save(buffer, format="PNG")
    cam_data = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return {
        "model": "EfficientNet-B0 Fusion V2",
        "research_only": True,
        "predicted_class": res["pred"],
        "predicted_name": risk.CLASS_NAMES.get(res["pred"]),
        "confidence": res["confidence"],
        "top3": res["top3"],
        "probs": res["probs"],
        "metrics": mv.metrics,
        "explain_method": method,
        "explanation_image": "data:image/png;base64," + cam_data,
        "duration_ms": res["ms"],
        "disclaimer": "Research prototype only. Not a medical diagnosis."
    }
