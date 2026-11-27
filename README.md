# DermAI: Skin Lesion Analysis from Mobile Images (PAD-UFES-20)

AI-assisted screening / clinical decision-support platform. **Not a diagnostic device.**

```
ml/         EDA, patient-level split, training (CNN + Transformer + metadata fusion), comparison, ensemble
backend/    FastAPI: auth/RBAC, cases, quality check, segmentation, inference, Grad-CAM++, risk, reports, dashboards
frontend/   React (Vite) web app, Chart.js dashboards
mobile/     Expo React Native app (camera/gallery -> quality check -> upload -> result)
tests/      pytest (CV modules + API flow)
```

## 1. Train the models (do this first, on a GPU machine or Colab)

```bash
pip install -r ml/requirements.txt
# download PAD-UFES-20 from Mendeley Data (metadata.csv + imgs_part_1..3) and unzip somewhere
python -m ml.split --data-root /path/to/pad-ufes-20 --out ml/splits.csv   # PATIENT-level split, asserts no leakage
python -m ml.eda   --splits ml/splits.csv                                 # figures for your report
python -m ml.train --arch efficientnet_b0 --loss focal --epochs 25        # one model
python -m ml.train --arch convnext_tiny --use-meta                        # image + clinical metadata fusion (research)
python -m ml.compare --archs resnet50 densenet121 efficientnet_b0 convnext_tiny swin_tiny_patch4_window7_224
```
Outputs in `ml/runs/<arch>/`: `best.pt`, `metrics.json` (accuracy, macro-F1, balanced acc, MCC, per-class sensitivity/specificity/AUC,
confusion matrix, ECE, temperature), `confusion_matrix.png`; plus `ml/runs/comparison.csv` (includes the ensemble).
Add `--mlflow` (compare.py does) and run `mlflow ui` for experiment tracking.

Deploy: copy the checkpoints you want to serve into `backend/models/`
(`cp ml/runs/efficientnet_b0/best.pt backend/models/efficientnet_b0.pt`). They are auto-registered at startup, the first becomes active,
and admins can switch/roll back in the **Models** page. To serve an ensemble, register a model whose `path` is `a.pt,b.pt` (POST `/admin/models`).

## 2. Run locally

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload            # from repo root; docs at http://localhost:8000/docs
cd frontend && npm install && npm run dev        # http://localhost:5173
cd mobile && npm install && npx expo start       # edit API IP in App.js first
pytest -q
```
Default admin: see `.env` (`admin@dermai.local` / `admin12345`; **change it**). Uses SQLite locally, PostgreSQL in Docker.

## 3. Docker / cloud

```bash
docker compose up --build        # web http://localhost:8080 , API http://localhost:8000/docs
```
Cloud: push images to ECR/ACR, run on an EC2/Azure VM (docker compose) or ECS/App Service; use RDS/Azure Postgres.
Image storage is a local volume behind `services/storage.py`: swap that file for S3/Azure Blob (boto3) when deploying; use presigned URLs instead of the public `/media` mount.

## 4. Module map (PDF section -> code)

| PDF module | Where |
|---|---|
| 1 Auth & RBAC, audit | `routers/auth.py`, `deps.py`, `AuditLog` |
| 2 Patient & case mgmt | `routers/cases.py`, `models.Case` |
| 3, 11 Capture, validation, quality | `services/quality.py`, `/quality/check`, mobile/web UI |
| 4 Preprocessing | `services/preprocess.py` (EXIF, hair removal, gray-world colour norm, CLAHE, ROI crop) |
| 5 Lesion segmentation | `services/segmentation.py` (GrabCut, no masks in PAD-UFES-20) |
| 6, 7 Classification, advanced DL | `ml/models.py`, `ml/train.py`, `ml/compare.py` |
| 8 Risk | `services/risk.py` (from dataset class groups only) |
| 9 Explainability | `services/explain.py` (Grad-CAM++, saliency for ViT/Swin) |
| 10, 12 Training & evaluation | `ml/evaluate.py` |
| 13, 14 Dashboards, analysis screen | `routers/dashboard.py`, `frontend/src/App.jsx` |
| 15 Reports PDF/JSON/CSV | `services/reporting.py` |
| 16 History & follow-up | `/analyses`, `/analyses/{a}/compare/{b}` |
| 17 Notifications | `notify()` in `deps.py` |
| 18 Model versioning | `ModelVersion`, `/admin/models` |
| 19 Testing, Docker, CI | `tests/`, `docker-compose.yml`, `.github/workflows` |

## 5. Honest limitations (put these in your report)

- **Segmentation has no ground truth** in PAD-UFES-20, so Dice/IoU cannot be reported properly. Hand-annotate ~50 images (e.g. with CVAT/LabelMe) to evaluate, or generate SAM pseudo-masks and train a U-Net.
- **Quality thresholds** (`services/quality.py`) are heuristics. Tune them on your own phone photos and report rejection rate.
- **Small, imbalanced data** (MEL ≈ 52 images): report macro-F1, balanced accuracy and per-class recall, never accuracy alone.
- **Risk categories** come from dataset class groups; the "Requires Clinical Review" rule (confidence below `LOW_CONFIDENCE_THRESHOLD` or poor image) is a project-defined protocol, not a clinical threshold.
- Metadata-fusion models are research-only here; serving is image-only so users need not enter clinical metadata.
- Not clinically validated. Keep the disclaimer. Use only de-identified data.
