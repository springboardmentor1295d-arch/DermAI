import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from ml.src.inference import EfficientNetB0InferenceEngine

SAMPLE_IMAGE_IDS = [
    "PAT_1006_53_716.png",  # ACK
    "PAT_684_1302_588.png",   # BCC
    "PAT_966_1825_584.png",   # MEL
    "PAT_1516_1765_530.png",  # NEV
    "PAT_380_1540_959.png",   # SCC
    "PAT_107_160_609.png",    # SEK
]

IMAGES_DIR = Path(r"C:\DermAI_EDA\data\PAD-UFES-20_zr7vgbcyr2-1\images")
OUTPUT_PATHS = [
    Path("artifacts/ml/inference_samples_eval.json"),
    Path("outputs/inference_samples_eval.json"),
    Path("MENTOR_SUBMISSION_FINAL/evaluation_results/inference_samples_eval.json")
]


def main():
    print("=" * 75)
    print("DERMAI - REUSABLE EFFICIENTNET-B0 INFERENCE EVALUATION")
    print("Testing 6 Authentic Clinical Dataset Images across Diagnostic Classes")
    print("=" * 75)

    engine = EfficientNetB0InferenceEngine()
    print(f"Model Version:    {engine.model_version}")
    print(f"Checkpoint Path:  {engine.checkpoint_path}")
    print(f"Calibrator:       {'Fitted (T=' + str(round(engine.calibrator.temperature, 4)) + ')' if engine.calibrator else 'None'}")
    print(f"Compute Device:   {engine.device}")
    print("-" * 75)

    all_records = []

    for img_id in SAMPLE_IMAGE_IDS:
        matches = list(IMAGES_DIR.rglob(img_id))
        if not matches:
            print(f"ERROR: Image {img_id} not found in {IMAGES_DIR}")
            continue
        img_path = matches[0]

        res = engine.predict(img_path, top_k=3)
        res["sample_id"] = img_id
        res["file_path"] = str(img_path)
        all_records.append(res)

        print(f"\n[Sample: {img_id}]")
        print(f"  Dimensions:        {res['input_dimensions'][0]}x{res['input_dimensions'][1]} ({res['image_channels']})")
        print(f"  Predicted Class:   {res['predicted_class']} ({res['class_name']})")
        print(f"  Raw Model Score:   {res['raw_model_score'] * 100:.2f}%")
        print(f"  Calibrated Score:  {res['calibrated_confidence'] * 100:.2f}%")
        print(f"  Top-3 Predictions:")
        for t in res["top_3_predictions"]:
            print(f"    - {t['class_code']} ({t['class_name']}): Raw {t['raw_percentage_str']} | Calibrated {t['calibrated_percentage_str']}")
        print(f"  Inference Status:  {res['inference_status']}")
        print(f"  Timestamp:         {res['inference_timestamp']}")

    # Save to all target paths
    for p in OUTPUT_PATHS:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"samples": all_records}, f, indent=2)
        print(f"\nSaved inference evaluation results to: {p}")


if __name__ == "__main__":
    main()

