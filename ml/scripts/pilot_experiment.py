from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
from ml.src.train import run_training_experiment

def main():
    print("=" * 60)
    print("DERMAI - RUNNING PILOT EXPERIMENT")
    print("=" * 60)
    start = time.time()
    
    # Run pilot experiment with small batch limit to test end-to-end tensor pipeline
    res = run_training_experiment(
        architecture="mobilenetv3_large",
        epochs=1,
        batch_size=16,
        lr=1e-4,
        seed=42,
        output_dir=Path("./artifacts/ml/pilot"),
        max_batches_per_epoch=5,
        pilot_mode=True
    )
    
    dur = time.time() - start
    print(f"\nPILOT COMPLETED in {dur:.2f} seconds.")
    print(f"Validation Balanced Accuracy: {res['best_val_balanced_acc']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
