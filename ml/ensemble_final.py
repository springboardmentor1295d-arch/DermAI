from pathlib import Path
import argparse
import json
import numpy as np
from ml.ensemble_advanced import predict
from ml.dataset import CLASSES
from ml.evaluate import compute_metrics, plot_confusion

paths = [
    Path("ml/runs/efficientnet_b0_fusion/efficientnet_b0_meta"),
    Path("ml/runs/efficientnet_b0_advanced/efficientnet_b0_meta"),
    Path("ml/runs/efficientnet_b0_fusion_v2/efficientnet_b0_meta")
]

out = Path("ml/runs/efficientnet_b0_ensemble_final")
out.mkdir(parents=True, exist_ok=True)


def load_predictions(split):
    predictions = []
    labels = None

    for i, path in enumerate(paths):
        print(f"Evaluating model {i + 1}: {split}")

        probs, y = predict(path, split)

        if labels is None:
            labels = y
        else:
            assert np.array_equal(labels, y)

        predictions.append(probs)

    return predictions, labels


def combine(predictions, weights, mode):
    return sum(
        w * p[mode]
        for w, p in zip(weights, predictions)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    print("Evaluating validation dataset...")

    val, yv = load_predictions("val")

    baseline_weights = [0.6, 0.4, 0.0]

    baseline = compute_metrics(
        yv,
        combine(val, baseline_weights, "tta2"),
        CLASSES
    )

    best = baseline
    best_weights = baseline_weights
    best_mode = "tta2"

    print("Previous validation accuracy:",
          baseline["accuracy"])

    print("Optimizing three-model ensemble...")

    for mode in ["original", "tta2", "tta4"]:

        for i in range(11):

            for j in range(11 - i):

                k = 10 - i - j

                weights = [
                    i / 10,
                    j / 10,
                    k / 10
                ]

                probs = combine(
                    val, weights, mode
                )

                met = compute_metrics(
                    yv, probs, CLASSES
                )

                if (
                    met["accuracy"] > best["accuracy"]
                    and met["macro_f1"] >= baseline["macro_f1"]
                    and met["balanced_accuracy"] >= baseline["balanced_accuracy"]
                    and met["per_class"]["MEL"]["sensitivity"] >= baseline["per_class"]["MEL"]["sensitivity"]
                    and met["per_class"]["SCC"]["sensitivity"] >= baseline["per_class"]["SCC"]["sensitivity"]
                ):
                    best = met
                    best_weights = weights
                    best_mode = mode

    print("\nBEST CONFIGURATION")

    print("Fusion V1 weight:", best_weights[0])
    print("Advanced weight:", best_weights[1])
    print("Fusion V2 weight:", best_weights[2])
    print("TTA method:", best_mode)

    for key in [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "macro_auc_ovr"
    ]:
        print(f"Validation {key}: {best[key]:.4f}")

    config = {
        "model_paths": [str(p) for p in paths],
        "weights": best_weights,
        "tta": best_mode,
        "validation_metrics": best
    }

    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if best["accuracy"] <= baseline["accuracy"]:
        print("No validation improvement.")
        print("Keep the previous ensemble.")
        return

    if not args.evaluate_test:
        print("Validation evaluation completed.")
        return

    print("\nEvaluating test dataset...")

    test, yt = load_predictions("test")

    probs = combine(
        test,
        best_weights,
        best_mode
    )

    metrics = compute_metrics(
        yt, probs, CLASSES
    )

    metrics["weights"] = best_weights
    metrics["tta"] = best_mode

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion(
        metrics["confusion_matrix"],
        CLASSES,
        out / "confusion_matrix.png",
        "Final EfficientNet-B0 Ensemble"
    )

    print("\nFINAL ENSEMBLE RESULTS")

    for key in [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "macro_auc_ovr"
    ]:
        print(f"{key}: {metrics[key]:.4f}")

    print("MEL recall:",
          metrics["per_class"]["MEL"]["sensitivity"])

    print("SCC recall:",
          metrics["per_class"]["SCC"]["sensitivity"])


if __name__ == "__main__":
    main()