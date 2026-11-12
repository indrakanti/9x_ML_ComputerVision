"""Walkthrough for Module 05 data discipline."""

from __future__ import annotations

from data_discipline import assign_split, binary_metrics, find_split_leakage


def main() -> None:
    sample_ids = ["frame_0001", "frame_0002", "frame_0003", "frame_0004", "frame_0005"]

    print("Deterministic split assignments")
    for sample_id in sample_ids:
        print(f"  {sample_id}: {assign_split(sample_id)}")

    leakage = find_split_leakage(
        {
            "train": ["scene_a", "scene_b"],
            "validation": ["scene_c"],
            "test": ["scene_b", "scene_d"],
        }
    )
    print("\nLeakage example")
    print(leakage)

    metrics = binary_metrics(
        truth=[1, 1, 1, 0, 0, 0],
        prediction=[1, 1, 0, 1, 0, 0],
    )

    print("\nBinary metrics")
    print(" TP:", metrics.true_positive)
    print(" TN:", metrics.true_negative)
    print(" FP:", metrics.false_positive)
    print(" FN:", metrics.false_negative)
    print(" accuracy:", metrics.accuracy)
    print(" precision:", metrics.precision)
    print(" recall:", metrics.recall)
    print(" specificity:", metrics.specificity)
    print(" f1:", metrics.f1)


if __name__ == "__main__":
    main()
