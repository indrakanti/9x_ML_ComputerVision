"""Walkthrough for Module 25."""

from __future__ import annotations

from pathlib import Path

from pose_estimation import decode_keypoints, load_artifact, pck


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = load_artifact(root / "artifacts" / "pose.txt")
    points = decode_keypoints(artifact)

    print("Decoded keypoints:")
    for index, point in enumerate(points):
        print(index, point)
    print("PCK:", pck(artifact, points))


if __name__ == "__main__":
    main()
