"""
Parses UCF-Crime Temporal_Anomaly_Annotation.txt and produces a clean
label lookup for videos present in your local dataset folder.
"""

import os

ANNOTATION_FILE = "Temporal_Anomaly_Annotation.txt"
DATASET_DIR = "dataset"  # adjust to your actual extracted dataset path


def parse_annotations(path):
    entries = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            filename, label, s1, e1, s2, e2 = parts[:6]
            entries[filename] = {
                "label": label,
                "segments": [
                    (int(s1), int(e1)),
                    (int(s2), int(e2)),
                ],
            }
    return entries


def find_local_videos(dataset_dir):
    found = {}
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith(".mp4"):
                found[f] = os.path.join(root, f)
    return found


if __name__ == "__main__":
    annotations = parse_annotations(ANNOTATION_FILE)
    local_videos = find_local_videos(DATASET_DIR)

    shoplifting_annotated = []
    shoplifting_unannotated = []
    normal_videos = []

    for fname, path in local_videos.items():
        if fname in annotations:
            entry = annotations[fname]
            if entry["label"] == "Shoplifting":
                shoplifting_annotated.append((fname, path, entry["segments"]))
            elif entry["label"] == "Normal":
                normal_videos.append((fname, path))
        elif "Shoplifting" in path:
            shoplifting_unannotated.append((fname, path))

    print(f"Shoplifting WITH frame annotations: {len(shoplifting_annotated)}")
    for fname, _, segs in shoplifting_annotated:
        segs_str = ", ".join(f"{s}-{e}" for s, e in segs if s != -1)
        print(f"  {fname}: theft frames {segs_str}")

    print(
        f"\nShoplifting WITHOUT annotations (weak label only): {len(shoplifting_unannotated)}"
    )
    print(f"\nNormal videos (all frames negative): {len(normal_videos)}")
