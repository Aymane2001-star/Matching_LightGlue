import json
from pathlib import Path
from collections import defaultdict
import re

def extract_pair_id(filename):
    # Extrait "pair1" depuis "pair1_img0.png"
    match = re.search(r'(pair\d+)_img(\d)', filename)
    if match:
        return match.group(1), int(match.group(2))  # ("pair1", 0) ou ("pair1", 1)
    return None, None

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def merge_pairs(input_path, output_path):
    original_data = load_json(input_path)

    grouped = defaultdict(dict)
    for entry in original_data:
        fname = entry.get("file_upload", "")
        pair_id, img_num = extract_pair_id(fname)
        if pair_id is not None:
            grouped[pair_id][img_num] = entry

    merged = []

    for pair_id, imgs in grouped.items():
        if 0 in imgs and 1 in imgs:
            entry0 = imgs[0]
            entry1 = imgs[1]

            # Créer la nouvelle tâche fusionnée
            merged_entry = {
                "data": {
                    "image1": entry0["data"]["img"],
                    "image2": entry1["data"]["img"]
                },
                "annotations": [{
                    "result": []
                }]
            }

            for ann in entry0["annotations"][0]["result"]:
                ann_cp = ann.copy()
                ann_cp["to_name"] = "img-1"  # image1
                merged_entry["annotations"][0]["result"].append(ann_cp)

            for ann in entry1["annotations"][0]["result"]:
                ann_cp = ann.copy()
                ann_cp["to_name"] = "img-2"  # image2
                merged_entry["annotations"][0]["result"].append(ann_cp)

            merged.append(merged_entry)
        else:
            print(f"[WARN] Paire incomplète pour {pair_id}, ignorée.")

    print(f"[INFO] {len(merged)} paires fusionnées avec succès.")
    save_json(merged, output_path)

# Exemple d'utilisation :
input_json = r"C:\Users\STAGE2025\Downloads\project-2-at-2025-04-18-15-19-57cc4cfa.json"
output_json = "database.json"
merge_pairs(input_json, output_json)
