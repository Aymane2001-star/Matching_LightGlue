import os
from pathlib import Path
import shutil
import random

# Chemin vers votre base de données actuelle
input_dir = Path(r"C:\My Program Files\LightGlue\data")  # Dossier contenant les paires d'images
output_dir = Path("organized_dataset")  # Dossier de sortie pour la nouvelle structure

# Dossiers cibles pour la nouvelle structure
sfm_dir = output_dir / "Undistorted_SfM"
depth_dir = output_dir / "depth_undistorted"  # Optionnel, vide ici
info_dir = output_dir / "scene_info"  # Optionnel, vide ici

# Créer les dossiers de sortie
sfm_dir.mkdir(parents=True, exist_ok=True)
depth_dir.mkdir(parents=True, exist_ok=True)
info_dir.mkdir(parents=True, exist_ok=True)

# Récupérer toutes les paires d'images
pairs = {}
for file in input_dir.iterdir():
    if file.is_file() and file.suffix in [".png", ".jpg", ".jpeg"]:  # Vérifie les fichiers image
        # Extraire le nom de la paire et de l'image
        pair_name, img_name = file.stem.split("_", 1)  # Exemple : "pair1_img0"
        if pair_name not in pairs:
            pairs[pair_name] = []
        pairs[pair_name].append(file)

# Trier les paires pour garantir un ordre cohérent
pairs = dict(sorted(pairs.items()))

# Diviser les paires en train (142) et validation (10)
pair_names = list(pairs.keys())
random.seed(42)  # Pour garantir la reproductibilité
train_pairs = pair_names[:142]
valid_pairs = pair_names[142:]

# Créer les fichiers texte
train_scenes_file = output_dir / "train_scenes_clean.txt"
valid_scenes_file = output_dir / "valid_scenes_clean.txt"
valid_pairs_file = output_dir / "valid_pairs.txt"

with train_scenes_file.open("w") as train_file, valid_scenes_file.open("w") as valid_file, valid_pairs_file.open("w") as pairs_file:
    for pair_name, images in pairs.items():
        # Créer le dossier de la scène
        scene_dir = sfm_dir / pair_name / "images"
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Copier les images dans le dossier de la scène
        for image in images:
            shutil.copy(image, scene_dir / image.name)

        # Ajouter la scène dans le fichier train ou valid
        if pair_name in train_pairs:
            train_file.write(f"{pair_name}\n")
        else:
            valid_file.write(f"{pair_name}\n")

        # Ajouter les paires d'images dans valid_pairs.txt si elles sont en validation
        if pair_name in valid_pairs:
            img0 = f"{pair_name}/images/{pair_name}_img0.png"
            img1 = f"{pair_name}/images/{pair_name}_img1.png"
            pairs_file.write(f"{img0} {img1}\n")

print(f"Base de données réorganisée dans : {output_dir}")
print(f"Fichiers texte créés :")
print(f"- {train_scenes_file}")
print(f"- {valid_scenes_file}")
print(f"- {valid_pairs_file}")
