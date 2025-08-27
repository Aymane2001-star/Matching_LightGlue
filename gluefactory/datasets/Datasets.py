from torch.utils.data import Dataset
import cv2
import torch
from pathlib import Path
from torchvision.transforms import functional as F
import json 
import numpy as np
import copy

class FloorPlanDataset(Dataset):
    def __init__(self, base_dir, json_path, augmentations=None):
        """
        base_dir: Chemin vers le dossier contenant les paires d'images
        json_path: Chemin vers le fichier JSON contenant les annotations
        """
        self.base_dir = Path(base_dir)
        self.pairs = self._get_image_pairs(base_dir)
        self.augmentations = augmentations
        self.annotations = self._load_annotations(json_path)
        # Créer un mapping entre les paires d'images et les annotations
        self.pair_to_annotation = self._create_pair_mapping()
        print(f"Nombre total de paires d'annotations: {len(self.pair_to_annotation)}")
        print("Structure d'une paire d'annotations:")
        if self.pair_to_annotation:
            print("Structure d'une paire d'annotations:")
            print(json.dumps(list(self.pair_to_annotation.values())[0], indent=2))
        else:
            print("Aucune annotation mappée — vérifiez le contenu de votre JSON.")

    def _get_image_pairs(self, base_dir):
        """Parcourt la structure des dossiers pour détecter les paires d'images."""
        base_dir = Path(base_dir)
        pairs = []

        for scene_dir in sorted(base_dir.iterdir(), key=lambda d: int(''.join(filter(str.isdigit, d.name)))):
            if scene_dir.is_dir():
                images_dir = scene_dir / "images"
                if images_dir.exists():
                    images = sorted(images_dir.glob("*.png"))
                    if len(images) == 2:
                        pairs.append({
                            'dir_name': scene_dir.name,
                            'img1_path': str(images[0]),
                            'img2_path': str(images[1]),
                            'pair_id': len(pairs)  # Index séquentiel
                        })
                    else:
                        print(f"Attention : {scene_dir} contient {len(images)} images (attendu : 2).")

        return pairs

    def _load_annotations(self, json_path):
        """Charge les annotations depuis le fichier JSON"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement des annotations: {e}")
            return []

    def _create_pair_mapping(self):
        mapping = {}

        print(f"Nombre de paires d'images trouvées: {len(self.pairs)}")
        print(f"Nombre d'annotations trouvées: {len(self.annotations)}")

        for pair in self.pairs:
            pair_dir = pair['dir_name']
            pair_number = ''.join(filter(str.isdigit, pair_dir))  # par ex. '99'

            for ann in self.annotations:
                data = ann.get("data", {})
                img1 = data.get("image1", "")
                img2 = data.get("image2", "")

                # On cherche le numéro de pair dans les noms de fichiers
                found1 = "pair" + pair_number in img1
                found2 = "pair" + pair_number in img2

                if found1 and found2:
                    mapping[pair['dir_name']] = ann  ###########################
                    print(f"✅ Mapping créé: {pair_dir} -> pair{pair_number}")
                    break

        print(f"Nombre de paires mappées: {len(mapping)}")
        unmapped_dirs = set(p['dir_name'] for p in self.pairs) - set(mapping.keys()) 
        if unmapped_dirs:
            print(f"⚠️  Paires sans mapping: {unmapped_dirs}")

        return mapping







    """
    def _create_pair_mapping(self):
        #Crée un mapping entre les paires d'images et les annotations
        mapping = {}
    
        print(f"Nombre de paires d'images trouvées: {len(self.pairs)}")
        print(f"Nombre d'annotations trouvées: {len(self.annotations)}")
    
        for pair in self.pairs:
            try:
                # Extraire le numéro de la paire depuis le nom du dossier
                pair_num = int(''.join(filter(str.isdigit, pair['dir_name'])))
            
                # Chercher l'annotation correspondante dans le JSON
                for ann in self.annotations:
                    if 'file_upload' in ann:
                        file_name = ann['file_upload']
                        # Rechercher spécifiquement "pairX_" dans le nom du fichier
                        if 'pair' in file_name:
                            # Extraire le numéro entre "pair" et "_"
                            pair_part = file_name[file_name.find('pair'):file_name.find('_')]
                            json_pair_num = int(''.join(filter(str.isdigit, pair_part)))
                        
                            if json_pair_num == pair_num:
                                mapping[pair['pair_id']] = ann
                                print(f"Mapping créé: Dossier pair{pair_num} -> Annotation de pair{json_pair_num}")
                                break
                    
            except ValueError as e:
                print(f"Erreur lors du mapping pour {pair['dir_name']}: {e}")
                continue
    
        # Vérification finale
        print(f"Nombre de paires mappées: {len(mapping)}")
        unmapped_pairs = set(p['pair_id'] for p in self.pairs) - set(mapping.keys())
        if unmapped_pairs:
            print(f"Attention: Paires sans mapping: {unmapped_pairs}")
    
        return mapping
        """

 
    def _get_keypoints(self, idx):
        #Extrait les points-clés annotés pour une paire d'images
        try:
            # Obtenir l'annotation correspondante
            pair_dir = self.pairs[idx]['dir_name']
            ann = self.pair_to_annotation.get(pair_dir)  ################################

            if ann is None:
                print(f"Pas d'annotation trouvée pour la paire {idx}")
                return torch.zeros((0, 2)), torch.zeros((0, 2))

            # Points-clés indexés par label K1..K11 pour garantir l'alignement
            kpts1 = torch.zeros((11, 2), dtype=torch.float32)
            kpts2 = torch.zeros((11, 2), dtype=torch.float32)
            valid1 = torch.zeros((11,), dtype=torch.bool)
            valid2 = torch.zeros((11,), dtype=torch.bool)

            if 'annotations' in ann and len(ann['annotations']) > 0:
                results = ann['annotations'][0].get('result', [])
                for point in results:
                    if 'value' not in point:
                        continue
                    val = point['value']
                    labels = val.get('keypointlabels', [])
                    if not labels:
                        continue
                    label = labels[0]
                    try:
                        idx = int(''.join(filter(str.isdigit, label))) - 1
                    except Exception:
                        continue
                    if not (0 <= idx < 11):
                        continue
                    x = val['x'] * point['original_width'] / 100
                    y = val['y'] * point['original_height'] / 100
                    if point.get('to_name') == 'img-1':
                        kpts1[idx] = torch.tensor([x, y])
                        valid1[idx] = True
                    elif point.get('to_name') == 'img-2':
                        kpts2[idx] = torch.tensor([x, y])
                        valid2[idx] = True

            return kpts1, kpts2

        except Exception as e:
            print(f"Erreur lors de l'extraction des points-clés pour l'index {idx}: {e}")
            return torch.zeros((0, 2)), torch.zeros((0, 2))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        ann = self.pair_to_annotation.get(pair['dir_name'])  # Utilisation du nom réel de la paire
        if ann is None:
            return self.__getitem__((idx + 1) % len(self.pairs))

        img1 = cv2.imread(pair['img1_path'], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(pair['img2_path'], cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Impossible de charger les images pour la paire {idx}")

        kpts1, kpts2 = self._get_keypoints(idx)

        if self.augmentations is not None:
            original_shape = img1.shape
            img1_aug, img2_aug = self.augmentations.augment_pair(img1, img2)

            # ✅ copier l’historique ici
            history = copy.deepcopy(self.augmentations.transform_history)

            if len(kpts1) > 0:
                kpts1 = self.augmentations.transform_keypoints(kpts1, original_shape, history)
                kpts2 = self.augmentations.transform_keypoints(kpts2, original_shape, history)

            # ✅ puis seulement maintenant on applique les images transformées
            img1, img2 = img1_aug, img2_aug

            # ✅ et on peut vider l’historique sans risque
            self.augmentations.transform_history.clear()

        # Padding
        if len(kpts1) != 11 or len(kpts2) != 11:
            if len(kpts1) < 11:
                padding1 = torch.zeros((11 - len(kpts1), 2))
                kpts1 = torch.cat([kpts1, padding1], dim=0)
            else:
                kpts1 = kpts1[:11]
            if len(kpts2) < 11:
                padding2 = torch.zeros((11 - len(kpts2), 2))
                kpts2 = torch.cat([kpts2, padding2], dim=0)
            else:
                kpts2 = kpts2[:11]

        img1 = img1.astype("float32") / 255.0
        img2 = img2.astype("float32") / 255.0
        img1 = torch.from_numpy(img1).unsqueeze(0)
        img2 = torch.from_numpy(img2).unsqueeze(0)

        # Construire les GT en tenant compte du padding: -1 pour non-appairés
        valid_mask0 = (kpts1[:, 0] > 0) | (kpts1[:, 1] > 0)
        valid_mask1 = (kpts2[:, 0] > 0) | (kpts2[:, 1] > 0)
        m = n = 11

        # gt_assignment: uniquement les positifs (M x N), pas de dummy ici
        gt_assignment = torch.zeros((m, n), dtype=torch.float32)

        # Faire correspondre uniquement les labels présents dans les deux images (intersection)
        common_idx = torch.nonzero(valid_mask0 & valid_mask1, as_tuple=False).squeeze(-1)
        if common_idx.numel() > 0:
            gt_assignment[common_idx, common_idx] = 1.0

        # matches: -1 quand non apparié
        gt_matches0 = torch.full((m,), -1, dtype=torch.long)
        gt_matches1 = torch.full((n,), -1, dtype=torch.long)
        if common_idx.numel() > 0:
            gt_matches0[common_idx] = common_idx
            gt_matches1[common_idx] = common_idx

        return {
            'image0': img1,
            'image1': img2,
            'keypoints0': kpts1,
            'keypoints1': kpts2,
            'image_size0': torch.tensor([img1.shape[1], img1.shape[2]]),
            'image_size1': torch.tensor([img2.shape[1], img2.shape[2]]),
            'gt_assignment': gt_assignment,
            'gt_matches0': gt_matches0,
            'gt_matches1': gt_matches1,
        }

