
import random
import cv2
import numpy as np
import torch


class DataAugmentation:
    def __init__(self, image_size):
        self.image_size = image_size  # (H, W)
        self.transform_history = []

    def safe_rotate(self, image, angle):
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adapter M pour recentrer
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)
        return rotated, M

    def apply_affine_to_keypoints(self, keypoints, M):
        if len(keypoints) == 0:
            return keypoints
        kpts = keypoints.clone().cpu().numpy().astype(np.float32).reshape(-1, 1, 2)
        transformed = cv2.transform(kpts, M).reshape(-1, 2)
        return torch.from_numpy(transformed).float()

    def augment_pair(self, img1, img2):
        self.transform_history.clear()
        h, w = img1.shape[:2]

        # Flip horizontal
        if random.random() < 0.5:
            img1 = cv2.flip(img1, 1)
            img2 = cv2.flip(img2, 1)
            self.transform_history.append({'type': 'flip_h'})

        # Flip vertical
        if random.random() < 0.5:
            img1 = cv2.flip(img1, 0)
            img2 = cv2.flip(img2, 0)
            self.transform_history.append({'type': 'flip_v'})

        # Rotation aléatoire
        angle = random.uniform(-30, 30)
        center = (w / 2, h / 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        img1 = cv2.warpAffine(img1, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img2 = cv2.warpAffine(img2, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # ✅ Ajout de 'angle' pour éviter KeyError
        self.transform_history.append({'type': 'rotation', 'matrix': M_rot, 'angle': angle})

        # Translation
        tx, ty = random.randint(-50, 50), random.randint(-50, 50)
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        img1 = cv2.warpAffine(img1, M_trans, (w, h), borderMode=cv2.BORDER_REFLECT)
        img2 = cv2.warpAffine(img2, M_trans, (w, h), borderMode=cv2.BORDER_REFLECT)
        self.transform_history.append({'type': 'translation', 'tx': tx, 'ty': ty})

        return img1, img2

    def transform_keypoints(self, keypoints, image_shape, history):
        """Applique les transformations à une copie numpy-safe des keypoints"""
        if len(keypoints) == 0:
            return keypoints

        kpts = keypoints.clone().cpu().numpy().astype(np.float32).reshape(-1, 1, 2)

        for transform in history:
            if transform['type'] == 'flip_h':
                kpts[:, 0, 0] = image_shape[1] - 1 - kpts[:, 0, 0]

            elif transform['type'] == 'flip_v':
                kpts[:, 0, 1] = image_shape[0] - 1 - kpts[:, 0, 1]

            elif transform['type'] == 'rotation':
                # ✅ Utilisation directe de la matrice de rotation
                M = transform['matrix']
                kpts = cv2.transform(kpts, M)

            elif transform['type'] == 'translation':
                tx = transform['tx']
                ty = transform['ty']
                kpts[:, 0, 0] += tx
                kpts[:, 0, 1] += ty

        kpts = kpts.reshape(-1, 2)
        return torch.from_numpy(kpts).float()




"""
import random
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomCrop
)

class DataAugmentation:
    def __init__(self, image_size):
        self.image_size = image_size  # Taille cible des images après augmentation
        self.transforms = Compose([
            HorizontalFlip(p=0.5),  # Flip horizontal
            VerticalFlip(p=0.5),    # Flip vertical
        ])
        self.transform_history = []

    def random_rotation_90(self, image):
        #Applique une rotation aléatoire de 90° à l'image.
        k = random.choice([0, 1, 2, 3])  # Choisir 0°, 90°, 180°, ou 270°
        return np.rot90(image, k)

    def random_translation(self, image, max_translation=50):
        #Applique une translation aléatoire à l'image.
        h, w = image.shape[:2]
        tx = random.randint(-max_translation, max_translation)
        ty = random.randint(-max_translation, max_translation)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(image, translation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        return translated

    def random_zoom(self, image, zoom_range=(0.8, 1.2)):
        #Applique un zoom in/out aléatoire à l'image.
        h, w = image.shape[:2]
        scale = random.uniform(*zoom_range)
        new_h, new_w = int(h * scale), int(w * scale)

        # Redimensionner l'image
        zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Si l'image est plus grande, on la recadre
        if scale > 1.0:
            crop_h = (new_h - h) // 2
            crop_w = (new_w - w) // 2
            zoomed = zoomed[crop_h:crop_h + h, crop_w:crop_w + w]
        # Si l'image est plus petite, on la remplit
        elif scale < 1.0:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            zoomed = cv2.copyMakeBorder(zoomed, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w,
                                        borderType=cv2.BORDER_REFLECT)

        return zoomed

    def augment_pair(self, img1, img2):
        self.transform_history.clear()
    
        # Flip horizontal avec probabilité 0.5
        if random.random() < 0.5:
            img1 = cv2.flip(img1, 1)
            img2 = cv2.flip(img2, 1)
            self.transform_history.append({'type': 'flip_h'})
    
        # Rotation aléatoire
        angle = random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((img1.shape[1]/2, img1.shape[0]/2), angle, 1.0)
        img1 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
        img2 = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))
        self.transform_history.append({'type': 'rotation', 'angle': angle})
    
        # Translation
        tx, ty = random.randint(-50, 50), random.randint(-50, 50)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img1 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
        img2 = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))
        self.transform_history.append({'type': 'translation', 'tx': tx, 'ty': ty})
    
        return img1, img2
    
    def transform_keypoints(self, keypoints, image_shape, history):
        #Applique les transformations à une copie numpy-safe des keypoints
        if len(keypoints) == 0:
            return keypoints  # rien à transformer

        # Sécuriser : clone, CPU, float32 numpy
        kpts = keypoints.clone().cpu().numpy().astype(np.float32).reshape(-1, 1, 2)

        for transform in history:
            if transform['type'] == 'flip_h':
                kpts[:, 0, 0] = image_shape[1] - kpts[:, 0, 0]
            elif transform['type'] == 'flip_v':
                kpts[:, 0, 1] = image_shape[0] - kpts[:, 0, 1]
            elif transform['type'] == 'rotation':
                angle = transform['angle']
                center = (image_shape[1] / 2, image_shape[0] / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                kpts = cv2.transform(kpts, M)
            elif transform['type'] == 'translation':
                kpts[:, 0, 0] += transform['tx']
                kpts[:, 0, 1] += transform['ty']

        # Vérification finale : retour au format torch
        kpts = kpts.reshape(-1, 2)
        return torch.from_numpy(kpts).float()

    
    def transform_keypoints(self, keypoints, image_shape):
    #Applique les mêmes transformations aux keypoints
        kpts = keypoints.clone()
        
        for transform in self.transform_history:
            if transform['type'] == 'flip_h':
                kpts[:, 0] = image_shape[1] - kpts[:, 0]
            elif transform['type'] == 'flip_v':
                kpts[:, 1] = image_shape[0] - kpts[:, 1]
            elif transform['type'] == 'rotation':
                angle = transform['angle']
                center = (image_shape[1] / 2, image_shape[0] / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                kpts = cv2.transform(kpts.numpy().reshape(-1, 1, 2), M).reshape(-1, 2)
                kpts = torch.from_numpy(kpts)
            elif transform['type'] == 'translation':
                kpts[:, 0] += transform['tx']
                kpts[:, 1] += transform['ty']
        
        self.transform_history.clear()  # Réinitialiser pour la prochaine paire
        return kpts
    """