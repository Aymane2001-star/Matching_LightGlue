import torch
import random
import torchvision.transforms.functional as TF

class SynchronizedAugmentations:
    def __init__(self, config):
        self.config = config

    def __call__(self, img1, img2, kpts1, kpts2):
        # Rotation 90°
        if self.config.get("random_rotation_90", False):
            k = random.randint(0, 3)
            img1 = TF.rotate(img1, angle=90 * k)
            img2 = TF.rotate(img2, angle=90 * k)
            for _ in range(k):
                kpts1 = self._rotate_90(kpts1, img1.shape[-2:])
                kpts2 = self._rotate_90(kpts2, img2.shape[-2:])

        # Flip horizontal
        if self.config.get("horizontal_flip", False) and random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            kpts1[:, 0] = img1.shape[-1] - kpts1[:, 0]
            kpts2[:, 0] = img2.shape[-1] - kpts2[:, 0]

        # Flip vertical
        if self.config.get("vertical_flip", False) and random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            kpts1[:, 1] = img1.shape[-2] - kpts1[:, 1]
            kpts2[:, 1] = img2.shape[-2] - kpts2[:, 1]

        # Zoom (scaling)
        if self.config.get("random_zoom", False):
            zoom = random.uniform(*self.config.get("zoom_range", [0.9, 1.1]))
            img1 = TF.resize(img1, [int(img1.shape[-2]*zoom), int(img1.shape[-1]*zoom)])
            img2 = TF.resize(img2, [int(img2.shape[-2]*zoom), int(img2.shape[-1]*zoom)])
            kpts1 *= zoom
            kpts2 *= zoom

        # Translation
        if self.config.get("random_translation", False):
            max_t = self.config.get("max_translation", 50)
            tx, ty = random.randint(-max_t, max_t), random.randint(-max_t, max_t)
            img1 = TF.affine(img1, angle=0, translate=(tx, ty), scale=1.0, shear=[0, 0])
            img2 = TF.affine(img2, angle=0, translate=(tx, ty), scale=1.0, shear=[0, 0])
            kpts1[:, 0] += tx
            kpts1[:, 1] += ty
            kpts2[:, 0] += tx
            kpts2[:, 1] += ty

        return img1, img2, kpts1, kpts2

    def _rotate_90(self, kpts, shape):
        # shape = (H, W)
        kpts = kpts.clone()
        x, y = kpts[:, 0], kpts[:, 1]
        kpts[:, 0] = y
        kpts[:, 1] = shape[1] - x
        return kpts