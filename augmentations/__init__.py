import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def various_augs(aug_type: int, aug_prob=0.75, img_size=512):
    if aug_type == 0:
        transform_train = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=aug_prob),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=aug_prob),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=aug_prob),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=aug_prob),
            A.Resize(img_size, img_size),
            A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=aug_prob),    
            A.Normalize(mean=0.5, std=0.5)
        ])
        transform_eval = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=0.5, std=0.5)
        ])
    elif aug_type == 1:
        transform_train = A.ReplayCompose([
            A.Resize(img_size, img_size),
            A.Perspective(p=aug_prob),
            A.Rotate(p=aug_prob, limit=(-25, 25)),
            #A.Normalize(mean=0.5, std=0.5),
        ])
        transform_eval = A.Compose([
            A.Resize(img_size, img_size),
            #A.Normalize(mean=0.5, std=0.5),
        ])
    elif aug_type == 2:
        transform_train = A.Compose([
            A.ShiftScaleRotate(scale_limit=0.2, shift_limit=0.1, rotate_limit=45, p=aug_prob),
            A.OneOf([
                A.RandomGamma(gamma_limit=(50, 150), always_apply=True),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, always_apply=True),
            ], p=0.25),
            A.OneOf([
                A.MotionBlur(always_apply=True),
                A.GaussianBlur(always_apply=True),
            ], p=0.25),
            A.OneOf([
                A.ElasticTransform(
                    alpha=1,
                    sigma=5,
                    alpha_affine=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    always_apply=True,
                )], p=0.5),
            A.Resize(img_size, img_size),
            A.Normalize(mean=0.5, std=0.5),
        ])
        transform_eval = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=0.5, std=0.5),
        ])
    return transform_train, transform_eval