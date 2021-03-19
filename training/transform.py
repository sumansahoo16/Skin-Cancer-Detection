import albumentations

def get_transforms():
    
    transforms_train = albumentations.Compose([    
                           albumentations.HorizontalFlip(p=0.5),
                           albumentations.VerticalFlip(p=0.5),
                           albumentations.Cutout(num_holes = 32, max_h_size = 16, max_w_size = 16, p = 0.5),
                           albumentations.Normalize(),
                           #albumentations.pytorch.transforms.ToTensorV2(),
                          ])

    transforms_val = albumentations.Compose([
                           albumentations.Normalize(),
                           #albumentations.pytorch.transforms.ToTensorV2(),
                          ])
    
    
    """
    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(256, 256),
        albumentations.Cutout(max_h_size=int(256 * 0.375), max_w_size=int(256 * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(256, 256),
        albumentations.Normalize()
    ])"""

    return transforms_train, transforms_val