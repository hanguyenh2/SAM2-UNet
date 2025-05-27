import albumentations as A


def get_training_augmentation(size: int):
    """Augmentations for training"""
    train_transform = [
        # 1. Initial Resizing/Cropping
        A.OneOf(
            [
                A.Compose([
                    A.SmallestMaxSize(max_size=(int(size), int(1.1 * size))),
                    A.RandomCrop(height=size, width=size),
                ]),
                A.Compose([
                    A.LongestMaxSize(max_size=size),
                    A.PadIfNeeded(min_height=size, min_width=size),
                ])
            ],
            p=1,
        ),
        # 2. Basic Geometric
        A.SquareSymmetry(p=1),

        # 3. Color Space / Type Reduction
        A.OneOf([
            A.ToGray(),
            A.MultiplicativeNoise(per_channel=True),
        ], p=0.2),

        # 4. Color Augmentations (Brightness, Contrast, Saturation, Hue)
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.ColorJitter(),
            A.HueSaturationValue(),
            A.RandomGamma(),
        ], p=0.7),

        # 5. Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
        ], p=0.3),

        # 6. Noise
        A.OneOf([
            A.ISONoise(),
        ], p=0.3),

        # 7. Compression / Downscaling Artifacts
        A.OneOf([
            A.ImageCompression(quality_range=(20, 80)),
            A.Downscale(scale_range=(0.6, 0.8)),
        ], p=0.2),

        # 8. Normalize and convert to tensor, comment to visualize
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,
                    normalization="standard", p=1.0),
        A.ToTensorV2(transpose_mask=True),
    ]
    return A.Compose(train_transform)


def get_testing_augmentation(size: int):
    """Augmentations for testing"""
    test_transform = [
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,
                    normalization="standard", p=1.0),
        A.ToTensorV2(),
    ]
    return A.Compose(test_transform)
