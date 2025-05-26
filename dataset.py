import os

import cv2
from torch.utils.data import Dataset as BaseDataset, DataLoader

from augmentations import get_training_augmentation, get_testing_augmentation
from utils import visualize_mask


class Dataset(BaseDataset):
    def __init__(self, images_dir: str, masks_dir: str, size: int, training: bool = True):
        self.images = [images_dir + f for f in os.listdir(images_dir) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.masks = [masks_dir + f for f in os.listdir(masks_dir) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)
        if training:
            self.augmentation = get_training_augmentation(size)
        else:
            self.augmentation = get_testing_augmentation(size)

    def __getitem__(self, i):
        # Get image's name
        name = self.images[i].split('/')[-1]

        # Read the image
        image = cv2.imread(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Read the mask in grayscale mode
        mask = cv2.imread(self.masks[i], 0)

        # Augmentation
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # Convert image to Tensor
        image = image.transpose(2, 0, 1)

        return image, mask, name

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    images_dir = "data/images/"
    masks_dir = "data/masks/"
    size = 1152
    train_dataset = Dataset(
        images_dir,
        masks_dir,
        size=size,
        training=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    for i in range(20):
        train_features, train_labels, name = next(iter(train_loader))
        print(name)
        img = train_features[0].squeeze().numpy().transpose(1, 2, 0)
        mask = train_labels[0].squeeze().numpy()
        colored_mask = visualize_mask(
            img,
            mask,
            5,
        )
        cv2.imwrite(f"{i}.png", img)
        cv2.imwrite(f"{i}_label.png", colored_mask)
