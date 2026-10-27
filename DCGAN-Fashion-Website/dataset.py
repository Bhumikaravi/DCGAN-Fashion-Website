
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


# DCGAN expects grayscale 28x28 images normalized to [-1, 1].
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def find_synthetic_images():
    """Find generated images in common project directories."""
    root = Path(__file__).resolve().parent

    folders = [
        root / "synthetic_images",
        root / "generated_images",
        root / "synthetic",
        root / "assets" / "synthetic_images",
        root / "assests" / "synthetic_images",
        root / "data" / "synthetic_images",
    ]

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    for folder in folders:
        if folder.is_dir():
            files = sorted(
                file for file in folder.rglob("*")
                if file.is_file()
                and file.suffix.lower() in extensions
            )
            if files:
                return files

    return []


class SyntheticDataset(Dataset):
    """Load existing synthetic images."""

    def __init__(self, limit=100):
        self.files = find_synthetic_images()

        if not self.files:
            raise FileNotFoundError(
                "Synthetic images not found. Select Fashion-MNIST "
                "or add generated images to synthetic_images/."
            )

        if limit is not None:
            self.files = self.files[:int(limit)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]

        with Image.open(path) as image:
            image = image.convert("L")
            image = IMAGE_TRANSFORM(image)

        return image, 0


def get_dataloader(
    source="Fashion-MNIST",
    batch_size=16,
    limit=100,
):
    """Return a DataLoader for Fashion-MNIST or synthetic images."""

    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    root = Path(__file__).resolve().parent

    if str(source).strip().lower() in {
        "fashion-mnist",
        "fashionmnist",
        "fashion mnist",
    }:
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        dataset = datasets.FashionMNIST(
            root=str(data_dir),
            train=True,
            download=True,
            transform=IMAGE_TRANSFORM,
        )

        if limit is not None:
            dataset = torch.utils.data.Subset(
                dataset,
                range(min(int(limit), len(dataset))),
            )

    elif str(source).strip().lower() in {
        "synthetic",
        "synthetic images",
    }:
        dataset = SyntheticDataset(limit=limit)

    else:
        raise ValueError(
            f"Unknown dataset: {source}. "
            "Choose Fashion-MNIST or Synthetic."
        )

    if len(dataset) == 0:
        raise ValueError("The dataset contains no images.")

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
