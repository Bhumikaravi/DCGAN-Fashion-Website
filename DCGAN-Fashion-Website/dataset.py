from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


class SyntheticDataset(Dataset):
    def __init__(self, root="data/synthetic"):
        self.files = sorted(Path(root).glob("*.png"))
        if not self.files:
            raise FileNotFoundError(
                "Synthetic images not found. Run: python generate_synthetic.py"
            )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("L")
        return self.transform(image), 0


def get_dataloader(source="Fashion-MNIST", batch_size=32, limit=1000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if source == "Fashion-MNIST":
        dataset = datasets.FashionMNIST(
            root="data", train=True, download=True, transform=transform
        )
        if limit and limit < len(dataset):
            generator = torch.Generator().manual_seed(42)
            dataset = Subset(dataset, range(limit))
    elif source == "Synthetic":
        dataset = SyntheticDataset()
    else:
        raise ValueError(f"Unknown dataset source: {source}")

    if len(dataset) < batch_size:
        raise ValueError("Dataset must contain at least one full batch.")

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True
    )
