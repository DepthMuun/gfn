import zipfile
import torch
from pathlib import Path
from typing import List, Optional, Tuple
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

def _resolve_dataset_root(root_dir: Path) -> Path:
    direct = root_dir / "train" / "images"
    if direct.exists():
        return root_dir
    nested = root_dir / "dataset" / "train" / "images"
    if nested.exists():
        return root_dir / "dataset"
    return root_dir


def _has_images(img_dir: Path) -> bool:
    for ext in (".jpg", ".jpeg", ".png"):
        if next(img_dir.glob(f"*{ext}"), None) is not None:
            return True
    return False


def _list_image_files(img_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in (".jpg", ".jpeg", ".png"):
        files.extend(img_dir.glob(f"*{ext}"))
    return sorted(files)


def _select_largest_box(label_path: Path, min_box_area: float) -> Optional[Tuple[float, float, float, float]]:
    if not label_path.exists():
        return None
    best_box = None
    best_area = 0.0
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x, y, w, h = parts
            w = float(w)
            h = float(h)
            area = w * h
            if area < min_box_area:
                continue
            if area > best_area:
                best_area = area
                best_box = (float(x), float(y), w, h)
    return best_box


class PureSeraphimDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        img_size=128,
        transform=None,
        min_box_area: float = 0.000625,
        max_samples: Optional[int] = None,
        include_empty: bool = True,
    ):
        self.root_dir = _resolve_dataset_root(Path(root_dir))
        self.split = split
        self.img_size = img_size
        self.img_dir = self.root_dir / split / "images"
        self.label_dir = self.root_dir / split / "labels"
        self.min_box_area = min_box_area
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Missing images directory: {self.img_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Missing labels directory: {self.label_dir}")

        img_files = _list_image_files(self.img_dir)
        if max_samples is not None:
            img_files = img_files[:max_samples]

        if not img_files:
            raise RuntimeError(f"No usable samples found in {self.img_dir}")

        if not include_empty:
            filtered: List[Path] = []
            for img_path in img_files:
                label_path = self.label_dir / f"{img_path.stem}.txt"
                box = _select_largest_box(label_path, min_box_area=min_box_area)
                if box is not None:
                    filtered.append(img_path)
            img_files = filtered
            if not img_files:
                raise RuntimeError(f"No labeled samples found in {self.img_dir}")

        self.img_files = img_files

        if transform:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"
        box = _select_largest_box(label_path, min_box_area=self.min_box_area)
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        x = img_tensor.flatten()

        target = torch.zeros(5, dtype=torch.float32)
        if box is not None:
            target[0] = 1.0
            target[1:] = torch.tensor(box, dtype=torch.float32)

        return x, target

def download_and_extract(
    dataset_id="lgrzybowski/seraphim-drone-detection-dataset",
    local_dir="D:/ASAS/datasets/seraphim",
    force_download: bool = False,
):
    root = Path(local_dir)
    repo_path = root
    data_root = _resolve_dataset_root(root)

    needs_download = force_download or not root.exists()
    if not needs_download:
        img_dir = data_root / "train" / "images"
        if not img_dir.exists() or not _has_images(img_dir):
            needs_download = True

    if needs_download:
        repo_path = Path(snapshot_download(repo_id=dataset_id, repo_type="dataset", local_dir=root))
        data_root = _resolve_dataset_root(repo_path)

    zip_files = list(data_root.rglob("*.zip"))
    if zip_files:
        for zip_path in zip_files:
            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(zip_path.parent)
                zip_path.unlink()
            except Exception as e:
                print(f"Error extracting {zip_path}: {e}")

    img_dir = data_root / "train" / "images"
    if not img_dir.exists() or not _has_images(img_dir):
        raise FileNotFoundError(f"No images found after extraction in {img_dir}")

    return data_root

def get_pure_dataloader(
    root_dir,
    split="train",
    batch_size=8,
    img_size=128,
    shuffle=True,
    min_box_area: float = 0.000625,
    max_samples: Optional[int] = None,
    include_empty: bool = True,
):
    dataset = PureSeraphimDataset(
        root_dir,
        split=split,
        img_size=img_size,
        min_box_area=min_box_area,
        max_samples=max_samples,
        include_empty=include_empty,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader

if __name__ == "__main__":
    path = download_and_extract()
    loader = get_pure_dataloader(path, batch_size=2, img_size=128, max_samples=100000000000)
    x, y = next(iter(loader))
    print(f"Image Vector (dim={x.shape[1]}): {x.shape}")
    print(f"Target (obj, x, y, w, h): {y}")
