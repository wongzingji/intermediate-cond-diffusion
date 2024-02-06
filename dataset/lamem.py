import numpy as np
import blobfile as bf
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
import os
from mpi4py import MPI

import importlib
image_datasets = importlib.import_module("image_datasets")


class MemNetDataset(Dataset):
    def __init__(
        self,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        transform=None
    ):
        super().__init__()
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.transform = transform
        
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        # if self.random_flip and random.random() < 0.5:
        #     arr = arr[:, ::-1]

        if self.transform is not None: # transform: Resize, toTensor
            image = self.transform(pil_image)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.float32)
            out_dict["name"] = self.local_images[idx]

        return image, out_dict


def load_data(
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = image_datasets._list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # # Assume classes are the first part of the filename,
        # # before an underscore.
        # class_names = [bf.basename(path).split("_")[0] for path in all_files]
        # sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        # classes = [sorted_classes[x] for x in class_names]
        score_dict = image_datasets.get_scores(data_dir)
        classes = [score_dict[os.path.basename(file)] for file in all_files]
    
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Lambda(lambda t: (t * 2) - 1),
    ])
    dataset = MemNetDataset(
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        transform=transform
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )
    while True:
        yield from loader