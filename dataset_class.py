from matplotlib import transforms
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, PILToTensor
from PIL import Image
import numpy as np


class FacialKeypoinsDataset(Dataset):
    def __init__(self, X_frame, y_frame):
        self.imgs = X_frame
        self.labels = y_frame

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img_pil = Image.fromarray(img)

        toTens = PILToTensor()

        img = toTens(img_pil)
        lables = self.labels[idx]
        return img, np.float32(lables)