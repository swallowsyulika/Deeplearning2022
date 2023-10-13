# %%
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class EMnistDataset(Dataset):
    def __init__(self, root_dir, label_dir=None, specific="", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = np.load(root_dir)
        self.labels = np.load(label_dir) if label_dir is not None else None
        self.specific = specific
        
        if self.labels is not None and len(self.data) != len(self.labels):
            assert "data and label not same size"
        
        if len(self.specific) and self.labels is not None:
            specific_data = []
            specific_word = [ord(x) for x in specific]
            for img, label in zip(self.data, self.labels):
                if label in specific_word:
                    specific_data.append(img)
            self.data = np.array(specific_data)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
     
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)

        return image
    
if __name__ == "__main__":
    
    E = EMnistDataset("./EMNIST/ndy/emnist-byclass-test-images.npy")
    print(E[0])
    print(len(E))

    Es = EMnistDataset("./EMNIST/ndy/emnist-byclass-test-images.npy", "./EMNIST/ndy/emnist-byclass-test-labels.npy", "ABC")
    print(Es[0])
    print(len(Es))