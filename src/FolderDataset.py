import os
from torch.utils.data import Dataset
from PIL import Image

class FolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform=transform
        self.images_paths = []
        self.labels = []
        self.classes = os.listdir(root)
        for i in self.classes:
            class_images = os.listdir(root + "/" + i)
            for j in class_images:
                self.images_paths.append(root + "/" + i + "/" + j)
                self.labels.append(self.classes.index(i))

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, item):
        img = Image.open(self.images_paths[item]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[item]


if __name__ == "__main__":
    data = FolderDataset("/home/belal/projects/Multi-Head CNN/data/eye/Train")
    X, y = data[280]
    print(y)
    X.show()
