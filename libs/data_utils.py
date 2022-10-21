import os
import torch
from typing import Union
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import *
import torchvision.transforms.functional as f

def readImage(filepath:str):
    assert os.path.exists(filepath), f"{filepath} is not exists"
    try:
        pilObj = Image.open(filepath)
        return pilObj
    except:
        raise ValueError(f"Failed to open {filepath}")

def checkSize(filepath:str, min_value:int):
    w, h = readImage(filepath).size
    if w >= min_value and h >=min_value:
        return True
    else:
        return False

def addGaussianNoise(x:torch.Tensor, i:float=0.2):
    return x + torch.randn_like(x) * i

class baseDataset(Dataset):
    def __init__(self, root:str, min_value:int=224):
        super().__init__()
        assert len(os.listdir(root)) > 0, f"{root} is empty"
        self.filelist = [os.path.join(root, x) for x in os.listdir(root) \
            if checkSize(os.path.join(root, x), min_value)]
        assert len(self.filelist) > 0, f"no valid img in {root}, check min_value={min_value}"
        self.to_tensor = Compose([
            ToTensor(),
            Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filelist)
    
    def getImage(self, idx:int):
        return readImage(self.filelist[idx])
    

class trainDataset(baseDataset):
    def __init__(self, root:str, size:Union[int, tuple]):
        if isinstance(size, int):
            min_value = size
            size = (size, size)
        elif isinstance(size, tuple):
            a, b = size
            min_value = min(a, b)
        super().__init__(root=root, min_value=min_value)
        self.size = size
        self.random_crop = RandomCrop(size=size)
    
    def __getitem__(self, idx:int):
        pilObj = self.getImage(idx)
        randomCropped = self.random_crop(pilObj)
        origTensor = self.to_tensor(randomCropped)
        
        #blurred = f.gaussian_blur(randomCropped, kernel_size=(3, 3))
        noisedTensor = addGaussianNoise(origTensor)
        
        return noisedTensor, origTensor
    
class testDataset(baseDataset):
    def __init__(self, root:str, size:Union[int, tuple]):
        if isinstance(size, int):
            min_value = size
            size = (size, size)
        elif isinstance(size, tuple):
            a, b = size
            min_value = min(a, b)
        super().__init__(root=root, min_value=min_value)
        self.size = size
        self.center_crop = CenterCrop(size=size)

    def __getitem__(self, idx:int):
        pilObj = self.getImage(idx)
        centerCropped = self.center_crop(pilObj)
        testTensor = self.to_tensor(centerCropped)
        return testTensor


# ToDo
# Function for Visualizer

if __name__ == "__main__":
    myTrainDataset = trainDataset(root="./libs/testdata/debug", size=(300, 300))
