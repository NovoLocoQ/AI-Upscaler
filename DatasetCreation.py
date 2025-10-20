from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class SRDataset(Dataset):
    def __init__(self, dirHr, transform=None, mode=0):
        self.dirHr = dirHr
        self.mode = mode
        self.filenames = sorted(os.listdir(dirHr))
        self.RdnCrp = transforms.RandomCrop(256)
        self.CntCrp = transforms.CenterCrop(256)
        self.ToTen = transforms.ToTensor()
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        hr_path = os.path.join(self.dirHr, self.filenames[index])
        hr_image = Image.open(hr_path).convert("RGB")
        
        if self.mode == 0:
            hr_image = self.RdnCrp(hr_image)
        else:
            hr_image = self.CntCrp(hr_image)
        
        lr_image = hr_image.resize((256//4, 256//4), resample=Image.BICUBIC)
        
        hr_tensor, lr_tensor = self.ToTen(hr_image), self.ToTen(lr_image)
        return lr_tensor, hr_tensor
