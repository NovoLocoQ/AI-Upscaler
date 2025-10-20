from DatasetCreation import SRDataset
from torch.utils.data.dataloader import DataLoader

Train_dataset=SRDataset("Images/train_images",mode=0)
Test_dataset=SRDataset("Images/test_images",mode=1)

Train_Load = DataLoader(Train_dataset, batch_size=32, num_workers=0, shuffle=True, 
                       pin_memory=True)
Test_Load = DataLoader(Test_dataset, batch_size=32, num_workers=0, shuffle=False,
                      pin_memory=True)