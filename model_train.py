import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from DataLoading import Train_Load,Test_Load
from SRmodel import UNETPixelShuffle
from trainandtestfun import test_step,train_step
from Timing_fun import print_train_time, timer
from tqdm.auto import tqdm
torch.manual_seed(42)
model=UNETPixelShuffle().to("cuda")
loss_fn=nn.MSELoss()
optimizer=torch.optim.Adam(params=model.parameters(),lr=1e-4)
timer_start=timer()
epochs=100
for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch}\n-------")
  train_step(model=model,data_loader=Train_Load,loss_fn=loss_fn,optimizer=optimizer)
  test_step(model=model,data_loader=Test_Load,loss_fn=loss_fn)
torch.save(model.state_dict(),'SuperResolutionV2.pth')
timer_stop=timer()
time=print_train_time(start=timer_start,end=timer_stop,device="cpu")
print(time)
