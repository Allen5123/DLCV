import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
import os
import numpy as np
from PIL import Image
import pandas as pd
config = {
    "device" :  "cuda" if torch.cuda.is_available() else "cpu",
    "numofclass" : 10,
    "svhn_F_checkpoint" : "hw2_3_mnistm_svhn_F.pth",
    "svhn_C_checkpoint" : "hw2_3_mnistm_svhn_C.pth",
    "usps_F_checkpoint" : "hw2_3_mnistm_usps_F.pth",
    "usps_C_checkpoint" : "hw2_3_mnistm_usps_C.pth"
}
val_transform = transforms.Compose([
    transforms.ToTensor(),
])
print('Device used :', config["device"])

class SVHN_FeatureExtractor(nn.Module):
    def __init__(self, channel=3):
        super(SVHN_FeatureExtractor, self).__init__()
        self.conv1 = self.block(dim_in=channel, dim_out=64) #(bsz, 64, 14, 14)
        self.conv2 = self.block(dim_in=64, dim_out=128) #(bsz, 128, 7, 7)
        self.conv3 = self.block(dim_in=128, dim_out=256, pad=0, maxpool=False) #(bsz, 256, 5, 5)
        self.conv4 = self.block(dim_in=256, dim_out=256, pad=0, maxpool=False) #(bsz, 256, 3, 3)
        self.conv5 = self.block(dim_in=256, dim_out=512, pad=0, maxpool=False) #(bsz, 512, 1, 1)
    def block(self, dim_in, dim_out, ksz=3, stride_=1, pad=1, maxpool=True):
        if maxpool:
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=pad),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(True),
                nn.MaxPool2d(2)
            )
        return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=pad),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(True)
            )    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return x5.flatten(1)

class SVHN_LabelPredictor(nn.Module):
    def __init__(self):
        super(SVHN_LabelPredictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, config["numofclass"])
        )
    def forward(self, h):
        c = self.layer(h)
        return c

class USPS_FeatureExtractor(nn.Module):
    def __init__(self, channel=1):
        super(USPS_FeatureExtractor, self).__init__()
        self.conv1 = self.block(dim_in=channel, dim_out=64) #(bsz, 64, 14, 14)
        self.conv2 = self.block(dim_in=64, dim_out=128) #(bsz, 128, 7, 7)
        self.conv3 = self.block(dim_in=128, dim_out=256, pad=0, maxpool=False) #(bsz, 256, 5, 5)
        self.conv4 = self.block(dim_in=256, dim_out=256, pad=0, maxpool=False) #(bsz, 256, 3, 3)
        self.conv5 = self.block(dim_in=256, dim_out=512, pad=0, maxpool=False) #(bsz, 512, 1, 1)
    def block(self, dim_in, dim_out, ksz=3, stride_=1, pad=1, maxpool=True):
        if maxpool:
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=pad),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(True),
                nn.MaxPool2d(2)
            )
        return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=pad),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(True)
            )    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return x5.flatten(1)

class USPS_LabelPredictor(nn.Module):
    def __init__(self):
        super(USPS_LabelPredictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, config["numofclass"])
        )
    def forward(self, h):
        c = self.layer(h)
        return c

class DigitDataset(Dataset):
    def __init__(self, datapath, transform=None) -> None:
        self.data = []
        if os.path.exists(datapath):
            filenames = glob.glob(os.path.join(datapath, "*"))
            for filename in filenames:
                self.data.append(filename)
        else:
            print("Can't find {}".format(datapath))
            exit(-1)
        self.data.sort()
        self.len = len(self.data)
        self.transform = transform
    def __getitem__(self, index):
        img_fn = self.data[index]
        img = Image.open(img_fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, os.path.split(img_fn)[-1] 
    def __len__(self):
        return self.len

def load_checkpoint(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    return checkpoint["model_state_dict"], checkpoint["optimizer_state_dict"]

data_path, output_path = sys.argv[1], sys.argv[2]
if not os.path.exists(data_path):
    print("Unknown data path {}".format(data_path))
    exit(-1)
print("input <--- {}\noutput ---> {}".format(data_path, output_path))
test_loader = DataLoader(DigitDataset(datapath=data_path, transform=val_transform))
output = {"image_name" : [], "label" : []}
if next(iter(test_loader))[0].shape[1] == 3:
    if os.path.exists(config["svhn_F_checkpoint"]) and os.path.exists(config["svhn_C_checkpoint"]):
        print("Model used : {} {}".format(config["svhn_F_checkpoint"], config["svhn_C_checkpoint"]))
        F, C = SVHN_FeatureExtractor().to(config["device"]), SVHN_LabelPredictor().to(config["device"])
        F_checkpoint, C_checkpoint = load_checkpoint(config["svhn_F_checkpoint"], device=config["device"])[0], load_checkpoint(config["svhn_C_checkpoint"], device=config["device"])[0]
        F.load_state_dict(F_checkpoint), C.load_state_dict(C_checkpoint)
    else:
        print("Error : can't found model {}".format(config["svhn_C_checkpoint"], config["svhn_F_checkpoint"]))
        exit(-1)
elif next(iter(test_loader))[0].shape[1] == 1:
    if os.path.exists(config["usps_F_checkpoint"]) and os.path.exists(config["usps_C_checkpoint"]):
        print("Model used : {} {}".format(config["usps_F_checkpoint"], config["usps_C_checkpoint"]))
        F, C = USPS_FeatureExtractor().to(config["device"]), USPS_LabelPredictor().to(config["device"])
        F_checkpoint, C_checkpoint = load_checkpoint(config["usps_F_checkpoint"], device=config["device"])[0], load_checkpoint(config["usps_C_checkpoint"], device=config["device"])[0]
        F.load_state_dict(F_checkpoint), C.load_state_dict(C_checkpoint)
    else:
        print("Error : can't found model {}".format(config["usps_C_checkpoint"], config["usps_F_checkpoint"]))
        exit(-1)
else:
    print("Error : unknown image channel {}".format(next(iter(test_loader))[0].shape[1]))
    exit(-1)
F.eval(), C.eval()
with torch.no_grad():
    for idx, (test_img, fn) in enumerate(test_loader):
        y = C(F(test_img.to(config["device"]))).cpu()
        predict = y.argmax(1).item()
        output["image_name"].append(fn[0]), output["label"].append(predict)
        if (idx + 1) % (len(test_loader.dataset) // 5) == 0:
            print("[{}/{}]".format(idx + 1, len(test_loader.dataset)))
print("[{}/{}] finish".format(idx + 1, len(test_loader.dataset)))
df = pd.DataFrame(output)
df.to_csv(output_path, index=False)