import os
import clip
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import glob
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import sys

config = {
    "device":"cuda" if torch.cuda.is_available() else "cpu",
    "bsz":64
}
print('Device used :', config["device"])

class DS(Dataset):
    def __init__(self, dirpath, transform=None) -> None:
        self.data = []
        self.transform = transform
        if os.path.exists(dirpath):
            filenames = glob.glob(os.path.join(dirpath, "*.png"))
            for filename in filenames:
                self.data.append(filename)
        else:
            print("Can't open {}".format(dirpath))
            exit(-1)
        self.len = len(self.data)
    def __getitem__(self, index):
        imgpath = self.data[index]
        img = Image.open(imgpath)
        if self.transform:
            img = self.transform(img)
        return img, os.path.split(imgpath)[-1]
    def __len__(self):
        return self.len

datapath, datajson, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
if not os.path.exists(datapath):
    print("Can't open {}".format(datapath))
    exit(-1)
if not os.path.exists(datajson):
    print("Can't open {}".format(datajson))
    exit(-1)
print("input <--- {} {}\noutput ---> {}".format(datapath, datajson, output_path))

with open(datajson) as f:
    id2label = json.load(f)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {val}?") for key, val in id2label.items()]).to(config["device"])
model, preprocess = clip.load('ViT-B/32', config["device"])
test_dataset = DS(dirpath=datapath, transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=config["bsz"], pin_memory=True)

output = {"filename":[], "label":[]}
with torch.no_grad():
    print("Start")
    for i, (img, name) in enumerate(test_loader):
        img = img.to(config["device"])
        logits_per_images, logit_per_text = model(img, text_inputs)
        pred = logits_per_images.softmax(dim=-1).argmax(dim=-1)
        output["filename"] += name
        output["label"] += pred.cpu().tolist()
        if (i + 1) % 5 == 0:
            print("[{}/{}]".format((i + 1) * config["bsz"], len(test_loader.dataset)))
    print("[{}/{}]".format(len(test_loader.dataset), len(test_loader.dataset)))

df = pd.DataFrame(output)
df.to_csv(output_path, index=False)