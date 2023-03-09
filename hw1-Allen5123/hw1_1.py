import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image

class TestDataSet(Dataset):
    def __init__(self, dirpath, transform=None) -> None:
        self.data = []
        self.transform = transform
        if os.path.exists(dirpath):
            filenames = glob.glob(os.path.join(dirpath, "*.png"))
            for filename in filenames:
                self.data.append(filename)
            self.len = len(self.data)
        else:
            print("Can't find {}".format(dirpath))

    def __getitem__(self, index):
        image_fn = self.data[index]
        image = Image.open(image_fn)
        if self.transform is not None:
            image = self.transform(image)
        return (image, os.path.split(image_fn)[-1]) # data, filename

    def __len__(self):
        return self.len

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device used:', device)
model_path = "resnext50.pth"
numofclass = 50
test_tfm = transforms.Compose([
    transforms.Resize((224,224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor()
])

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    # print(checkpoint)
else:
    print("can't find model {}".format(model_path))
    sys.exit(-1)

if len(sys.argv) < 3:
    print("Miss argument {}".format(sys.argv))
    sys.exit(-1)

test_path, output_path = sys.argv[1], sys.argv[2]
test_dataloader = DataLoader(TestDataSet(test_path, test_tfm))
# print(test_dataloader.dataset[-1])
numoftest = len(test_dataloader.dataset)
model = torchvision.models.resnext50_32x4d()
model.fc = nn.Linear(model.fc.in_features, numofclass)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)
# for parameters in model.state_dict():
#     print(parameters)

printstatus = (numoftest * np.linspace(0., 1.0, 10)).astype(np.int32)
output_data = {"filename":[], "label":[]}
with torch.no_grad():
    for idx, (img, filename) in enumerate(test_dataloader):
        if idx in printstatus:
            print("[{}/{}] {:.2%}".format(idx+1, numoftest, (idx+1)/numoftest)) 
        img = img.to(device)
        output = model(img)
        predict = output.argmax(-1).item()
        # print("{}, {}".format(filename[0], predict))
        output_data["filename"].append(filename[0])
        output_data["label"].append(predict)

df = pd.DataFrame(output_data)
df.to_csv(output_path, index=False)