import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import glob
import os
import numpy as np
from PIL import Image

class TestDataSet(Dataset):
    def __init__(self, dirpath, transform=None) -> None:
        self.data = []
        self.transform = transform
        if os.path.exists(dirpath):
            filenames = glob.glob(os.path.join(dirpath, "*.jpg"))
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

class Deeplabv3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=numofclass)
    def forward(self, x):
        y = self.deeplabv3(x)['out']
        return y

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device used:', device)
model_path = "deeplabv3.pth"
numofclass = 7

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
else:
    print("can't find model {}".format(model_path))
    sys.exit(-1)

if len(sys.argv) < 3:
    print("Miss argument {}".format(sys.argv))
    sys.exit(-1)

test_tfm = transforms.Compose([
    transforms.PILToTensor()
])
test_path, output_path = sys.argv[1], sys.argv[2]
test_dataloader = DataLoader(TestDataSet(test_path, transform=test_tfm))
numoftest = len(test_dataloader.dataset)
model = Deeplabv3()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)
# for parameters in model.state_dict():
#     print(parameters)

printstatus = (numoftest * np.linspace(0., 1.0, 10)).astype(np.int32)
cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0]
}

def ClassToRGB(class_img):
    class_img = np.array(class_img)
    m, n = class_img.shape
    rgb = np.zeros((m, n, 3), dtype=np.uint8)
    for i in range(numofclass):
        rgb[class_img == i] = cls_color[i]
    return rgb

with torch.no_grad():
    for idx, (img, filename) in enumerate(test_dataloader):
        if idx in printstatus:
            print("[{}/{}] {:.2%}".format(idx+1, numoftest, (idx+1)/numoftest)) 
        img = img.to(device, dtype=torch.float32)
        output = model(img)
        predict = Image.fromarray(ClassToRGB(torch.squeeze(output, dim=0).argmax(dim=0).detach().cpu().numpy()), 'RGB')
        predict.save(os.path.join(output_path, filename[0].split('.')[0]+'.png'))