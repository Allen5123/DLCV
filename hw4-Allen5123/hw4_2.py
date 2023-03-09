import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
import sys
from torchvision import models
import pandas as pd

config = {
    'device':'cuda' if torch.cuda.is_available() else 'cpu',
    'model_pth':'hw4_2.pth',
    'bsz':64,
    'imgsz':128,
    'numofclass':65
}

val_transform = transforms.Compose([
    transforms.Resize((config['imgsz'], config['imgsz'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
])

print('Device used :', config['device'])

label2class = {'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 'Calendar': 8, 'Candles': 9, 'Chair': 10, 'Clipboards': 11, 'Computer': 12, 
    'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15, 'Drill': 16, 'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 'Glasses': 25,
    'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38,
    'Oven': 39, 'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44, 'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 'Screwdriver': 51,
    'Shelf': 52, 'Sink': 53, 'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57, 'TV': 58, 'Table': 59, 'Telephone': 60, 'ToothBrush': 61, 'Toys': 62, 'Trash_Can': 63, 'Webcam': 64}
class2label = {0: 'Alarm_Clock', 1: 'Backpack', 2: 'Batteries', 3: 'Bed', 4: 'Bike', 5: 'Bottle', 6: 'Bucket', 7: 'Calculator', 8: 'Calendar', 9: 'Candles', 10: 'Chair', 11: 'Clipboards', 12: 'Computer', 
    13: 'Couch', 14: 'Curtains', 15: 'Desk_Lamp', 16: 'Drill', 17: 'Eraser', 18: 'Exit_Sign', 19: 'Fan', 20: 'File_Cabinet', 21: 'Flipflops', 22: 'Flowers', 23: 'Folder', 24: 'Fork', 25: 'Glasses', 
    26: 'Hammer', 27: 'Helmet', 28: 'Kettle', 29: 'Keyboard', 30: 'Knives', 31: 'Lamp_Shade', 32: 'Laptop', 33: 'Marker', 34: 'Monitor', 35: 'Mop', 36: 'Mouse', 37: 'Mug', 38: 'Notebook', 
    39: 'Oven', 40: 'Pan', 41: 'Paper_Clip', 42: 'Pen', 43: 'Pencil', 44: 'Postit_Notes', 45: 'Printer', 46: 'Push_Pin', 47: 'Radio', 48: 'Refrigerator', 49: 'Ruler', 50: 'Scissors', 51: 'Screwdriver', 
    52: 'Shelf', 53: 'Sink', 54: 'Sneakers', 55: 'Soda', 56: 'Speaker', 57: 'Spoon', 58: 'TV', 59: 'Table', 60: 'Telephone', 61: 'ToothBrush', 62: 'Toys', 63: 'Trash_Can', 64: 'Webcam'}

def load_model_only(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint["model_state_dict"]

class DS(Dataset):
    def __init__(self, datapath, csvpath, transform=None) -> None:
        self.transform = transform
        self.data = []
        if os.path.exists(csvpath):
            df = pd.read_csv(csvpath)
            self.data = [(img_id, os.path.join(datapath, img_name), img_name) for img_id, img_name in zip(df['id'], df['filename'])]
        else:
            print(f"Can't find {csvpath}")
            exit(-1)
        self.len = len(self.data)

    def __getitem__(self, index):
        img_id, imgpath, imgname = self.data[index]
        img = Image.open(imgpath)
        if self.transform:
            img = self.transform(img)
        return img, img_id, imgname

    def __len__(self):
        return self.len

class ClassifierC(nn.Module):
    def __init__(self, backbonepth=None) -> None:
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        if backbonepth is not None:
            self.backbone.load_state_dict(load_model_only(backbonepth, device=config['device']))
            print(f'load backbone from {backbonepth}')
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.classifier = nn.Linear(2048, config['numofclass'])
    
    def forward(self, x):
        y = self.backbone(x).flatten(1)
        return self.classifier(y)


inputcsv, imgpth, output = sys.argv[1], sys.argv[2], sys.argv[3]
test_loader = DataLoader(DS(datapath=imgpth, csvpath=inputcsv, transform=val_transform), batch_size=config['bsz'], shuffle=False, pin_memory=True)
print(f"Input <--- {inputcsv} {imgpth} output ---> {output}")

model = ClassifierC().to(config['device'])
if os.path.exists(config['model_pth']):
    model_state = load_model_only(config['model_pth'], config['device'])
    model.load_state_dict(model_state)
    print(f"load model from {config['model_pth']}")
else:
    print(f"Can't load model from {config['model_pth']}")
    exit(-1)

datadict = {'id':[], 'filename':[], 'label':[]}
model.eval()
print("Predicting...")
with torch.no_grad():
    for idx, (img, img_id, img_name) in enumerate(test_loader):
        img = img.to(config['device'])
        logit = model(img).cpu()
        datadict['id']+=img_id.tolist()
        datadict['filename']+=img_name
        datadict['label']+=[class2label[class_] for class_ in logit.argmax(-1).tolist()]

df = pd.DataFrame(datadict)
df.to_csv(output, index=False)
print("Done")