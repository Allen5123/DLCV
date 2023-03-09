import sys
import random
import torch
import torch.nn as nn
import torchvision
import glob
import os
import numpy as np

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, c_noise, feature_dim, c_img) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.ConvTranspose2d(c_noise, feature_dim*16, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(feature_dim*16),
            nn.ReLU(True)
        )
        self.conv = nn.Sequential(
            self.dconv_bn_relu(feature_dim*16, feature_dim*8),
            self.dconv_bn_relu(feature_dim*8, feature_dim*4),
            self.dconv_bn_relu(feature_dim*4, feature_dim*2)
        )
        self.last = nn.Sequential(
            nn.ConvTranspose2d(feature_dim*2, c_img, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh() 
        )
        self.apply(weights_init)

    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        x1 = self.project(x)
        x2 = self.conv(x1)
        return self.last(x2)

config = {
    "device" :  "cuda" if torch.cuda.is_available() else "cpu",
    "c_noise" : 100,
    "feature_dim" : 64,
    "G_checkpoint": "hw2_1_G.pth"
}
print(config["device"])
same_seeds(7777)
if os.path.exists(config["G_checkpoint"]):
    G_checkpoint = torch.load(config["G_checkpoint"], map_location=torch.device(config["device"]))
    print("model used : {}".format(config["G_checkpoint"]))
else:
    print("can't find model {}".format(config["G_checkpoint"]))
    sys.exit(-1)

output_dir = sys.argv[1]
if not os.path.exists(output_dir):
    print("can't find {}".format(output_dir))
    sys.exit(-1)
else:
    print("output --> {}".format(output_dir))

G = Generator(config["c_noise"], config["feature_dim"], 3).to(config["device"])
G.load_state_dict(G_checkpoint["model_state_dict"])
G.eval()
G.to(config["device"])
numofgenerate = 1000
with torch.no_grad():
    random_z = torch.randn(numofgenerate, config["c_noise"], 1, 1, device=config["device"])
    print("Generating...")
    fake_imgs = G(random_z)
    for (idx, fake_img) in enumerate(fake_imgs):
        fake_img = (fake_img + 1) / 2
        torchvision.utils.save_image(fake_img, fp=output_dir + "{}.png".format(idx + 1))
        if (idx + 1) % 200 == 0:
            print("[{}/{}]".format(idx + 1, numofgenerate))