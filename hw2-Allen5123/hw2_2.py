import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import numpy as np

config = {
    "device" :  "cuda" if torch.cuda.is_available() else "cpu",
    "numofclass" : 10,
    "maxT" : 1000,
    "beta_min" : 1.e-4,
    "beta_max" : 2.e-2,
    "model_path" : "hw2_2.pth", 
    "num_of_img_per_digits" : 100
}
print('Device used :', config["device"])
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
same_seeds(7777)

# modidied from https://github.com/dome272/Diffusion-Models-pytorch
class WConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None, residual = False) -> None:
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.wconv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )
    def forward(self, x):
        return F.gelu(x + self.wconv(x)) if self.residual else self.wconv(x)

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class U_Down(nn.Module):
    def __init__(self, in_channels, out_channels, sz, emb_dim=256) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            WConv(in_channels, in_channels, residual=True),
            WConv(in_channels, out_channels)
        ) #(bsz, out_channels, sz // 2, sz // 2)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
        self.attention = SelfAttention(out_channels, sz // 2)
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        ret = self.attention(x + emb)
        return ret

class U_Up(nn.Module):
    def __init__(self, in_channels, out_channels, sz, emb_dim=256) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            WConv(2*in_channels, 2*in_channels, residual=True),
            WConv(2*in_channels, out_channels)
        ) #(bsz, out_channels, 2 * sz, 2 * sz)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
        self.attention = SelfAttention(out_channels, sz * 2)
    def forward(self, x, skip_x, t):
        x = self.upsample(x) #(bsz, in_c, 2*sz, 2*sz)
        cat_x = torch.cat([skip_x, x], dim=1) #(bsz, 2*in_c, 2*sz, 2*sz)
        x = self.conv(cat_x) #(bsz, out_c, 2*sz, 2*sz)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        ret = self.attention(x + emb)
        return ret

class Unet(nn.Module):
    def __init__(self, feature_sz=28, emb_dim=256, numofclass=config["numofclass"]) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.down_channels = [(64, 128), (128, 128)] #(in_channel, out_channel)
        self.up_channels = [(128, 64), (64, 64)] #(in_channel, out_channel)
        self.label_emb = nn.Embedding(numofclass, emb_dim)
        self.conv1 = WConv(in_channels=3, out_channels=64)
        self.down_layers = nn.ModuleList([U_Down(in_channels=i, out_channels=j, sz=feature_sz // (2 ** k)) for k, (i, j) in enumerate(self.down_channels)])
        self.bottom = nn.Sequential(
            WConv(in_channels=128, out_channels=256),
            WConv(in_channels=256, out_channels=256),
            WConv(in_channels=256, out_channels=128)
        )
        self.up_layers = nn.ModuleList([U_Up(in_channels=i, out_channels=j, sz=feature_sz // 2 ** (len(self.up_channels) - k)) for k, (i, j) in enumerate(self.up_channels)])
        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=config["device"]).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc #(bsz, channels)
    
    def forward(self, x, t, label):
        #embedding time and label
        t = t.unsqueeze(-1).float()
        t = self.pos_encoding(t, self.emb_dim) #(bsz, emb_dim)
        if label is not None:
            t += self.label_emb(label) #(bsz, emb_dim)
        x1 = self.conv1(x)
        skip_x = [x1]
        cur_flow = x1
        for idx, down_layer in enumerate(self.down_layers):
            cur_flow = down_layer(cur_flow, t)
            if idx < len(self.down_layers) - 1:
                skip_x.append(cur_flow)
        cur_flow = self.bottom(cur_flow) #(bsz, 128, 7, 7)
        for idx, up_layer in enumerate(self.up_layers):
            cur_flow = up_layer(cur_flow, skip_x[-(idx+1)], t)
        return self.out(cur_flow)

def load_checkpoint(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    return checkpoint["model_state_dict"], checkpoint["optimizer_state_dict"]

if os.path.exists(config["model_path"]):
    model_checkpoint = load_checkpoint(config["model_path"], device=config["device"])[0] 
    model = Unet().to(config["device"])
    model.load_state_dict(model_checkpoint)
    print("model used : {}".format(config["model_path"]))
else:
    print("can't find model {}".format(config["model_path"]))
    sys.exit(-1)

output_dir = sys.argv[1]
if not os.path.exists(output_dir):
    print("can't find {}".format(output_dir))
    sys.exit(-1)
else:
    print("output --> {}".format(output_dir))

def CosineSchedule():
    s = 8.e-3
    t = torch.arange(config["maxT"] + 1)
    f = torch.cos((torch.pi / 2) * (t / t.shape[0] + s) / (1 + s)) ** 2
    alpha_hat = f / f[0]
    alpha_hat_shiftright = torch.roll(alpha_hat, 1, dims=0)
    alpha_hat_shiftright[0] = 1
    beta = 1 - alpha_hat / alpha_hat_shiftright
    return alpha_hat, beta

alpha_hat, beta = CosineSchedule()
cfg_weight = 5.
model.eval()
with torch.no_grad():
    for digit in range(0, 10):
        imgs =  torch.randn(config["num_of_img_per_digits"], 3, 28, 28).to(config["device"]) #(100, 3, 28, 28)
        digit_tensor = digit * torch.ones((imgs.shape[0],), dtype=torch.long).to(config["device"])
        for t in range(config["maxT"], 0, -1):
            t_tensor = t * torch.ones((imgs.shape[0], ), dtype=torch.long).to(config["device"]) #(10,)
            noise = cond_noise = model(imgs, t_tensor, digit_tensor)
            if np.random.random() > 0.65:
                uncond_noise = model(imgs, t_tensor, None)
                noise = torch.lerp(uncond_noise, cond_noise, cfg_weight)
            alpha_t = (1 - beta[t_tensor])[:, None, None, None].to(config["device"])
            alpha_t_hat = alpha_hat[t_tensor][:, None, None, None].to(config["device"])
            sigma_t = torch.sqrt(beta[t_tensor])[:, None, None, None].to(config["device"])
            imgs = (imgs - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_hat)) * noise) / torch.sqrt(alpha_t)
            if t > 1 :
                imgs += sigma_t * torch.randn_like(imgs)
        for idx, img in enumerate(imgs):
            img = (img + 1) / 2
            torchvision.utils.save_image(img, os.path.join(output_dir, "{}_{:03d}.png".format(digit, idx + 1)))
        print("[{}/1000]".format((digit + 1) * config["num_of_img_per_digits"]))