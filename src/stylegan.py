import os
import math
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torch.cuda.amp import autocast, GradScaler


# -------------------- Kaggle-specific paths --------------------
input_data_directory = '/kaggle/input/coin-dataset/data'
input_images_directory = '/kaggle/input/coin-dataset/images'
working_directory = '/kaggle/working'
training_directory = os.path.join(working_directory, 'results', 'training_weights')
generated_images = os.path.join(working_directory, 'results', 'generated_images')


# Create output directories
os.makedirs(training_directory, exist_ok=True)
os.makedirs(generated_images, exist_ok=True)

# -------------------- Hyperparameters --------------------
latent_dim    = 512
img_size      = 256
img_channels  = 3
batch_size    = 8
learning_rate = 1e-4
num_epochs    = 50
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------- Dataset --------------------
class CoinDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['images']
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# Load train.csv produced by CycleGAN preprocessing
train_csv = os.path.join(working_directory, 'Train.csv')
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
dataset = CoinDataset(train_csv, transform=transform)
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


# -------------------- Model Components --------------------
class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, latent_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features)
        self.style_scale = nn.Linear(latent_dim, num_features)
        self.style_shift = nn.Linear(latent_dim, num_features)

    def forward(self, x, w):
        scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        shift = self.style_shift(w).unsqueeze(2).unsqueeze(3)
        x = self.norm(x)
        return x * (scale + 1) + shift

class StyledConvBlock(nn.Module):
    def __init__(self, in_c, out_c, latent_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)
        self.noise = nn.Parameter(torch.zeros(1, out_c, 1, 1))
        self.adain = AdaptiveInstanceNorm2d(out_c, latent_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        x = self.conv(x)
        b, c, h, w_ = x.shape
        noise = torch.randn(b, 1, h, w_, device=x.device)
        x = x + self.noise * noise
        x = self.adain(x, w)
        x = self.act(x)
        return x

class StyleGANGenerator(nn.Module):
    def __init__(self, latent_dim, img_channels, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        layers = [PixelNorm()]
        for _ in range(8):
            layers += [nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(0.2)]
        self.mapping = nn.Sequential(*layers)
        self.constant_input = nn.Parameter(torch.randn(1, latent_dim, 4, 4))
        resolutions = [4, 8, 16, 32, 64, 128, 256]
        in_channels = latent_dim
        self.blocks, self.to_rgb = nn.ModuleList(), nn.ModuleList()
        for res in resolutions[1:]:
            out_c = max(latent_dim // (2 ** (resolutions.index(res))), 64)
            self.blocks.append(StyledConvBlock(in_channels, out_c, latent_dim))
            self.to_rgb.append(nn.Conv2d(out_c, img_channels, kernel_size=1, stride=1, padding=0))
            in_channels = out_c

    def forward(self, z):
        w = self.mapping(z)
        batch = z.size(0)
        x = self.constant_input.repeat(batch, 1, 1, 1)
        for block, rgb in zip(self.blocks, self.to_rgb):
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = block(x, w)
        return torch.tanh(rgb(x))

class Discriminator(nn.Module):
    def __init__(self, img_channels, img_size):
        super().__init__()
        channels = [img_channels, 64, 128, 256, 512, 512, 512]
        layers = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(channels[-1], 1, kernel_size=4, stride=1, padding=0)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).view(x.size(0), -1)
    


# Initialize models
G = StyleGANGenerator(latent_dim, img_channels, img_size).to(device)
D = Discriminator(img_channels, img_size).to(device)
G.apply(lambda m: isinstance(m, (nn.Conv2d, nn.Linear)) and nn.init.normal_(m.weight, 0, 0.02))

optim_G = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.0, 0.99))
optim_D = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.0, 0.99))
scaler = GradScaler()
criterion = nn.BCEWithLogitsLoss()

fixed_z = torch.randn(64, latent_dim, device=device)


# -------------------- Training Loop --------------------
for epoch in range(1, num_epochs+1):
    start = time.time()
    for real in loader:
        real = real.to(device)
        batch = real.size(0)
        real_labels = torch.ones(batch, 1, device=device)
        fake_labels = torch.zeros(batch, 1, device=device)

        # Train Discriminator
        optim_D.zero_grad()
        with autocast():
            fake = G(torch.randn(batch, latent_dim, device=device))
            d_real = D(real)
            d_fake = D(fake.detach())
            loss_D = (criterion(d_real, real_labels) + criterion(d_fake, fake_labels)) * 0.5
        scaler.scale(loss_D).backward()
        scaler.step(optim_D)

        # Train Generator
        optim_G.zero_grad()
        with autocast():
            d_fake_for_g = D(fake)
            loss_G = criterion(d_fake_for_g, real_labels)
        scaler.scale(loss_G).backward()
        scaler.step(optim_G)
        scaler.update()

    # Save samples and checkpoints
    G.eval()
    with torch.no_grad():
        samples = G(fixed_z)
        save_image(samples, os.path.join(generated_images, f"epoch_{epoch}.png"), nrow=8, normalize=True)
    G.train()
    torch.save({'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'optim_G': optim_G.state_dict(),
                'optim_D': optim_D.state_dict()},
               os.path.join(training_directory, f"checkpoint_{epoch}.pth"))
    print(f"Epoch [{epoch}/{num_epochs}] Time: {time.time()-start:.2f}s Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")

# Save final models
torch.save(G.state_dict(), os.path.join(training_directory, 'stylegan_G_final.pth'))
torch.save(D.state_dict(), os.path.join(training_directory, 'stylegan_D_final.pth'))
print('Training complete, models saved.')


