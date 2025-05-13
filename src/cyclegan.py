import os
import cv2
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import scipy.ndimage as ndi
import torch.optim as optim
from collections import Counter
from torchvision import transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.utils import save_image
import glob

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define Kaggle-specific paths
input_data_directory = '/kaggle/input/coin-dataset/data'
input_images_directory = '/kaggle/input/coin-dataset/images'
working_directory = '/kaggle/working'
training_directory = os.path.join(working_directory, 'results', 'training_weights')
generated_images = os.path.join(working_directory, 'results', 'generated_images')
reconstructions_dir = os.path.join(working_directory, 'results', 'reconstructions')

# Create output directories
os.makedirs(training_directory, exist_ok=True)
os.makedirs(generated_images, exist_ok=True)
os.makedirs(reconstructions_dir, exist_ok=True)

# Define dataset preparation function
def dataset(path):
    df = pd.read_csv(os.path.join(input_data_directory, 'query.csv'))[['RecordId', 'Denomination']]
    data_list = []
    for folder in os.listdir(path):
        if folder in df['RecordId'].values:
            idx = df[df['RecordId'] == folder].index[0]
            denomination = df.at[idx, 'Denomination']
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(path, folder, filename)
                        data_list.append({'images': img_path, 'denomination': denomination})
        else:
            print(f"Warning: Folder {folder} not found in query.csv")
    data = pd.DataFrame(data_list)
    if not data.empty:
        data['labels'] = data['denomination'].astype('category').cat.codes
    else:
        print("No data found in the dataset.")
    return data

# Prepare dataset
df = dataset(input_images_directory)
df.to_csv(os.path.join(working_directory, 'coins-dataset.csv'), index=False)
num_classes = len(df['denomination'].unique()) if not df.empty else 0
print(f"Number of classes: {num_classes}")

# Train-test split and balance
if not df.empty:
    # Keep track of original image paths for reconstruction testing
    test_size = 0.25
    train = df.sample(frac=1-test_size, random_state=394)
    test = df.drop(train.index)
    
    # Save some original test images for reconstruction comparison
    test.to_csv(os.path.join(working_directory, 'Test.csv'), index=False)
    
    # Balance training data
    class_size = max(train['denomination'].value_counts()) if not train.empty else 0
    train = train.groupby(['denomination']).apply(lambda x: x.sample(class_size, replace=True)).reset_index(drop=True)
    train = train.sample(frac=1).reset_index(drop=True)
    train.to_csv(os.path.join(working_directory, 'Train.csv'), index=False)




# Preprocessing class
class preprocess(object):
    def __init__(self):
        self.input_image = None
        self.local_range = None
        radius = 3
        self._footprint = np.zeros((2*radius+1, 2*radius+1), dtype=np.bool_)
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                d_sq = dx*dx + dy*dy
                if d_sq > radius * radius:
                    continue
                self._footprint[dx + radius, dy + radius] = True

    def segment(self, image):
        self.input_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.local_range = ndi.maximum_filter(self.input_image, footprint=self._footprint)
        self.local_range -= ndi.minimum_filter(self.input_image, footprint=self._footprint)
        self.local_range = self.local_range / float(np.amax(self.local_range))
        best_threshold = 0
        best_contour = None
        best_form_factor = 0.0
        best_bin_im = None
        for threshold in np.arange(0.05, 0.65, 0.05):
            contour_im = self.local_range >= threshold
            contours, _ = cv2.findContours(np.array(contour_im, dtype=np.uint8),
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)
            areas = list(cv2.contourArea(c) for c in contours)
            if areas:
                max_index = np.argmax(areas)
                contour = contours[max_index]
                area = areas[max_index]
                perim = cv2.arcLength(contour, closed=True)
                if perim > 0:
                    form_factor = 4.0 * np.pi * area / (perim * perim)
                    if area <= 0.9 * np.product(self.local_range.shape):
                        if form_factor >= best_form_factor:
                            best_threshold = threshold
                            best_contour = contour
                            best_form_factor = form_factor
                            best_bin_im = contour_im
        if best_contour is not None:
            self.edge = np.reshape(best_contour, (len(best_contour), 2))
            self.edge_mask = best_bin_im.astype('float64')
            self.edge_threshold = best_threshold
            self.edge_form_factor = best_form_factor
        else:
            self.edge = None

# DataLoader class
class Coins_Dataloader(Dataset):
    def __init__(self, csv_file, transforms, return_paths=False):
        self.Dataset = pd.read_csv(csv_file)
        self.transforms = transforms
        self.return_paths = return_paths

    def __len__(self):
        return len(self.Dataset['images'].to_list())

    def __getitem__(self, idx):
        image_path = self.Dataset['images'].to_list()[idx]
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Warning: Could not read image {image_path}")
            original_image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Process image
        scale = 512.0 / np.amin(original_image.shape[0:2])
        image = cv2.resize(original_image, (int(np.ceil(original_image.shape[1] * scale)), 
                                           int(np.ceil(original_image.shape[0] * scale))))
        
        # Keep a copy of pre-processed image
        processed_image = image.copy()
        
        p = preprocess()
        p.segment(image)
        if p.edge is not None:
            rect = cv2.boundingRect(p.edge)
            x, y, w, h = rect
            if w > 0 and h > 0:
                image = image[y:y+h, x:x+w]
            else:
                image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            
        label = self.Dataset['labels'].to_list()[idx]
        
        if self.transforms is not None:
            image = self.transforms(image)
            processed_image = self.transforms(processed_image)  # Also transform the pre-processed image
        
        if self.return_paths:
            return image, label, image_path
        return image, label

# Encoder class for converting images to latent space
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            # Input: 3x128x128
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),  # Output: 64x64x64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # Output: 128x32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # Output: 256x16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),  # Output: 512x8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),  # Output: 1024x4x4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc_mu = nn.Linear(1024 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 4 * 4, latent_dim)
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024 * 4 * 4)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Generator class with dynamic num_classes
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        
        # Processing the class label
        self.label_embedding = nn.Embedding(num_classes, 100)
        self.label_projection = nn.Linear(100, 16)
        
        # Processing the latent vector
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Combine and generate image
        self.model = nn.Sequential(
            # Input: 513x4x4
            nn.ConvTranspose2d(513, 512, 4, 2, 1, bias=False),  # Output: 512x8x8
            nn.BatchNorm2d(512, momentum=0.1, eps=0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # Output: 256x16x16
            nn.BatchNorm2d(256, momentum=0.1, eps=0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # Output: 128x32x32
            nn.BatchNorm2d(128, momentum=0.1, eps=0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # Output: 64x64x64
            nn.BatchNorm2d(64, momentum=0.1, eps=0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # Output: 3x128x128
            nn.Tanh()
        )

    def forward(self, inputs):
        latent_vector, label = inputs
        
        # Process label
        label_output = self.label_embedding(label)
        label_output = self.label_projection(label_output)
        label_output = label_output.view(-1, 1, 4, 4)
        
        # Process latent vector
        latent_output = self.latent_projection(latent_vector)
        latent_output = latent_output.view(-1, 512, 4, 4)
        
        # Combine and generate
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(concat)
        
        return image


# Discriminator class with dynamic num_classes
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        
        # Process label for conditioning
        self.label_embedding = nn.Embedding(num_classes, 100)
        self.label_projection = nn.Linear(100, 3 * 128 * 128)
        
        # Combined image and label processing
        self.model = nn.Sequential(
            # Input: 6x128x128 (3 channels from image + 3 channels from label conditioning)
            nn.Conv2d(6, 64, 4, 2, 1, bias=False),  # Output: 64x64x64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # Output: 128x32x32 (using stride 2 instead of 3)
            nn.BatchNorm2d(128, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # Output: 256x16x16
            nn.BatchNorm2d(256, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # Output: 512x8x8
            nn.BatchNorm2d(512, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512 * 8 * 8, 1),  # Adjusted to match feature map size
            nn.Sigmoid()
        )

    def forward(self, inputs):
        img, label = inputs
        
        # Process label for conditioning
        label_output = self.label_embedding(label)
        label_output = self.label_projection(label_output)
        label_output = label_output.view(-1, 3, 128, 128)
        
        # Combine image and label
        concat = torch.cat((img, label_output), dim=1)
        
        # Process combined input
        output = self.model(concat)
        
        return output

# Reparameterization trick for VAE
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

# Plotting functions
def show_images(images, title=None):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=16)
    ax.imshow(make_grid(images.detach().cpu(), nrow=8).permute(1, 2, 0))
    plt.savefig(os.path.join(generated_images, f'{title if title else "training_batch"}.png'))
    plt.close()

def show_batch(dl, title=None):
    for images, _ in dl:
        show_images(images, title)
        break

def performance(H):
    fig = plt.figure(figsize=(10, 8))
    plt.plot(H["D_losses"], label="Discriminator Loss")
    plt.plot(H["G_losses"], label="Generator Loss")
    plt.plot(H["Recon_losses"], label="Reconstruction Loss")
    plt.title("Model Performance", fontsize=18)
    plt.xlabel("Number of epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(working_directory, 'results', 'performance.png'), dpi=1000, bbox_inches='tight')
    plt.close()


# Function to compare original and reconstructed images
def compare_reconstruction(original_images, reconstructed_images, epoch, batch_idx=0):
    # Take up to 8 images to display
    n = min(8, original_images.size(0))
    
    fig, axes = plt.subplots(2, n, figsize=(n*3, 6))
    
    for i in range(n):
        # Original images on top row
        orig_img = original_images[i].detach().cpu()
        axes[0, i].imshow(orig_img.permute(1, 2, 0).clip(0, 1))
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        # Reconstructed images on bottom row
        recon_img = reconstructed_images[i].detach().cpu()
        axes[1, i].imshow(recon_img.permute(1, 2, 0).clip(0, 1))
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(reconstructions_dir, f'reconstruction_epoch_{epoch}_batch_{batch_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reconstruction comparison to {save_path}")


# Training parameters
num_epochs = 120  # Reduced to fit your constraint
latent_dim = 100
batch_size = 64    # Reduced to avoid memory issues
learning_rate = 0.0002
checkpoint_interval = 40  # Save checkpoints every 40 epochs as requested


# Initialize models
if num_classes > 0:
    # Initialize encoder
    encoder = Encoder(latent_dim).to(device)
    encoder.apply(weights_init)
    
    # Initialize generator
    generator = Generator(latent_dim, num_classes).to(device)
    generator.apply(weights_init)
    
    # Initialize discriminator
    discriminator = Discriminator(num_classes).to(device)
    discriminator.apply(weights_init)
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.MSELoss()
    
    # Optimizers with slightly adjusted betas for better stability
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Initialize training history
    start_epoch = 0
    training_history = {'D_losses': [], 'G_losses': [], 'Recon_losses': [], 'KL_losses': [], 'per_epoch_time': []}
    
    # Data transformations - adding normalization for better training stability
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] range
    ])
    
    # Load training data
    train_dataset = Coins_Dataloader(csv_file=os.path.join(working_directory, 'Train.csv'), transforms=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    
    # Load test data
    test_dataset = Coins_Dataloader(csv_file=os.path.join(working_directory, 'Test.csv'), transforms=transform, return_paths=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
    # Show a batch of training images
    show_batch(train_loader, "original_training_batch")
    
    # Function to save checkpoint
    def save_checkpoint(epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'generator_optimizer': generator_optimizer.state_dict(),
            'discriminator_optimizer': discriminator_optimizer.state_dict(),
            'training_history': training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(training_directory, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")
        
        # Also save as latest checkpoint for potential resuming
        latest_path = os.path.join(training_directory, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best model if specified
        if is_best:
            best_path = os.path.join(training_directory, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {epoch}")
    
    # Training loop
    print('Training starts!')
    best_loss = float('inf')
    fixed_noise = torch.randn(16, latent_dim, device=device)
    fixed_labels = torch.randint(0, num_classes, (16, 1), device=device)
    
    for epoch in range(start_epoch, num_epochs):
        start = time.time()
        
        # Training mode
        encoder.train()
        generator.train()
        discriminator.train()
        
        epoch_d_loss = []
        epoch_g_loss = []
        epoch_recon_loss = []
        epoch_kl_loss = []
        
        for index, (real_images, labels) in enumerate(train_loader):
            batch_size = real_images.size(0)
            
            # Convert to proper format and move to device
            real_images = real_images.to(device)
            labels = labels.to(device).unsqueeze(1).long()
            
            # Ground truths with label smoothing for more stable training
            valid = torch.ones(batch_size, 1, device=device) * 0.9  # Smooth labels for better training
            fake = torch.zeros(batch_size, 1, device=device) + 0.1
            
            # -----------------
            # Train Discriminator
            # -----------------
            discriminator_optimizer.zero_grad()
            
            # Loss on real images
            real_validity = discriminator((real_images, labels))
            d_real_loss = adversarial_loss(real_validity, valid)
            
            # Encode real images to get latent space
            mu, logvar = encoder(real_images)
            z = reparameterize(mu, logvar)
            
            # Generate fake images
            fake_images = generator((z, labels))
            
            # Loss on fake images
            fake_validity = discriminator((fake_images.detach(), labels))
            d_fake_loss = adversarial_loss(fake_validity, fake)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            discriminator_optimizer.step()
            
            # -----------------
            # Train Generator and Encoder
            # -----------------
            # Don't train generator and encoder every iteration to allow discriminator to improve
            if index % 1 == 0:  # Can be adjusted to improve stability
                encoder_optimizer.zero_grad()
                generator_optimizer.zero_grad()
                
                # Encode real images again (needed for backprop)
                mu, logvar = encoder(real_images)
                z = reparameterize(mu, logvar)
                
                # Generate images again
                fake_images = generator((z, labels))
                
                # Adversarial loss
                fake_validity = discriminator((fake_images, labels))
                g_loss = adversarial_loss(fake_validity, valid)
                
                # Reconstruction loss
                # Using L1 loss combined with MSE for better details
                l1_loss = nn.L1Loss()(fake_images, real_images)
                mse_loss = reconstruction_loss(fake_images, real_images)
                recon_loss = mse_loss + 0.5 * l1_loss  # Combining both losses
                
                # KL divergence loss with annealing factor for better training stability
                kl_weight = min(1.0, epoch / 20.0)  # Gradual increase of KL weight
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / (batch_size * 3 * 128 * 128)  # Normalize by image dimensions
                
                # Total generator and encoder loss - adjusted weights for better balance
                recon_weight = 20.0  # Higher weight on reconstruction
                ge_loss = g_loss + recon_weight * recon_loss + kl_weight * 0.1 * kl_loss
                ge_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                
                generator_optimizer.step()
                encoder_optimizer.step()
            
            # Record losses
            epoch_d_loss.append(d_loss.item())
            epoch_g_loss.append(g_loss.item())
            epoch_recon_loss.append(recon_loss.item())
            epoch_kl_loss.append(kl_loss.item() if 'kl_loss' in locals() else 0)
            
            # Print batch results
            if index % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{index}/{len(train_loader)}] "
                      f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, "
                      f"Recon_loss: {recon_loss.item():.4f}, KL_loss: {kl_loss.item():.4f}")
                
                # Save sample reconstructed images every few batches
                if index % 20 == 0:
                    with torch.no_grad():
                        # Generate reconstructed images from current batch
                        mu, logvar = encoder(real_images[:8])
                        z = reparameterize(mu, logvar)
                        recon_images = generator((z, labels[:8]))
                        
                        # Create a grid with original and reconstructed
                        combined = torch.cat([real_images[:8], recon_images])
                        save_image((combined + 1) / 2,  # Denormalize from [-1,1] to [0,1]
                                  os.path.join(generated_images, f'recon_epoch_{epoch}_batch_{index}.png'),
                                  nrow=8, normalize=False)
        
        # Calculate average epoch losses
        avg_d_loss = sum(epoch_d_loss) / len(epoch_d_loss)
        avg_g_loss = sum(epoch_g_loss) / len(epoch_g_loss)
        avg_recon_loss = sum(epoch_recon_loss) / len(epoch_recon_loss)
        avg_kl_loss = sum(epoch_kl_loss) / len(epoch_kl_loss)
        
        # Record training history
        training_history['D_losses'].append(avg_d_loss)
        training_history['G_losses'].append(avg_g_loss)
        training_history['Recon_losses'].append(avg_recon_loss)
        training_history['KL_losses'].append(avg_kl_loss)
        
        # End of epoch summary
        end = time.time()
        elapsed = end - start
        training_history['per_epoch_time'].append(time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed)))
        
        print(f'Epoch [{epoch}/{num_epochs}] completed in {time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed))}')
        print(f'D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f} | Recon_loss: {avg_recon_loss:.4f} | KL_loss: {avg_kl_loss:.4f}')
        
        # Save model checkpoint at specified intervals
        if (epoch + 1) % checkpoint_interval == 0 or epoch == num_epochs - 1:
            save_checkpoint(epoch)
            
            # Evaluate reconstruction on test data
            evaluate_reconstruction(epoch)
            
            # Update plots - with KL loss now included
            fig = plt.figure(figsize=(10, 8))
            plt.plot(training_history["D_losses"], label="Discriminator Loss")
            plt.plot(training_history["G_losses"], label="Generator Loss")
            plt.plot(training_history["Recon_losses"], label="Reconstruction Loss")
            plt.plot(training_history["KL_losses"], label="KL Loss")
            plt.title("Model Performance", fontsize=18)
            plt.xlabel("Number of epochs", fontsize=14)
            plt.ylabel("Loss", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(working_directory, 'results', f'performance_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Check if this is the best model so far - using reconstruction loss as metric
        current_loss = avg_recon_loss
        if current_loss < best_loss:
            best_loss = current_loss
            save_checkpoint(epoch, is_best=True)
            
        # Generate sample images with fixed noise
        with torch.no_grad():
            # Generate with fixed noise and labels for consistency across epochs
            gen_imgs = generator((fixed_noise, fixed_labels))
            save_image((gen_imgs + 1) / 2,  # Denormalize from [-1,1] to [0,1]
                      os.path.join(generated_images, f'fixed_noise_samples_epoch_{epoch}.png'),
                      nrow=4, normalize=False)
            
            # If we're on the last epoch or at checkpoint intervals, save a larger batch of examples
            if epoch == num_epochs - 1 or (epoch + 1) % checkpoint_interval == 0:
                # Generate 64 examples
                large_noise = torch.randn(64, latent_dim, device=device)
                large_labels = torch.tensor([[i % num_classes] for i in range(64)], device=device)
                large_imgs = generator((large_noise, large_labels))
                save_image((large_imgs + 1) / 2,  # Denormalize
                          os.path.join(generated_images, f'large_samples_epoch_{epoch}.png'),
                          nrow=8, normalize=False)
                
                # Also create a special visualization showing one example per class
                if num_classes <= 16:
                    class_noise = torch.randn(num_classes, latent_dim, device=device)
                    class_labels = torch.tensor([[i] for i in range(num_classes)], device=device)
                    class_imgs = generator((class_noise, class_labels))
                    save_image((class_imgs + 1) / 2,
                              os.path.join(generated_images, f'per_class_samples_epoch_{epoch}.png'),
                              nrow=min(8, num_classes), normalize=False)

    # At the end of training, save the final models separately for easy loading
    final_model_dir = os.path.join(working_directory, 'results', 'final_models')
    os.makedirs(final_model_dir, exist_ok=True)
    
    torch.save(encoder.state_dict(), os.path.join(final_model_dir, 'encoder_final.pth'))
    torch.save(generator.state_dict(), os.path.join(final_model_dir, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(final_model_dir, 'discriminator_final.pth'))
    
    print("Training complete! Final models saved.")
    
    # Final evaluation on test dataset
    print("Performing final evaluation on test dataset...")
    evaluate_reconstruction(num_epochs)
    
    # Generate a large grid of samples as final output
    with torch.no_grad():
        num_samples = min(100, num_classes * 10)  # Generate multiple samples per class
        sample_noise = torch.randn(num_samples, latent_dim, device=device)
        sample_labels = torch.tensor([[i % num_classes] for i in range(num_samples)], device=device)
        sample_imgs = generator((sample_noise, sample_labels))
        
        # Save a large grid of generated images
        save_image((sample_imgs + 1) / 2,
                  os.path.join(generated_images, 'final_generation_grid.png'),
                  nrow=10, normalize=False)
        
        print("Final image grid generated!")
