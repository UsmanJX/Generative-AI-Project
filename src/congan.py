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

# Create output directories
os.makedirs(training_directory, exist_ok=True)
os.makedirs(generated_images, exist_ok=True)

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
    train = df.sample(frac=0.75, random_state=394)
    test = df.drop(train.index)
    class_size = max(train['denomination'].value_counts()) if not train.empty else 0
    train = train.groupby(['denomination']).apply(lambda x: x.sample(class_size, replace=True)).reset_index(drop=True)
    train = train.sample(frac=1).reset_index(drop=True)
    train.to_csv(os.path.join(working_directory, 'Train.csv'), index=False)
    test.to_csv(os.path.join(working_directory, 'Test.csv'), index=False)

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
    def __init__(self, csv_file, transforms):
        self.Dataset = pd.read_csv(csv_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.Dataset['images'].to_list())

    def __getitem__(self, idx):
        imagePath = self.Dataset['images'].to_list()[idx]
        image = cv2.imread(imagePath)
        if image is None:
            print(f"Warning: Could not read image {imagePath}")
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        scale = 512.0 / np.amin(image.shape[0:2])
        image = cv2.resize(image, (int(np.ceil(image.shape[1] * scale)), int(np.ceil(image.shape[0] * scale))))
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
        return image, label

# Generator class with dynamic num_classes
class Generator(nn.Module):
    def __init__(self, num_classes):
        super(Generator, self).__init__()
        self.label_conditioned_generator = nn.Sequential(nn.Embedding(num_classes, 100),
                                                         nn.Linear(100, 16))
        self.latent = nn.Sequential(nn.Linear(100, 4*4*512),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(nn.ConvTranspose2d(513, 64*8, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64*8, momentum=0.1, eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64*4, momentum=0.1, eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64*2, momentum=0.1, eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64*2, 64*1, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64*1, momentum=0.1, eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64*1, 3, 4, 2, 1, bias=False),
                                   nn.Tanh())

    def forward(self, inputs):
        noise_vector, label = inputs
        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 1, 4, 4)
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 512, 4, 4)
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(concat)
        return image

# Discriminator class with dynamic num_classes
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.label_condition_disc = nn.Sequential(nn.Embedding(num_classes, 100),
                                                  nn.Linear(100, 3*128*128))
        self.model = nn.Sequential(nn.Conv2d(6, 64, 4, 2, 1, bias=False),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64, 64*2, 4, 3, 2, bias=False),
                                   nn.BatchNorm2d(64*2, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64*2, 64*4, 4, 3, 2, bias=False),
                                   nn.BatchNorm2d(64*4, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64*4, 64*8, 4, 3, 2, bias=False),
                                   nn.BatchNorm2d(64*8, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Flatten(),
                                   nn.Dropout(0.4),
                                   nn.Linear(4608, 1),
                                   nn.Sigmoid())

    def forward(self, inputs):
        img, label = inputs
        label_output = self.label_condition_disc(label)
        label_output = label_output.view(-1, 3, 128, 128)
        concat = torch.cat((img, label_output), dim=1)
        output = self.model(concat)
        return output

# Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

# Plotting functions
def show_images(images):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images.detach(), nrow=16).permute(1, 2, 0))
    plt.savefig(os.path.join(generated_images, 'training_batch.png'))

def show_batch(dl):
    for images, _ in dl:
        show_images(images)
        break

def performance(H):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(H["D_losses"], label="discriminator_loss")
    plt.plot(H["G_losses"], label="generator_loss")
    plt.title("Model Performance")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_directory, 'results', 'performance.png'), dpi=1000, bbox_inches='tight')

# Training parameters
num_epochs = 155
latent_dim = 100
batch_size = 128
learning_rate = 0.0002

# Initialize models
if num_classes > 0:
    G = Generator(num_classes).to(device)
    G.apply(weights_init)
    D = Discriminator(num_classes).to(device)
    D.apply(weights_init)

    # Loss and optimizers
    loss = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Load or initialize training history
    checkpoint_path = os.path.join(training_directory, 'latest_checkpoint.pth')
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        G.load_state_dict(checkpoint['G_state_dict'])
        D.load_state_dict(checkpoint['D_state_dict'])
        G_optimizer.load_state_dict(checkpoint['G_optimizer'])
        D_optimizer.load_state_dict(checkpoint['D_optimizer'])
        training_history = checkpoint['training_history']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        training_history = {'D_losses': [], 'G_losses': [], 'per_epoch_time': []}

    # Data transformations
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor()])

    # Load training data
    TrainDS = Coins_Dataloader(csv_file=os.path.join(working_directory, 'Train.csv'), transforms=transform)
    train_loader = DataLoader(TrainDS, shuffle=True, batch_size=batch_size)
    show_batch(train_loader)

    # Training loop
    print('Training starts!')
    for epoch in range(start_epoch, num_epochs):
        start = time.time()
        discriminator_loss, generator_loss = [], []
        for index, (real_images, labels) in enumerate(train_loader):
            D_optimizer.zero_grad()
            real_images = real_images.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1).long().abs()
            real_target = Variable(torch.ones(real_images.size(0), 1).to(device))
            fake_target = Variable(torch.zeros(real_images.size(0), 1).to(device))
            D_real_loss = loss(D((real_images, labels)), real_target)
            noise_vector = torch.randn(real_images.size(0), latent_dim, device=device)
            generated_image = G((noise_vector, labels))
            output = D((generated_image.detach(), labels))
            D_fake_loss = loss(output, fake_target)
            D_total_loss = (D_real_loss + D_fake_loss) / 2
            discriminator_loss.append(D_total_loss)
            D_total_loss.backward()
            D_optimizer.step()
            G_optimizer.zero_grad()
            G_loss = loss(D((generated_image, labels)), real_target)
            generator_loss.append(G_loss)
            G_loss.backward()
            G_optimizer.step()
        print('Epoch: [%d/%d]: D_loss: %.3f | G_loss: %.3f' % (
            epoch, num_epochs,
            torch.mean(torch.FloatTensor(discriminator_loss)),
            torch.mean(torch.FloatTensor(generator_loss))))
        training_history['D_losses'].append(torch.mean(torch.FloatTensor(discriminator_loss)))
        training_history['G_losses'].append(torch.mean(torch.FloatTensor(generator_loss)))
        save_image(generated_image.data[:50], os.path.join(generated_images, f'sample_epoch_{epoch}.png'), nrow=10, normalize=True)
        torch.save({
            'epoch': epoch,
            'G_state_dict': G.state_dict(),
            'D_state_dict': D.state_dict(),
            'G_optimizer': G_optimizer.state_dict(),
            'D_optimizer': D_optimizer.state_dict(),
            'training_history': training_history
        }, checkpoint_path)
        end = time.time()
        elapsed = end - start
        training_history['per_epoch_time'].append(time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed)))

    print('Training ends!')
    print('Training Start Time in UTC:', time.asctime(time.localtime(start)))
    print('Training End Time in UTC:', time.asctime(time.localtime(end)))
    print('Total Training Time in Minutes:', elapsed/60)

    # Plot performance
    performance(training_history)

    # Load test data
    TestDS = Coins_Dataloader(csv_file=os.path.join(working_directory, 'Test.csv'), transforms=transform)
    test_loader = DataLoader(TestDS, shuffle=False, batch_size=batch_size)

    # Evaluation
    print('Evaluating on test images!')
    for index, (test_images, labels) in enumerate(test_loader):
        test_images = test_images.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(1).long().abs()
        real_target = Variable(torch.ones(test_images.size(0), 1).to(device))
        fake_target = Variable(torch.zeros(test_images.size(0), 1).to(device))
        D_test_loss = loss(D((test_images, labels)), real_target)
        noise_vector = torch.randn(test_images.size(0), latent_dim, device=device)
        generated_image = G((noise_vector, labels))
        output = D((generated_image.detach(), labels))
        D_fake_loss = loss(output, fake_target)
        D_loss = (D_test_loss + D_fake_loss) / 2
        G_loss = loss(D((generated_image, labels)), real_target)
        save_image(generated_image.data, os.path.join(generated_images, f'test_sample_{index}.png'), nrow=12, normalize=True)
        print('Batch: [%d/%d]: D_loss: %.3f | G_loss: %.3f' % (
            index, len(test_loader)-1, D_loss, G_loss))

    # Generate final sample images
    with torch.no_grad():
        noise = torch.randn(16, latent_dim).to(device)
        sample_labels = torch.randint(0, num_classes, (16,)).to(device)
        generated = G((noise, sample_labels))
        save_image(generated, os.path.join(generated_images, 'final_samples.png'), nrow=4, normalize=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(make_grid(generated.detach().cpu(), nrow=4).permute(1, 2, 0))
        ax.axis('off')
        plt.savefig(os.path.join(generated_images, 'final_samples_display.png'))

    print("Check /kaggle/working/results/generated_images/ for reconstructed coins.")
else:
    print("No data available for training.")
    
    