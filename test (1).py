import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
#from torch.utils.tensorboard import SummaryWriter
from DCGAN import  Discriminator, Generator , initialize_weights
import matplotlib.pyplot as plt



# Parametri
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L_R = 0.0002
BATCH_SIZE = 128
IMAGE_SIZE = 64
Z_DIM = 100
CHANNELS_IMG = 3
NUM_EPOCHS = 20
FEATURES_DISC = 64
FEATURES_GEN = 64
'''
beta1 = 0.5
beta2 = 0.999
 
'''
# Definizione delle trasformazioni
transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    #transforms.CenterCrop(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)] ,[0.5 for _ in range(CHANNELS_IMG)]),
])
dataset_full = datasets.ImageFolder(root="img_align_celeba",transform = transforms )
num_img = 3500

subset_idx = list(range(num_img))
subset_sampler = SubsetRandomSampler(subset_idx)
dataset_subset = torch.utils.data.Subset(dataset_full, subset_idx)

loader = DataLoader ( dataset_subset ,batch_size= BATCH_SIZE ,shuffle=True)
gen = Generator(Z_DIM,CHANNELS_IMG,FEATURES_GEN,).to(device)
disc = Discriminator(CHANNELS_IMG,FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)
opt_gen = optim.Adam(gen.parameters(),lr = L_R, betas=(0.5,0.999))
opt_disc = optim.Adam(disc.parameters(),lr = L_R, betas=(0.5,0.999))
criterion = nn.BCELoss()

fixed_noise= torch.randn(32,Z_DIM,1,1).to(device)
#writer_real= SummaryWriter(f"logs/real")
#writer_fake= SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM,1,1)).to(device)
        fake = gen(noise)

        ### Train Discriminator max log(D(x)) + log(1- D(G(z)))
        disc_real= disc(real).reshape(-1)
        loss_disc_real=criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake)/ 2
        disc.zero_grad()
        loss_disc.backward(retain_graph = True)
        opt_disc.step()

        ### Train Generator min log(1-D(g(z))) === max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 10 == 0:
            print(
                f"[Epoch {epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                   D Loss: {loss_disc:.4f}, G Loss: {loss_gen:.4f}"
            )
            with torch.no_grad():
                fake = gen(fixed_noise)


                # Visualizza un campione di immagini generate
                fig, axes = plt.subplots(5, 5, figsize=(10, 10))
                for i, ax in enumerate(axes.flatten()):
                    ax.imshow(fake[i].cpu().permute(1, 2, 0).numpy())
                    ax.axis('off')
                plt.show()
            step += 1


