from dcgan import Discriminator, Generator, LinearWalk
from lib import myfun
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from math import isqrt

from torchvision.utils import save_image, make_grid
LATENT_DIM = 100

# Training loop
def train_fix_gen(
        gen_path = 'weights/netG_epoch_99.pth',
        save_path = "out/method3_walk_fix_gen",
        edit_mode = 's',
        batch_num = 100,
        batch_size = 64,
        epochs = 10,
        lr_walk = 0.005,
        alpha_range = np.linspace(-1,1,6),
        eval_alpha_range = np.linspace(-1,1,5),
        ):
    device = torch.device("cuda")
    generator = Generator(ngpu=1).eval()
    generator.load_state_dict(torch.load(gen_path))

    walk_model = LinearWalk()

    optimizer_w = optim.Adam(walk_model.parameters(), lr=lr_walk)
    criterion_walk = nn.MSELoss()
    edit = myfun.get_edit_function(mode=edit_mode)
    
    walk_model.to(device)
    generator.to(device)

    torch.manual_seed(0)
    fixed_noise = torch.randn(batch_size, LATENT_DIM, 1, 1).to(device)

    w_losses =[]
    start = time.time()
    for epoch in range(epochs):
        all_gen_z = torch.randn(batch_num * batch_size, LATENT_DIM, 1, 1).to(device)
        for batch_start in range(0, len(all_gen_z), batch_size):
            alpha = np.random.choice(alpha_range)
            
            # Train generator Walk model
            walk_model.zero_grad()
            noise = all_gen_z[batch_start:batch_start+batch_size] #z
            noise_image = generator(noise) #G(z)
            noise_image_edit = edit(noise_image, alpha) #edit(G(z), a)
            noise_edit = walk_model.forward(noise, alpha) # z + aw
            noise_edit_images = generator(noise_edit) #G(z + aw)
            
            w_loss = criterion_walk(noise_edit_images, noise_image_edit)
            w_loss.backward()
            optimizer_w.step()

            # Print progress
            if (batch_start / batch_size)%25 == 0:
                w_losses.append(w_loss)
                print(
                    f"[Epoch {epoch}/{epochs}] [Batch {batch_start}/{batch_num * batch_size}] [W loss: {w_loss.item()}]"
                )

        # Save generated images at the end of each epoch
        with torch.no_grad():
            fig, axs = plt.subplots(ncols=len(eval_alpha_range), sharex=True, figsize=(3 * len(eval_alpha_range), 3))
            fig.suptitle(f'Epoch {epoch + 1}')  # Assuming epoch is defined somewhere before this code snippet
            
            for i, alpha in enumerate(eval_alpha_range):
                noise_edit = walk_model.forward(fixed_noise, alpha)
                images = generator(noise_edit)
                images = make_grid(images, normalize=True, nrow = isqrt(batch_size)).cpu()
                
                axs[i].set_title(f'alpha = {alpha:.2f}')
                axs[i].imshow(images.permute(1, 2, 0))

    end = time.time()
    print(f'Finished Training after {end-start} s ')
    torch.save(walk_model.state_dict(), save_path)


# Training loop
def train_gen_with_walk(
        gen_path = 'weights/netG_epoch_99.pth',
        gen_save_path = "out/gen",
        dis_path = 'weights/netD_epoch_99.pth',
        dis_save_path = 'out/dis',
        walk_save_path = "out/method3_walk",
        edit_mode = 's',
        batch_size = 64,
        epochs = 10,
        alpha_range = np.linspace(-1,1,6),
        eval_alpha_range = np.linspace(-1,1,5),
        gan_lr = 0.0002,
        lr_walk = 0.005,
        device = torch.device("cuda"),
        latent_dim = 100,
        ):

    # Load DCGAN
    generator = Generator(ngpu=1)
    generator.load_state_dict(torch.load(gen_path))
    optimizerG = optim.Adam(generator.parameters(), lr=gan_lr, betas=(0.5, 0.999))

    discriminator = Discriminator(ngpu=1)
    discriminator.load_state_dict(torch.load(dis_path))
    optimizerD = optim.Adam(discriminator.parameters(), lr=gan_lr, betas=(0.5, 0.999))

    criterion_GAN = nn.BCELoss()

    #initail walk
    walk_model = LinearWalk()
    optimizer_w = optim.Adam(walk_model.parameters(), lr=lr_walk)

    criterion_walk = nn.MSELoss()

    
    # send model to GPU
    walk_model.to(device)
    generator.to(device)
    discriminator.to(device)


    # prepare training and evaluating data
    torch.manual_seed(0)
    fixed_noise = torch.randn(batch_size, LATENT_DIM, 1, 1).to(device)
    transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,)),
                       ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    edit = myfun.get_edit_function(mode=edit_mode)


    w_losses =[]
    start = time.time()
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device) #x
            cur_batch_size =len(real_images)
            alpha = np.random.choice(alpha_range)
            real_labels = torch.ones(cur_batch_size).to(device)
            fake_labels = torch.zeros(cur_batch_size).to(device)

            # Train discriminator
            discriminator.zero_grad()

            real_images_edit = edit(real_images, alpha) #edit(x,a)
            real_edit_outputs = discriminator(real_images_edit) #D(edit(x,a))
            real_loss = criterion_GAN(real_edit_outputs, real_labels) #E(D(edit(x,a)))
            
            noise = torch.randn(cur_batch_size, LATENT_DIM, 1, 1).to(device) #z
            noise_edit = walk_model.forward(noise, alpha) # z + aw
            noise_edit_images = generator(noise_edit) #G(z + aw)
            noise_edit_outputs = discriminator(noise_edit_images.detach()) #D(G(z+aw))
            fake_loss = criterion_GAN(noise_edit_outputs, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizerD.step()


            # Train generator
            generator.zero_grad()
            noise = torch.randn(cur_batch_size, LATENT_DIM, 1, 1).to(device) #z
            noise_edit = walk_model.forward(noise, alpha) # z + aw
            noise_edit_images = generator(noise_edit) #G(z + aw)
            fake_outputs = discriminator(noise_edit_images) #D(G(z + aw))

            g_loss = criterion_GAN(fake_outputs, real_labels)
            g_loss.backward()
            optimizerG.step()
            
            # Train generator Walk model
            walk_model.zero_grad()
            noise = torch.randn(cur_batch_size, LATENT_DIM, 1, 1).to(device) #z
            noise_image = generator(noise) #G(z)
            noise_image_edit = edit(noise_image, alpha) #edit(G(z), a)
            noise_edit = walk_model.forward(noise, alpha) # z + aw
            noise_edit_images = generator(noise_edit) #G(z + aw)
            
            w_loss = criterion_walk(noise_edit_images, noise_image_edit)
            w_loss.backward()
            optimizer_w.step()

            # Print progress
            if i % 1000 == 0:
                w_losses.append(w_loss)
                print(
                    f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [W loss: {w_loss.item()}]"
                )


        # Save generated images at the end of each epoch
        if epoch % (epochs//5) == 0:
            with torch.no_grad():
                fig, axs = plt.subplots(ncols=len(eval_alpha_range), sharex=True, figsize=(3 * len(eval_alpha_range), 3))
                fig.suptitle(f'Epoch {epoch + 1}')  
                
                for i, alpha in enumerate(eval_alpha_range):
                    noise_edit = walk_model.forward(fixed_noise, alpha)
                    images = generator(noise_edit)
                    images = make_grid(images, normalize=True, nrow = isqrt(batch_size)).cpu()
                    
                    axs[i].set_title(f'alpha = {alpha:.2f}')
                    axs[i].imshow(images.permute(1, 2, 0))

    end = time.time()
    print(f'Finished Training after {end-start} s ')
    torch.save(walk_model.state_dict(), walk_save_path)
    torch.save(generator.state_dict(), gen_save_path)
    torch.save(discriminator.state_dict(), dis_save_path)