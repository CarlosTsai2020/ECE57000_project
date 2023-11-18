from dcgan import Generator, LinearWalk
from lib import myfun
import matplotlib.pyplot as plt
from math import isqrt
import numpy as np
import torch
from torchvision.utils import save_image, make_grid

LATENT_DIM = 100

def test(
        gen_path = 'weights/netG_epoch_99.pth',
        walk_path = "out/method3_walk",
        edit_mode = 's',
        eval_alpha_range = np.linspace(-1,1,5),
        device = torch.device("cuda"),
        ):
    # Load DCGAN
    generator = Generator(ngpu=1).eval()
    generator.load_state_dict(torch.load(gen_path))

    walk_model = LinearWalk().eval()
    walk_model.load_state_dict(torch.load(walk_path))

    # send model to GPU
    walk_model.to(device)
    generator.to(device)
    edit = myfun.get_edit_function(mode=edit_mode)

    
    fixed_noise = torch.randn(16, LATENT_DIM, 1, 1).to(device)
    fig, axs = plt.subplots(ncols=len(eval_alpha_range)+1, sharex=True, figsize=(3 * len(eval_alpha_range)+1, 3))
    fig.suptitle(f'test')
    target = generator(fixed_noise)
    target = edit(target, 1)
    target = make_grid(target, normalize=True, nrow = isqrt(16)).cpu()
    axs[0].set_title(f'targett: alpha = 1')
    axs[0].imshow(target.permute(1, 2, 0))
    for i, alpha in enumerate(eval_alpha_range):
        noise_edit = walk_model.forward(fixed_noise, alpha)
        images = generator(noise_edit)
        images = make_grid(images, normalize=True, nrow = isqrt(16)).cpu()
        
        axs[i+1].set_title(f'alpha = {alpha:.2f}')
        axs[i+1].imshow(images.permute(1, 2, 0))


def test_gen_with_walk(
        gen_path = 'weights/netG_epoch_99.pth',
        walk_path = "out/method3_walk",
        edit_mode = 's',
        eval_alpha_range = np.linspace(-1,1,5),
        device = torch.device("cuda"),
        n = 16,
        ):
    # Load DCGAN
    generator = Generator(ngpu=1).eval()
    generator.load_state_dict(torch.load(gen_path))

    walk_model = LinearWalk().eval()
    walk_model.load_state_dict(torch.load(walk_path))

    # send model to GPU
    walk_model.to(device)
    generator.to(device)
    edit = myfun.get_edit_function(mode=edit_mode)


    m = len(eval_alpha_range)
    fixed_noise = torch.randn(n, LATENT_DIM, 1, 1).to(device)
    noise = torch.ones(n*m, LATENT_DIM, 1, 1).to(device)
    for i, alpha in enumerate(eval_alpha_range):
        noise[i*n:(i+1)*n] = walk_model.forward(fixed_noise, alpha)

    mode_dic = {'r': "rotate", 's': "shift", 'z': "zoom"}

    fig, axs = plt.subplots(sharex=True, figsize=(m*1.2,m*1.2))
    images = generator(noise)
    images = make_grid(images, normalize=True, nrow = n).cpu()
    title = 'mode:' + mode_dic[edit_mode]
    axs.set_title(title)
    axs.imshow(images.permute(1, 2, 0))