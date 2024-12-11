import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os

from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule
from torchvision.utils import save_image

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()


def sample_and_save_images(n_images, diffusor, model, device, store_path, transform=None, testloader=None):
    # TODO: Implement - adapt code and method signature as needed
    os.makedirs(store_path, exist_ok=True)
    model.eval()
    
    # Get a batch of test images
    test_iter = iter(testloader)
    images, _ = next(test_iter)  # Fetch the first batch
    images = images[:n_images].to(device)  # Select the first `n_images`

    # Generate images
    with torch.no_grad():
        noise = torch.randn_like(images)  # Generate noise with the same shape as test images
        generated_images = diffusor.sample_images(model, noise)

    # Save each generated image
    for i, img in enumerate(generated_images):
        # Reverse transform the generated image
        reversed_img = transform(img.cpu())
        # Save the image
        reversed_img.save(os.path.join(store_path, f"generated_image_{i}.png"))

def test(model, testloader, diffusor, device, args):
    # TODO: Implement - adapt code and method signature as needed
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    metric_results = []
    batch_size = args.batch_size
    timesteps = args.timesteps

    with torch.no_grad():  # Disable gradient computation for testing
        pbar = tqdm(testloader, desc="Testing")
        for step, (images, labels) in enumerate(pbar):
            images = images.to(device)

            # Algorithm 1 line 3: sample t uniformly for every example in the batch
            t = torch.randint(0, timesteps, (len(images),), device=device).long()

            # Compute the loss for the test batch
            loss = diffusor.p_losses(model, images, t, loss_type="l2")
            total_loss += loss.item()
            print(f"Test Loss: {total_loss:.6f}")
            # Optional: compute additional metrics like SSIM or PSNR
            if args.compute_metrics:
                generated_images = diffusor.sample_images(model, images, t)
                batch_metrics = args.compute_metrics(generated_images, images)
                metric_results.extend(batch_metrics)

            if args.dry_run:
                break

        # Calculate average loss over the test set
        avg_loss = total_loss / len(testloader)
        print(f"Avg Test Loss: {avg_loss:.6f}")

        # Optionally log additional metrics
        if args.compute_metrics:
            avg_metrics = {
                metric: sum(m[metric] for m in metric_results) / len(metric_results)
                for metric in metric_results[0].keys()
            }
            print("Additional Metrics:", avg_metrics)

    return avg_loss

def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    pbar = tqdm(trainloader)
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, loss_type="l2")

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break


def test(args):
    # TODO (2.2): implement testing functionality, including generation of stored images.
    pass


def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to aid diffusion process
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10('../CIFAR10/', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10('../CIFAR10/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)

    for epoch in range(epochs):
        train(model, trainloader, optimizer, diffusor, epoch, device, args)
        test(model, valloader, diffusor, device, args)

    test(model, testloader, diffusor, device, args)

    save_path = "./img"  # TODO: Adapt to your needs
    n_images = 8
    sample_and_save_images(n_images, diffusor, model, device, save_path, reverse_transform, testloader)
    torch.save(model.state_dict(), os.path.join("/proj/aimi-adl/models", args.run_name, f"ckpt.pt"))




if __name__ == '__main__':
    args = parse_args()
    # TODO (2.2): Add visualization capabilities
    run(args)
    
