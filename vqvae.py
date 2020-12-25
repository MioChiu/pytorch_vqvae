import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from options import Options
from modules import VectorQuantizedVAE, to_scalar
from datasets import MiniImagenet
from data import load_data
from tensorboardX import SummaryWriter


def train(data_loader, model, optimizer, args, writer, epoch):
    pbar = tqdm(data_loader)
    for images, _ in pbar:
        images = images.to(args.device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + args.beta * loss_commit
        loss.backward()

        pbar.set_description("epoch %s"%epoch)
        pbar.set_postfix(loss_recons=loss_recons.item(), loss_vq=loss_vq.item())

        # Logs
        writer.add_scalar('loss/train/reconstruction', loss_recons.item(),
                          args.steps)
        writer.add_scalar('loss/train/quantization', loss_vq.item(),
                          args.steps)

        optimizer.step()
        args.steps += 1


def test(data_loader, model, args, writer):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, _ in data_loader:
            images = images.to(args.device)
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    # Logs
    writer.add_scalar('loss/test/reconstruction', loss_recons.item(),
                      args.steps)
    writer.add_scalar('loss/test/quantization', loss_vq.item(), args.steps)

    return loss_recons.item(), loss_vq.item()


def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde


def main(args):
    # set manualseed
    random.seed(args.manualseed)
    torch.manual_seed(args.manualseed)
    torch.cuda.manual_seed_all(args.manualseed)
    np.random.seed(args.manualseed)
    torch.backends.cudnn.deterministic = True
    
    writer = SummaryWriter(args.log_dir)
    save_filename = args.save_dir

    # if args.dataset in ['mnist', 'fashion-mnist', 'cifar10']:
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #     ])
    #     if args.dataset == 'mnist':
    #         assert args.nc == 1
    #         # Define the train & test datasets
    #         train_dataset = datasets.MNIST(args.dataroot,
    #                                        train=True,
    #                                        download=True,
    #                                        transform=transform)
    #         test_dataset = datasets.MNIST(args.dataroot,
    #                                       train=False,
    #                                       transform=transform)

    #     elif args.dataset == 'fashion-mnist':
    #         # Define the train & test datasets
    #         train_dataset = datasets.FashionMNIST(args.dataroot,
    #                                               train=True,
    #                                               download=True,
    #                                               transform=transform)
    #         test_dataset = datasets.FashionMNIST(args.dataroot,
    #                                              train=False,
    #                                              transform=transform)

    #     elif args.dataset == 'cifar10':
    #         # Define the train & test datasets
    #         train_dataset = datasets.CIFAR10(args.dataroot,
    #                                          train=True,
    #                                          download=True,
    #                                          transform=transform)
    #         test_dataset = datasets.CIFAR10(args.dataroot,
    #                                         train=False,
    #                                         transform=transform)

    #     valid_dataset = test_dataset
    # elif args.dataset == 'miniimagenet':
    #     transform = transforms.Compose([
    #         transforms.RandomResizedCrop(128),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #     ])
    #     # Define the train, valid & test datasets
    #     train_dataset = MiniImagenet(args.dataroot,
    #                                  train=True,
    #                                  download=True,
    #                                  transform=transform)
    #     valid_dataset = MiniImagenet(args.dataroot,
    #                                  valid=True,
    #                                  download=True,
    #                                  transform=transform)
    #     test_dataset = MiniImagenet(args.dataroot,
    #                                 test=True,
    #                                 download=True,
    #                                 transform=transform)

    # Define the data loaders
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=args.batch_size,
    #                                            shuffle=False,
    #                                            num_workers=args.num_workers,
    #                                            pin_memory=True)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset,
    #                                            batch_size=args.batch_size,
    #                                            shuffle=False,
    #                                            drop_last=True,
    #                                            num_workers=args.num_workers,
    #                                            pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                           batch_size=16,
    #                                           shuffle=True)

    dataloader = load_data(args)
    train_loader = dataloader['train']
    valid_loader = dataloader['valid']
    test_loader = dataloader['test']

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(valid_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    model = VectorQuantizedVAE(args.nc, args.hidden_size,
                               args.k).to(args.device)
    if len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids, output_device=args.gpu_ids[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Generate the samples first once
    reconstruction = generate_samples(fixed_images, model, args)
    grid = make_grid(reconstruction.cpu(),
                     nrow=8,
                     range=(-1, 1),
                     normalize=True)
    writer.add_image('reconstruction', grid, 0)

    best_loss = -1.
    # for epoch in range(args.num_epochs):
    for epoch in range(1, args.num_epochs+1):
        train(train_loader, model, optimizer, args, writer, epoch)
        loss, _ = test(test_loader, model, args, writer)

        reconstruction = generate_samples(fixed_images, model, args)
        grid = make_grid(reconstruction.cpu(),
                         nrow=8,
                         range=(-1, 1),
                         normalize=True)
        writer.add_image('reconstruction', grid, epoch + 1)

        if (epoch == 1) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        if (epoch % args.save_step) == 0:
            with open('{0}/model_{1}.pt'.format(save_filename, epoch),
                    'wb') as f:
                torch.save(model.state_dict(), f)


if __name__ == '__main__':

    # ARGUMENTS
    args = Options().parse()

    # Create logs and models folder if they don't exist
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.steps = 0

    main(args)
