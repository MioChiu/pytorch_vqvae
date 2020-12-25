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
from models.vq_vae import VQVAE


def train(data_loader, model, optimizer, args, writer, epoch):
    pbar = tqdm(data_loader)
    for images, _ in pbar:
        images = images.to(args.device)

        optimizer.zero_grad()
        x_tilde, loss_vq = model(images)
        loss_vq = torch.mean(loss_vq)
        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)

        loss = loss_recons + loss_vq
        loss.backward()

        pbar.set_description("epoch %s" % epoch)
        pbar.set_postfix(loss_recons=loss_recons.item(),
                         loss_vq=loss_vq.item())

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
            x_tilde, loss_vq_ = model(images)
            loss_vq_ = torch.mean(loss_vq_)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += loss_vq_

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
        x_tilde, _ = model(images)
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

    dataloader = load_data(args)
    train_loader = dataloader['train']
    valid_loader = dataloader['valid']
    test_loader = dataloader['test']

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(valid_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    # model = VectorQuantizedVAE(args.nc, args.hidden_size,
    #                            args.k).to(args.device)
    model = VQVAE(in_channels=args.nc,
                  embedding_dim=args.hidden_size,
                  num_embeddings=args.k,
                  hidden_dims=[128, 256],
                  beta=args.beta,
                  img_size=args.input_size).to(args.device)
    if len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model,
                                      device_ids=args.gpu_ids,
                                      output_device=args.gpu_ids[0])

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
    for epoch in range(1, args.num_epochs + 1):
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
