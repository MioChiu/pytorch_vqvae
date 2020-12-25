import argparse
import os
import torch


class Options():
    """Options class
    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--dataset', default='cifar10', help='folder | cifar10 | mnist ')
        self.parser.add_argument('--dataroot', default='./database/cifar10', help='path to dataset')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channels')

        self.parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
        self.parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=8)
        self.parser.add_argument('--input_size', type=int, default=32, help='input image size.')

        self.parser.add_argument('--hidden_size', type=int, default=256, help='size of the latent vector')
        self.parser.add_argument('--k', type=int, default=128, help='number of latent vectors')

        self.parser.add_argument('--device', type=str, default='cuda', help='Device: cuda | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--manualseed', default=16, type=int, help='manual seed')
        self.parser.add_argument('--abnormal_class', default='car', help='Anomaly class idx for mnist and cifar datasets')
        self.parser.add_argument('--proportion', type=float, default=0.1, help='Proportion of anomalies in test set.')
        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric.')

        ##
        # Train
        # self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
        self.parser.add_argument('--beta', type=float, default=0.25, help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')

        self.parser.add_argument('--save_step', type=int, default=5, help='frequency of saving model')
        self.parser.add_argument('--save_dir', type=str, default='./checkpoint/cifar10_new', help='save dir')
        self.parser.add_argument('--log_dir', type=str, default='./log/cifar10_new', help='log dir')
        # self.parser.add_argument('--w_adv', type=float, default=1, help='Adversarial loss weight')
        # self.parser.add_argument('--w_con', type=float, default=50, help='Reconstruction loss weight')
        # self.parser.add_argument('--w_enc', type=float, default=1, help='Encoder loss weight.')

        # self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")

        self.isTrain = True
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            int_id = int(str_id)
            if int_id >= 0:
                self.opt.gpu_ids.append(int_id)
        # set gpu ids
        self.opt.device = torch.device("cuda:{}".format(self.opt.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
        return self.opt
