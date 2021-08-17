from __future__ import print_function
import time
from tqdm import tqdm
# %matplotlib inline
import argparse
import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from IPython.display import HTML

from scipy.linalg import sqrtm
from pytorch_fid import fid_score
from pytorch_fid import *

from lib.Generator import Generator
from lib.Discriminator import Discriminator

class CatGan:
    def __init__(self, fileName=None, randomSeed=10, batch_size=128, ngf=80,
                 data_root=os.path.join(os.getcwd(), 'data/cats/real_cats'),
                 save_img_path=os.path.join(os.getcwd(), 'data/cats/gan_cats'),
                 model_save_path=os.path.join(os.getcwd(), 'data/cats/models'),
                 grid_save_path=os.path.join(os.getcwd(), 'data/cats/grid')):

        random.seed(randomSeed)
        torch.manual_seed(randomSeed)

        self.REAL_IMG_PATH = data_root
        self.SAVE_IMG_PATH = save_img_path
        self.MODEL_SAVE_PATH = model_save_path
        self.GRID_SAVE_PATH = grid_save_path

        # Number of workers for dataloader
        self.workers = 2
        # Batch size during training
        self.batch_size =batch_size
        # Spatial size of training images. All images will be resized to this
        # size using a transformer.
        self.image_size = 64
        # Number of channels in the training images. For color images this is 3
        self.nc = 3
        # Size of z latent vector (i.e. size of generator input)
        self.nz = 100
        # Size of feature maps in generator
        self.ngf = ngf
        # Size of feature maps in discriminator
        self.ndf = ngf
        # Number of training epochs
        self.num_epochs = 50
        # Learning rate for optimizers
        # self.lr = 0.0002 # For Adam
        self.lr = 0.00005
        # Beta1 hyperparam for Adam optimizers
        self.beta1 = 0.2
        #        self.beta1 = 0.5
        self.ngpu = 1

        self.total_train_epoch_ctr = 0
        self.fid_hist = { 'fid' : [], 'epoch' : [] }
        self.device = torch.device("cuda")

        if fileName:
            self.loadModel(fileName)
        else:
            self.generator = self.createGenerator()
            self.discriminator = self.createDiscriminator()

        # Initialize BCELoss function
        self.criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        # self.optimizerD = optim.SGD(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=1e-3)
        # self.optimizerG = optim.SGD(filter(lambda p: p.requires_grad, generator.parameters()), lr=1e-3)

    def loadData(self):
        # Загрузка изображений

        self.dataset = dset.ImageFolder(root=self.REAL_IMG_PATH,
                                        transform=transforms.Compose([
                                            transforms.Resize(self.image_size),
                                            transforms.CenterCrop(self.image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]))
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                      shuffle=True, num_workers=self.workers)
        self.imgIterator = torch.utils.data.DataLoader(self.dataset, batch_size=1,
                                                       shuffle=True, num_workers=self.workers)
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")



    def plotRealImageGrid(self):
        # Отображение части реальных изображений

        real_batch = next(iter(self.dataloader))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

    def weights_init(self, model):
        # Инициализация весов сети

        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)

    def plotRes(self, img_list, out_file_name=None):
        # Grid с реальными и сгенерированными изображениями

        real_batch = next(iter(self.dataloader))

        # Plot the real images
        plt.figure(figsize=(40, 40))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

        # Plot the fake images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        # plt.show()
        if out_file_name:
            plt.savefig(os.path.join(self.GRID_SAVE_PATH, out_file_name))

    def plotLoss(self, G_losses, D_losses, out_file_name=None):
        # График Loss-функция

        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        if out_file_name:
            plt.savefig(os.path.join(self.GRID_SAVE_PATH, out_file_name))

    def train(self, num_epochs, plt_frc=10, G_losses=[], D_losses=[], img_list=[]):
        # Цикл обучения

        print('Start train loop')
        for epoch in tqdm(range(num_epochs)):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):
                self.discriminator.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.generator(noise)
                label.fill_(self.fake_label)
                # Classify all fake batch with D
                output = self.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                # (2) Update G network: maximize log(D(G(z)))
                self.generator.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if epoch % plt_frc == 0 or ((epoch == num_epochs - 1) and (i == len(self.dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.generator(self.fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        self.total_train_epoch_ctr += num_epochs
        return (img_list, G_losses, D_losses)

    def createGenerator(self, nz=None, nc=None, ngf=None):
        # Создание генератора

        if not nz:
            nz = self.nz
        if not nc:
            nc = self.nc
        if not ngf:
            ngf = self.ngf

        generator = Generator(self.ngpu, nz, nc, ngf).to(self.device)
        generator.apply(self.weights_init)

        return generator

    def createDiscriminator(self, nc=None, ndf=None):
        # Создание дискриминатора

        if not nc:
            nc = self.nc
        if not ndf:
            ndf = self.ndf

        discriminator = Discriminator(self.ngpu, nc, ndf).to(self.device)
        discriminator.apply(self.weights_init)

        return discriminator

    def saveModel(self, savePath=None, fileName=None):
        # Сохранить checkpoint

        if not savePath:
            savePath = self.MODEL_SAVE_PATH
        if not fileName:
            fileName = 'GAN_torch_ngf={}_epoch={}_lr={}_beta={}.pth'.format(self.ngf,
                                                                            self.total_train_epoch_ctr, self.lr, self.beta1)

        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'total_train_epoch_ctr': self.total_train_epoch_ctr,
            'ngf' : self.ngf,
            'ndf' : self.ndf,
            'nc' : self.nc,
            'nz' : self.nz,
            'lr' : self.lr,
            'beta1' : self.beta1,
            'fid_hist' : self.fid_hist
        }, os.path.join(savePath, fileName))

    def loadModel(self, fileName, loadPath = None, ngf=None):
        # Загрузить checkpoint

        if not loadPath:
            loadPath = self.MODEL_SAVE_PATH
        device = self.device
        checkpoint = torch.load(os.path.join(loadPath, fileName))
        if ngf:
            ndf = ngf
            nc = self.nc
            nz = self.nz
            lr = self.lr
            beta1 = self.beta1
        else:
            ngf = checkpoint['ngf']
            ndf = checkpoint['ndf']
            nc = checkpoint['nc']
            nz = checkpoint['nz']
            lr = checkpoint['lr']
            beta1 = checkpoint['beta1']
            self.fid_hist = checkpoint['fid_hist']
        self.generator = Generator(self.ngpu, nz, nc, ngf)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.to(device)
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])

        self.discriminator = Discriminator(self.ngpu, nc, ndf)
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.discriminator.to(device)
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

        self.total_train_epoch_ctr = checkpoint['total_train_epoch_ctr']

        return self.generator, self.discriminator, self.optimizerG, self.optimizerD

    def getGenImg(self, save=True, show=True, count=1):
        # Сгенерировать изображения

        print('Start img gen')
        max_fake_size = 1000
        if count > max_fake_size:
            img_ctr = 0
            loop_num = int(np.ceil(count/max_fake_size))
            for _ in tqdm(range(loop_num)):
                with torch.no_grad():
                    noise = torch.randn(max_fake_size, self.nz, 1, 1, device=self.device)
                    fake = self.generator(noise)

                for i in range(max_fake_size):
                    vutils.save_image(fake[i], os.path.join(self.SAVE_IMG_PATH, '{}.jpg'.format(img_ctr+1)), normalize=True)
                    img_ctr += 1
        else:
            with torch.no_grad():
                noise = torch.randn(count, self.nz, 1, 1, device=self.device)
                fake = self.generator(noise)

            for i in tqdm(range(count)):
                vutils.save_image(fake[i], os.path.join(self.SAVE_IMG_PATH, '{}.jpg'.format(i + 1)), normalize=True)

    def getRealImg(self, count=1, show=True):
        # Получить список с реальными изображениями

        if count == 1:
            real_img = next(iter(self.imgIterator))
            img = np.transpose(real_img[0][0].cpu(), (1, 2, 0))
            if show:
                plt.figure()
                plt.axis('off')
                plt.imshow(img)
            return img
        else:
            imgLst = []
            for i in range(count):
                imgLst.append(np.transpose(next(iter(self.imgIterator))[0][0].cpu(), (1, 2, 0)))
                if show:
                    plt.figure()
                    plt.axis('off')
                    plt.imshow(imgLst[-1])
            return imgLst

    def clearDir(self):
        # Удаление всех файлов в папке со сгенерированными файлами

        path = os.path.join(self.SAVE_IMG_PATH, '*')
        files = glob.glob(path)
        for f in files:
            os.remove(f)

    def fidScore(self, pic_num=1000, real_img_path=None, gan_img_path=None, dims=2048, show=True, gen=True):
        # Расчет FID

        if gen:
            self.getGenImg(show=False, count=pic_num)
        print('Start FID score')
        if not real_img_path:
            real_img_path = os.path.join(self.REAL_IMG_PATH, 'real_cats')
        if not gan_img_path:
            gan_img_path = self.SAVE_IMG_PATH

        fid_value = fid_score.calculate_fid_given_paths([real_img_path, gan_img_path], min([self.batch_size, pic_num]), 'gpu', dims)
        self.fid_hist['fid'].append(fid_value)
        self.fid_hist['epoch'].append(self.total_train_epoch_ctr)
        if show:
            plt.figure()
            plt.plot(self.fid_hist['epoch'], self.fid_hist['fid'])

        # self.clearDir()
        return fid_value



