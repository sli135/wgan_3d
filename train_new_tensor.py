"""
videogan PT version

Original torch ver from https://github.com/cvondrick/videogan
TF ver from https://github.com/GV1028/videogan
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torchvision.utils import make_grid
import imageio
import os
import scipy.misc
import numpy as np
import glob
#from utils import *
import sys
from argparse import ArgumentParser
from datetime import datetime
from random import shuffle
import matplotlib as mpl
mpl.use('Agg') 
import matplotlib.pyplot as plt
#from skimage.transform import resize

import batchgenerator as bg 

parser = ArgumentParser()
parser.add_argument(
    "-e", help="Epoch to train",
    type=int, default=1
)
parser.add_argument(
    "-d", help="The dimension of each video, must be of shape [1,14,7,350]",
    nargs='*', default=[1,14,7,350]
)
parser.add_argument(
    "-zd", help="The dimension of latent vector [100]",
    type=int, default=104
)
parser.add_argument(
    "-nb", help="The size of batch images [64]",
    type=int, default=20
)
parser.add_argument(
    "-l", help="The value of sparsity regularizer [0.1]",
    type=float, default=0.1
)
parser.add_argument(
    "-c", help="The checkpoint file name",
    type=str
)#, default="2021-02-10-14:04"
parser.add_argument(
    "-s", help="Saving checkpoint file, every [1] epochs",
    type=int, default=1
)
parser.add_argument(
    "-lr", type=float, default=0.0002, 
    help="adam: learning rate")
parser.add_argument(
    "-b1", type=float, default=0.5, 
    help="adam: decay of first order momentum of gradient")
parser.add_argument(
    "-b2", type=float, default=0.999, 
    help="adam: decay of first order momentum of gradient")
parser.add_argument(
    "-checkpoint", type=str, default='', 
    help="The saved check point path")
args = parser.parse_args()
print(args)
losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
gp_weight = 10
save_training_gif = True
BATCH_SIZE = 32
if len(args.checkpoint) == 0:
    checkpoint_flag = False
else:
    checkpoint_flag = True


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 64, (3,3,3), (1,1,2), (1,1,1))

        self.conv2 = nn.Conv3d(64, 128, (3,3,3), (1,1,2), (1,1,1))
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 256, (3,3,3), (1,1,2), (1,1,1))
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4 = nn.Conv3d(256, 512, (3,3,3), (1,1,2), (1,1,1))
        self.bn4 = nn.BatchNorm3d(512)

        self.conv5 = nn.Conv3d(512, 1, [1,1,1], [1,1,1],(1,1,1))

        self.features_to_prob = nn.Linear(3600, 1)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif classname.lower().find('bn') != -1:
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, l):
        lb_img = l.reshape(l.shape[0],1,1,1,4)
        lb_img = lb_img.tile(1,1,14,7,1)
        x = torch.cat((x,lb_img),4)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.conv5(x)
        x = x.view(x.size()[0], -1)
        x = self.features_to_prob(x)

        return x

class Generator(torch.nn.Module):
    def __init__(self,zdim):
        super(Generator, self).__init__()
        self.zdim = zdim # for the noise dimension
        self.leaky_relu = nn.LeakyReLU(0.2)
        # Background
        self.conv1b = nn.ConvTranspose2d(zdim, 512, [4,4], [1,1])
        self.bn1b = nn.BatchNorm2d(512)

        self.conv2b = nn.ConvTranspose2d(512, 256, [4,4], [2,2], [1,1])
        self.bn2b = nn.BatchNorm2d(256)

        self.conv3b = nn.ConvTranspose2d(256, 128, [4,4], [2,2], [1,1])
        self.bn3b = nn.BatchNorm2d(128)

        self.conv4b = nn.ConvTranspose2d(128, 64, [4,4], [2,2], [1,1])
        self.bn4b = nn.BatchNorm2d(64)

        self.conv5b = nn.ConvTranspose2d(64, 3, [4,4], [2,2], [1,1])

        # Foreground
        self.conv1 = nn.ConvTranspose3d(zdim, 512, kernel_size=[7,7,7], stride =[1,1,1])
        self.bn1 = nn.BatchNorm3d(512)

        self.conv2 = nn.ConvTranspose3d(512, 256, kernel_size=[4,3,5], stride=[2,1,5], padding=[1,1,0])
        self.bn2 = nn.BatchNorm3d(256)

        self.conv3 = nn.ConvTranspose3d(256, 128, [3,3,3], [1,1,5], [1,1,1])
        self.bn3 = nn.BatchNorm3d(128)

        self.conv4 = nn.ConvTranspose3d(128, 64, [3,3,5], [1,1,2], [1,1,0])
        self.bn4 = nn.BatchNorm3d(64)

        self.conv5 = nn.ConvTranspose3d(64, 1, [3,3,6], [1,1,1], [1,1,0])
        self.bn5 = nn.BatchNorm3d(1)
        # Mask
        self.conv5m = nn.ConvTranspose3d(64, 1, [3,3,5], [1,1,1], [1,1,0])
        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif classname.lower().find('bn') != -1:
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
    def forward(self,z,l):
        z = torch.cat((z,l),1)
        # Foreground
        f = self.leaky_relu(self.conv1(z.unsqueeze(2).unsqueeze(3).unsqueeze(4)))
        f = self.leaky_relu(self.conv2(f))
        f = self.leaky_relu(self.conv3(f))
        f = self.leaky_relu(self.conv4(f))
        f = self.leaky_relu(self.conv5(f))
        #m = torch.sigmoid(self.conv5m(f))   # b, 1, 32, 64, 64
        #f = torch.tanh(self.conv5(f))   # b, 3, 32, 64, 64
        #out = m*f 
        return f
def save_checkpoint(epoch,model,optimizer,loss,PATH = 'models/',NAME = 'model.tar'):
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, PATH+NAME)
def load_checkpoint(TYPE,PATH):
    device = torch.device("cuda")
    if TYPE == "G":
        model = Generator(zdim=args.zd)
    elif TYPE == "D":
        model = Discriminator()
    else:
        print("No such model.")
        return 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model,optimizer,epoch,loss
def wgan_gradient_penalty(discriminator,label,real_data, generated_data):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    if cuda:
        alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    label = Variable(label, requires_grad=True)
    if cuda:
        interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated,label)

    # Calculate gradients of probabilities with respect to examples
    #print(prob_interpolated.size(),interpolated.size())
    gradients = torch_grad(outputs=prob_interpolated, inputs=[interpolated,label],
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda() if cuda else torch.ones(
                           prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.reshape(batch_size, -1)
    #print('gradient size',gradients.size())
    #print(gradients.norm(2, dim=1).mean().item())
    losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()

def save_fig(wf,epochs,PATH='./genvideos/'):
    images = []
    xmax,xmin = np.max(wf),np.min(wf)
    for i in range(350):
        im = wf[:,i]
        im = im.reshape(14,7)
        neg = plt.imshow(im,cmap=plt.cm.gray_r,vmin=xmin, vmax=xmax)
        plt.colorbar(neg)
        plt.title(r't = %i [$\mu$s]'%i)
        plt.savefig(PATH+'%03d.png'%i)
        plt.clf()

    # This method will show image in any image viewer
    #im.save('out.gif', save_all=True, append_images=[im1, im2, ...])
    path = 'pics'
    pic_lst = os.listdir(path)
    pic_lst.sort()
    gif_images = []
    for name in pic_lst:
        filename = os.path.join(path, name)
        gif_images.append(imageio.imread(filename))  # read pics

    imageio.mimsave(PATH+'training_{}_epochs.gif'.format(epochs), gif_images, 'GIF', duration=0.5)
    os.system('rm '+PATH+'*.png')
# ------------ Start -------------------#
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

file_path,label_path = "train_small_set_file_se_2_new_tensor.npy","train_small_set_label_se_2.npy"
file,label = bg.load_data(file_path,label_path)
batch_generator = bg.Batch_Generator(file,label,BATCH_SIZE)


if not os.path.exists("./checkpoints"):
    os.makedirs("./checkpoints")
if not os.path.exists("./genvideos"):
    os.makedirs("./genvideos")
'''
X,Y = batch_generator.__getitem__(0)
X,Y = torch.tensor(X),torch.tensor(Y)
X,Y = Variable(X.type(Tensor)),Variable(Y.type(Tensor))
print(X.shape)
Xg,Yg = batch_generator.__getitem__(1)
Xg,Yg = torch.tensor(Xg),torch.tensor(Yg)
Xg,Yg = Variable(Xg.type(Tensor)),Variable(Yg.type(Tensor))
'''
if checkpoint_flag == False:
    Input = torch.randn(32, 100)
    Label = torch.randn(32,4)
    discriminator = Discriminator() # output size = ([b, 1])
    generator = Generator(zdim=args.zd)
    #net_output_d = discriminator(X,Label) # output size = ([32, 1, 13, 4, 19])
    #gradient_penalty = gradient_penalty(discriminator,Label,X, Xg)
    #print(gradient_penalty.size())
    #net_output_g = generator(Input,Label) # output size = b, 1, 14, 7, 350

    if cuda:
        generator.cuda()
        discriminator.cuda()
    # Optimizer
    params_G = list(filter(lambda p: p.requires_grad, generator.parameters()))
    optimizer_G = optim.Adam(params_G, lr=args.lr, betas=(args.b1, args.b2))
    params_D = list(filter(lambda p: p.requires_grad, discriminator.parameters()))
    optimizer_D = optim.Adam(params_D, lr=args.lr, betas=(args.b1, args.b2))
else: 
    generator,optimizer_G,Epoch_save,losses = load_checkpoint("G",args.checkpoint+"generator.tar")
    discriminator,optimizer_D,Epoch_save,_ = load_checkpoint("D",args.checkpoint+"discriminator.tar")
    print("Reload networks from "+args.checkpoint)
    if cuda:
        generator.cuda()
        discriminator.cuda()
#print('discriminator outputs',[i.shape for i in net_output_d])
#print('generator outputs',[i.shape for i in net_output_g])

'''
# Load pretrained
if args.c is not None:
    generator.load_state_dict(torch.load("./checkpoints/{}_G.pth".format(args.c)).state_dict(), strict=True)
    discriminator.load_state_dict(torch.load("./checkpoints/{}_D.pth".format(args.c)).state_dict(), strict=True)
    print("Model restored")
'''

start = 0
iterations = batch_generator.__len__()
indices = np.array([i for i in range(iterations - 1)])

if save_training_gif:
    # Fix latents to see how image generation improves during training
    fixed_latents = Variable(Tensor(np.random.normal(0, 1, (BATCH_SIZE, 100))))
    if cuda:
        fixed_latents = fixed_latents.cuda()
    training_progress_images = []

critic_iterations = 5
# ----------
#  Training
# ----------

for epoch in range(args.e):
    np.random.shuffle(indices)
    num_steps = 0 
    for i in indices:
        num_steps += 1
        videos,labels = batch_generator.__getitem__(i)
        videos,labels = torch.tensor(videos),torch.tensor(labels)
        videos,labels = Variable(videos.type(Tensor)),Variable(labels.type(Tensor))
        # --------------------
        # Train Discriminator 
        # --------------------
        z = Variable(Tensor(np.random.normal(0, 1, (videos.shape[0], 100))))
        generated_videos = generator(z,labels)

        if cuda:
            videos = videos.cuda()

        d_real = discriminator(videos,labels)
        d_generated = discriminator(generated_videos,labels)

        # Get gradient penalty
        gradient_penalty = wgan_gradient_penalty(discriminator,labels,videos, generated_videos)
        losses['GP'].append(gradient_penalty.item())

        # Create total loss and optimize
        optimizer_D.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        optimizer_D.step()

        # Record loss
        losses['D'].append(d_loss.item())

        # Only update generator every |critic_iterations| iterations
        if num_steps % critic_iterations == 0:
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (videos.shape[0], 100))))

            # Generate a batch of images
            gen_videos = generator(z,labels)

            # Calculate loss and optimize
            d_generated = discriminator(gen_videos,labels)
            g_loss = - d_generated.mean()
            g_loss.backward()
            optimizer_G.step()

            # Record loss
            losses['G'].append(g_loss.item())

            print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [GP loss:%f]"# [D loss: %f] [G loss: %f]"
                    % (epoch, args.e, num_steps, len(indices), d_loss.item(), g_loss.item(),gradient_penalty.item())#, d_loss.item(), g_loss.item())
                )
        save_checkpoint(epoch,generator,optimizer_G,losses,"./checkpoints/","generator.tar")
        save_checkpoint(epoch,discriminator,optimizer_D,losses,"./checkpoints/","discriminator.tar")
    if save_training_gif:
        waveform_array = gen_videos.cpu()[0].detach().numpy().reshape(98,350)
        save_fig(waveform_array,epoch,"./genvideos/")
