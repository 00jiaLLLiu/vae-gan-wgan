import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
import math
from torchvision.utils import make_grid, save_image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 128

transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])  
"""
transform = transforms.Compose([transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
"""
   

train_dataset = datasets.CIFAR10(root='./DATA_CIFAR10', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
#cifar layer4
class Generator(nn.Module):
    def __init__(self):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
            nn.ConvTranspose2d( noise_z, numG_feature * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(numG_feature * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(numG_feature * 4, numG_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(numG_feature * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( numG_feature * 2, numG_feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(numG_feature),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( numG_feature, nChannel, 4, 2, 1, bias=False),
            nn.Tanh()
            )

    def forward(self, input):
            return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
            nn.Conv2d(nChannel, numD_feature, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(numD_feature, numD_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(numD_feature * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(numD_feature * 2, numD_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(numD_feature * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(numD_feature * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )

    def forward(self, input):
            return self.main(input)
"""
#cifar layer5
class Generator(nn.Module):
    def __init__(self):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
            nn.ConvTranspose2d( noise_z, numG_feature * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(numG_feature * 8),
            nn.ReLU(True),
            #(numG_feature * 8) x 4 x 4 
            
            nn.ConvTranspose2d(numG_feature * 8, numG_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(numG_feature * 4),
            nn.ReLU(True),
            #(numG_feature * 4) x 8 x 8 
            
            nn.ConvTranspose2d( numG_feature * 4, numG_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(numG_feature * 2),
            nn.ReLU(True),
            #(numG_feature * 2) x 16 x 16 
            
            nn.ConvTranspose2d( numG_feature * 2, numG_feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(numG_feature),
            nn.ReLU(True),
            #(numG_feature * 2) x 32 x 32
            
            nn.ConvTranspose2d( numG_feature, nChannel, 4, 2, 1, bias=False),
            nn.Tanh()
            #(numG_feature ) x 64 x 64
            )

    def forward(self, input):
            return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
            # input = nChannel x 64 x 64
            nn.Conv2d(nChannel, numD_feature, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #(numD_feature) x 32 x 32
            
            nn.Conv2d(numD_feature, numD_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(numD_feature * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #(numD_feature*2) x 16 x 16
            
            nn.Conv2d(numD_feature*2, numD_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(numD_feature * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #(numD_feature * 4) x 32 x 32
            
            nn.Conv2d(numD_feature * 4, numD_feature * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(numD_feature * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #(numD_feature * 8) x 64 x 64
            
            nn.Conv2d(numD_feature * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            
            )

    def forward(self, input):
            return self.main(input)
        #.view(-1, 1).squeeze(1)"""



def  gen_img_plot(model,fix_noise):  
    generated_img = model(fix_noise).cpu().detach() #remove  gredient
    generated_img = make_grid(generated_img, padding =2, normalize = True)
    save_image(generated_img, f"./images/epoch{epoch+1}.png")
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

noise_z = 128   #64/128
learning_rate=2e-4
nChannel = 3   #channel
numG_feature = 64        #generator feature 
numD_feature = 64

#gen_img_plot
sample_num=64
fix_noise =torch.randn(sample_num,noise_z, 1 , 1 ,device=device)

generator = Generator().to(device)
generator.apply(weights_init)
discriminator = Discriminator().to(device)
discriminator.apply(weights_init)

optim_g = optim.Adam(generator.parameters(), lr=learning_rate,  betas=(0.5, 0.999))
optim_d = optim.Adam(discriminator.parameters(), lr=learning_rate,  betas=(0.5, 0.999))

criterion = nn.BCELoss()  #discrimitor loss

losses_G = [] 
losses_D = []
epoch_list =[]
losses_jsd=[]
layer_latent =[]

  
for epoch in range(100):
    epo_loss_g = 0.0
    epo_loss_d = 0.0
    
    jsd =0.0
    count = len(train_loader)
    
    for idx, (image, _) in enumerate(train_loader):
        image = image.to(device)
        img_size = image.size(0)  #number= batch size
        
        random_noise = torch.randn(img_size,noise_z,1,1,device=device)
        
        #discriminator train
        optim_d.zero_grad()
        
        real_output = discriminator(image).view(-1)  #predict real image 0/1 
        d_real_loss = criterion(real_output,
                                torch.ones_like(real_output)) #hope 1
        d_real_loss.backward()
        
        fake_img=generator(random_noise)  #fake_img
        fake_output = discriminator(fake_img.detach( )).view(-1) #predict generated img 0/1
        d_fake_loss = criterion(fake_output,
                                torch.zeros_like(fake_output)) # hope 0
        d_fake_loss.backward()
        
        loss_d = d_real_loss + d_fake_loss
        jsd += 0.5 * (-loss_d + math.log(4))
        optim_d.step()
        
        #generator trian
        optim_g.zero_grad() 
        fake_output = discriminator(fake_img)  #predict generated img 0/1
        loss_g = criterion(fake_output,
                            torch.ones_like(fake_output)) # hope 1
        loss_g.backward()
        optim_g.step()
        
        #with torch.no_grad():
        epo_loss_d += loss_d
        epo_loss_g += loss_g
    
    with torch.no_grad():
        epo_loss_d  /=  count
        epo_loss_g  /=  count  
        epo_jsd = (jsd /count).item()
        
        print('Epoch: {},generator loss: {} , discriminator loss : {}, JSD loss{}'.format(epoch+1,epo_loss_g ,epo_loss_d,epo_jsd))
        gen_img_plot(generator,fix_noise)

        
        losses_D.append(epo_loss_d.item())
        losses_G.append(epo_loss_g.item())
        losses_jsd.append(epo_jsd)
        epoch_list.append(epoch + 1)
        
        #layer_latent.append('ly4_dim64')
        #layer_latent.append('ly4_dim128')
        layer_latent.append('ly5_dim64')
        
        result = pd.DataFrame(list(zip(layer_latent,epoch_list,losses_G,losses_D,losses_jsd)), columns=['layer_latent','epoch','loss_G', 'loss_D', 'loss_jsd'])
        result.to_csv("cifar_loss.csv")

def plot_loss(losses_D,losses_G,losses_jsd):
    fig = plt.figure()
    plt.plot(epoch_list, losses_D,  label = "discriminator loss")
    plt.plot(epoch_list, losses_G,  label = "generator loss")
    plt.legend(loc='best', prop={'size': 10})
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN_CIFAR10_loss')
    plt.savefig(f"./loss/generater discriminator.png")
    plt.close()


    fig = plt.figure()
    plt.plot(epoch_list, losses_jsd )
    plt.xlabel('Epoch')
    plt.ylabel('JSD')
    plt.yticks(np.arange(0, 1.05, step=0.1)) 
    plt.title('GAN_CIFAR10_JSD')
    plt.savefig(f"./loss/JSD.png")
    plt.close()
plot_loss(losses_D,losses_G,losses_jsd)
