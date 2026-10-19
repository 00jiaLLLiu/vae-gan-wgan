import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from matplotlib import pyplot as plt
from torchvision.utils import make_grid, save_image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 128
"""transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5, 0.5),(0.5, 0.5, 0.5)),])
"""
transform = transforms.Compose([transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

train_dataset = datasets.CIFAR10(root='./DATA_CIFAR10', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)


"""
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
            #(numD_feature * 4) x 8 x 8
            
            nn.Conv2d(numD_feature * 4, numD_feature * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(numD_feature * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #(numD_feature * 8) x 4 x 4
            
            nn.Conv2d(numD_feature * 8, 1, 4, 1, 0, bias=False),
            
            )

    def forward(self, input):
            return self.main(input)


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

noise_z = 64   #64/128
#noise_z = 128
learning_rate= 5e-5
weight_clip = 0.01
nChannel = 3   #channel
numG_feature = 32   #generator feature
numD_feature = 32
K=5

#gen_img_plot
sample_num=64
fix_noise =torch.randn(sample_num,noise_z, 1 , 1 ,device=device)

generator = Generator().to(device)
generator.apply(weights_init)
discriminator = Discriminator().to(device)
discriminator.apply(weights_init)

#change optimizer
optim_g = optim.RMSprop(generator.parameters(), lr=learning_rate)
optim_d = optim.RMSprop(discriminator.parameters(), lr=learning_rate)

losses_G = [] 
losses_D = []
epoch_list =[]
emd_list=[]
layer_latent =[]

for epoch in range(200):
    epo_loss_g = 0.0
    epo_loss_d = 0.0
    
    em_distance = 0.0
    count = len(train_loader)
    
    for idx, (image, _) in enumerate(train_loader):
        image = image.to(device)
        img_size = image.size(0)  #number= batch size
        
        
        for step in range(K):
            
            #discriminator train
            optim_d.zero_grad()
            real_output = discriminator(image).view(-1)  #predict real image 0/1 
            
            random_noise = torch.randn(img_size,noise_z,1,1,device=device) #128 64
            fake_img=generator(random_noise)  #fake_img
            fake_output = discriminator(fake_img.detach( )).view(-1) #predict generated img 0/1
            
            loss_d = -(torch.mean(real_output) - torch.mean(fake_output)) #no log loss
            em_distance += (-loss_d)
            loss_d.backward()
            optim_d.step()
            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-weight_clip, weight_clip)
                
        #generator trian
        optim_g.zero_grad()
        random_noise = torch.randn(img_size,noise_z,1,1,device=device) #128 64
        fake_img=generator(random_noise)  #fake_img 
        fake_output = discriminator(fake_img)  #predict generated img 0/1 
        loss_g= -torch.mean(fake_output)
        loss_g.backward()
        optim_g.step()
        
        #with torch.no_grad():
        epo_loss_d += loss_d
        epo_loss_g += loss_g
    
    with torch.no_grad():
        epo_loss_d  /=  (K * count )
        epo_loss_g  /=   count
        epoch_emd = em_distance / (K * count)
        
        print('Epoch: {},generator loss: {} , discriminator loss : {}, EMD{}'.format(epoch+1,epo_loss_g ,epo_loss_d,epoch_emd))
        gen_img_plot(generator,fix_noise)

        
        losses_D.append(epo_loss_d.item())
        losses_G.append(epo_loss_g.item())
        emd_list.append(epoch_emd.item())
        epoch_list.append(epoch + 1)
        #layer_latent.append('ly4_dim64')
        #layer_latent.append('ly4_dim128')
        layer_latent.append('ly5_dim64')
        
        
        result = pd.DataFrame(list(zip(layer_latent,epoch_list,losses_G,losses_D,emd_list)), columns=['layer_latent','epoch','loss_G', 'loss_D', 'emd'])
        result.to_csv("cifar_loss.csv")

def plot_loss(losses_D,losses_G,emd_list):
    fig = plt.figure()
    plt.plot(epoch_list, losses_D,  label = "discriminator loss")
    plt.plot(epoch_list, losses_G,  label = "generator loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('WGAN_CIFAR10_loss')
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig(f"./loss/loss.png")
    plt.close()


    fig = plt.figure()
    plt.plot(epoch_list, emd_list)
    plt.xlabel('Epoch')
    plt.ylabel('EM Distance')
    plt.title('WGAN_CIFAR10_EMD')
    plt.savefig(f"./loss/wgan_emd.png")
    plt.close()
plot_loss(losses_D,losses_G,emd_list)

