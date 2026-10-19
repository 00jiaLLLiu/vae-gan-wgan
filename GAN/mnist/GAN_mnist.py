import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from matplotlib import pyplot as plt

import numpy as np
import math
from torchvision.utils import make_grid, save_image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 128

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,)),]) 
train_dataset = datasets.MNIST(root='../dataset/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)


#layer3
class Generator(nn.Module):
    def __init__(self, noise_z):
            super(Generator, self).__init__()
            self.noise_z = noise_z
            self.main = nn.Sequential(
                nn.Linear(self.noise_z, 256),
                nn.LeakyReLU(0.2), 

                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),

                nn.Linear(512, 28*28),
                nn.Tanh(),
          )

    def forward(self, x): #x = noise_z
        img = self.main(x)
        return img.view(-1, 1, 28, 28)   

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784
        self.main = nn.Sequential(
                    nn.Linear(self.n_input, 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Linear(256, 1),
                    nn.Sigmoid(),
            )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)

#layer4
"""
class Generator(nn.Module):
    def __init__(self, noise_z):
            super(Generator, self).__init__()
            self.noise_z = noise_z
            self.main = nn.Sequential(
                nn.Linear(self.noise_z, 256),
                nn.LeakyReLU(0.2), 

                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),

                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2),

                nn.Linear(1024, 28*28),
                nn.Tanh(),
          )

    def forward(self, x): #x = noise_z
        img = self.main(x)
        return img.view(-1, 1, 28, 28)   


#leakyRelu   x>0 -> f(x)=0 ;  x<0 f(x)=0.2*x
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784
        self.main = nn.Sequential(
                    nn.Linear(self.n_input, 1024),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Linear(256, 1),
                    nn.Sigmoid(),
            )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)"""

def  gen_img_plot(model,fix_noise):  
    generated_img = model(fix_noise).cpu().detach() #remove  gredient
    generated_img = make_grid(generated_img, padding =2, normalize = True)
    save_image(generated_img, f"./images/epoch{epoch+1}.png")


noise_z = 64   #64/128
learning_rate=2e-4

generator = Generator(noise_z).to(device)
discriminator = Discriminator().to(device)

optim_g = optim.Adam(generator.parameters(), lr=learning_rate)
optim_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

criterion = nn.BCELoss()  #discrimitor loss

losses_G = [] 
losses_D = []
epoch_list =[]
losses_jsd=[]
layer_latent =[]

#gen_img_plot
sample_num=64
fix_noise =torch.randn(sample_num,noise_z,device=device)
  
for epoch in range(100):
    epo_loss_g = 0.0
    epo_loss_d = 0.0
    
    jsd =0.0
    count = len(train_loader)
    
    for idx, (image, _) in enumerate(train_loader):
        image = image.to(device)
        img_size = image.size(0)  #number= batch size
        
        random_noise = torch.randn(img_size,noise_z,device=device)
        
        #discriminator train
        optim_d.zero_grad()
        
        real_output = discriminator(image)  #predict real image 0/1 
        d_real_loss = criterion(real_output,
                                torch.ones_like(real_output)) #hope 1
        d_real_loss.backward()
        
        gen_img=generator(random_noise)  #fake_img
        fake_output = discriminator(gen_img.detach( )) #predict generated img 0/1
        d_fake_loss = criterion(fake_output,
                                torch.zeros_like(fake_output)) # hope 0
        d_fake_loss.backward()
        
        loss_d = d_real_loss + d_fake_loss
        jsd += 0.5 * (-loss_d + math.log(4))
        
        optim_d.step()
        
        #generator trian
        optim_g.zero_grad() 
        fake_output = discriminator(gen_img)  #predict generated img 0/1
        loss_g = criterion(fake_output,
                            torch.ones_like(fake_output)) # hope 1
        loss_g.backward()
        optim_g.step()
        
        #with torch.no_grad():
        epo_loss_d += loss_d
        epo_loss_g += loss_g
    
    #with torch.no_grad():
    epo_loss_d  /=  count
    epo_loss_g  /=  count  
    epo_jsd = (jsd /count).item()
        
    print('Epoch: {},generator loss: {} , discriminator loss : {}, JSD loss{}'.format(epoch+1,epo_loss_g ,epo_loss_d,epo_jsd))
    gen_img_plot(generator,fix_noise)

        
    losses_D.append(epo_loss_d.item())
    losses_G.append(epo_loss_g.item())
    losses_jsd.append(epo_jsd)
    epoch_list.append(epoch + 1)
    
    layer_latent.append('ly3_dim64')
    #layer_latent.append('ly4_dim64')
    #layer_latent.append('ly3_dim128')
        
        
    result = pd.DataFrame(list(zip(layer_latent,epoch_list,losses_G,losses_D,losses_jsd)), columns=['layer_latent','epoch','loss_G', 'loss_D', 'loss_jsd'])
    result.to_csv("mnist_loss.csv")
    
def plot_loss(losses_D,losses_G,losses_jsd):
    fig = plt.figure()
    plt.plot(epoch_list, losses_D,  label = "discriminator loss")
    plt.plot(epoch_list, losses_G,  label = "generator loss")
    plt.legend(loc='best', prop={'size': 10})
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN_MNIST_loss')
    plt.savefig(f"./generater discriminator.png")
    plt.close()


    fig = plt.figure()
    plt.plot(epoch_list, losses_jsd )
    plt.xlabel('Epoch')
    plt.ylabel('JSD')
    plt.yticks(np.arange(0, 1.05, step=0.1)) 
    plt.title('GAN_MNIST_JSD')
    plt.savefig(f"./JSD.png")
    plt.close()
plot_loss(losses_D,losses_G,losses_jsd)