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

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,)),]) 
train_dataset = datasets.MNIST(root='../dataset/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)


"""#layer3
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
            )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)
"""
#layer4

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
        
            )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)
        

def  gen_img_plot(model,fix_noise): 
    generated_img = model(fix_noise).cpu().detach() #remove  gredient
    generated_img = make_grid(generated_img, padding =2, normalize = True)
    save_image(generated_img, f"./images/epoch{epoch+1}.png")


noise_z = 64   #64/128
#noise_z = 128
learning_rate= 5e-5
weight_clip = 0.01
K=5

#gen_img_plot
sample_num=64
fix_noise =torch.randn(sample_num,noise_z,device=device)

generator = Generator(noise_z).to(device)
discriminator = Discriminator().to(device)

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
            real_output = discriminator(image)  #predict real image 0/1 
            
            random_noise = torch.randn(img_size,noise_z,device=device) #128 64
            fake_img=generator(random_noise)  #fake_img
            fake_output = discriminator(fake_img.detach( )) #predict generated img 0/1
            
            loss_d = -(torch.mean(real_output) - torch.mean(fake_output)) #no log loss
            em_distance += (-loss_d)
            loss_d.backward()
            optim_d.step()
            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-weight_clip, weight_clip)
                
        #generator trian
        optim_g.zero_grad()
        random_noise = torch.randn(img_size,noise_z,device=device) #128 64
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
        layer_latent.append('ly4_dim64')
        layer_latent.append('ly4_dim128')
        #layer_latent.append('ly3_dim？？？')
        
        result = pd.DataFrame(list(zip(layer_latent,epoch_list,losses_G,losses_D,emd_list)), columns=['layer_latent','epoch','loss_G', 'loss_D', 'emd'])
        result.to_csv("mnist_loss.csv")

def plot_loss(losses_D,losses_G,emd_list):
    fig = plt.figure()
    plt.plot(epoch_list, losses_D, label = "discriminator loss")
    plt.plot(epoch_list, losses_G, label = "generator loss")
    plt.legend(loc='best', prop={'size': 10})
    plt.xlabel("epoch",size = 'x-large')
    plt.ylabel('Loss',size = 'x-large')
    plt.title(" mnist layer4 lat_dim=64\n",size ='x-large')
    #plt.title(" mnist layer4 lat_dim=128\n",size ='x-large')
    #plt.title(" mnist layer3 lat_dim=???????\n",size ='x-large')
    plt.savefig(f"./loss/loss.png")
    plt.close()


    fig = plt.figure()
    plt.plot(epoch_list, emd_list, color='black')
    plt.xlabel("epoch",size = 'x-large')
    plt.ylabel('EM Distance',size = 'x-large')
    plt.title('WGAN_MNIST_EMD',size = 'x-large')
    plt.savefig(f"./loss/wgan_emd.png")
    plt.close()
plot_loss(losses_D,losses_G,emd_list)

