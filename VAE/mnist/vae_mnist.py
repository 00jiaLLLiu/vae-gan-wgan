import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 

import pandas as pd
from torchvision.utils import save_image,  make_grid

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 64

transform = transforms.Compose([transforms.ToTensor(),]) 
train_dataset = datasets.MNIST(root='../dataset/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

#layer4
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3, z_dim):
        super(VAE, self).__init__()

        #Encoding layers
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, h_dim3)
        self.fc4_mu = nn.Linear(h_dim3, z_dim)
        self.fc4_log_var = nn.Linear(h_dim3, z_dim)

        #Decoding layers
        self.fc5 = nn.Linear(z_dim, h_dim3)
        self.fc6 = nn.Linear(h_dim3, h_dim2)
        self.fc7 = nn.Linear(h_dim2, h_dim1)
        self.fc8 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        mu = self.fc4_mu(h)
        log_var = self.fc4_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) 
        eps = torch.randn_like(std)
        sample = mu + eps * log_var
        return sample

    def decoder(self, z):
        h = F.relu(self.fc5(z))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        reconstruction = torch.sigmoid(self.fc8(h))
        return reconstruction

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        #mu, log_var = self.encoder(x.view(-1, 3*32*32))
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

#layer2
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, z_dim): #h_dim2
        super(VAE, self).__init__()
        
        #Encoding layers
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2_mu = nn.Linear(h_dim1, z_dim)
        self.fc2_log_var = nn.Linear(h_dim1, z_dim)

        #Decoding layers
        self.fc3 = nn.Linear(z_dim, h_dim1)
        self.fc4 = nn.Linear(h_dim1, x_dim)

    
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        log_var = self.fc2_log_var(h)
        return mu, log_var



    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) 
        eps = torch.randn_like(std)
        sample = mu + eps * log_var
        return sample

    def decoder(self, z):
        h = F.relu(self.fc3(z))
        reconstruction = torch.sigmoid(self.fc4(h))
        return reconstruction


    def forward(self, x):
        #mnist
        mu, log_var = self.encoder(x.view(-1, 784))
        #cifar
        #mu, log_var = self.encoder(x.view(-1, 3*32*32))
        z= self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

kl_loss_fun = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#mnist
recon_loss_fun= lambda recon_x, x: F.binary_cross_entropy(recon_x, x, size_average=False)
#cifar
#recon_loss_fun= lambda recon_x, x: F.mse_loss(recon_x, x, size_average=False)

def trainmodel():
    model.train()
    running_loss, running_loss_recon, running_loss_kl = 0.0, 0.0, 0.0
    train_num = len(train_loader.dataset)

    for idx, (x, _) in enumerate(train_loader):
        #mnist
        x = x.to(device).reshape(-1, 784)
        #cifar
        #x = x.to(device).reshape( -1, 3*32*32)
        optimizer.zero_grad()

        recon_x, mu, logvar = model(x)
        recon_loss = recon_loss_fun(recon_x, x)
        kl_loss = kl_loss_fun(mu, logvar)

        loss = recon_loss + kl_loss
        loss.backward()
        running_loss += loss.item()
        running_loss_recon += recon_loss.item()
        running_loss_kl += kl_loss.item()
        optimizer.step()
        # save the last batch input and output of every epoch
        if idx == int(train_num/train_loader.batch_size) - 1:
                    recon_img = recon_x.view(batch_size, 1, 28, 28)
                    generated_img = make_grid(recon_img, padding =2, normalize = True)
                    save_image(generated_img, f"./images/epoch{epoch+1}.png")
    train_loss = running_loss/train_num
    train_loss_recon = running_loss_recon/train_num
    train_loss_kl = running_loss_kl/train_num
    
    print('Epoch: {},loss : {} , loss_recon : {}, loss_kl:{}'.format(epoch+1,train_loss ,train_loss_recon,train_loss_kl))

    return train_loss,train_loss_recon,train_loss_kl

def testmodel():
    model.eval()
    running_loss= 0.0
    test_num = len(test_loader.dataset)
    with torch.no_grad():
        for idx, (img, _) in enumerate(test_loader):
            #mnist
            img = img.to(device).reshape(-1, 784)
            #cifar
            #img = img.to(device).reshape( -1, 3*32*32)
            recon_x, mu, logvar = model(img)
            recon_loss = recon_loss_fun(recon_x, img)
            kl_loss = kl_loss_fun(mu, logvar)
            loss = recon_loss + kl_loss
            running_loss += loss.item()
            
    test_loss = running_loss / test_num
    print('test_loss',test_loss)
    return test_loss

x_dim=784
"""h_dim1= 512
h_dim2=256
h_dim3=128"""
h_dim1=512 

#z_dim=3  
z_dim= 64
model = VAE(x_dim, h_dim1, z_dim).to(device)
#model = VAE(x_dim, h_dim1, h_dim2,h_dim3, z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

Epoch = 100
train_losses, train_losses_recon, train_losses_kl, test_losses = [],[],[],[]
epoch_list =[]
layer_latent =[]
for epoch in range(Epoch):
    epoch_list.append(epoch+1)
    layer_latent.append('ly2_dim64')
    #layer_latent.append('ly4_dim3')
    #layer_latent.append('ly4_dim64')
    train_loss,train_loss_recon,train_loss_kl = trainmodel()
    train_losses.append(train_loss)
    train_losses_recon.append(train_loss_recon)
    train_losses_kl.append(train_loss_kl)

    test_loss= testmodel()
    test_losses.append(test_loss)
    

result = pd.DataFrame(list(zip(layer_latent,epoch_list,train_losses,train_losses_recon,train_losses_kl,test_losses)), columns=['layer_latent','epoch','loss', 'loss_recon', 'loss_kl','test_loss'])
result.to_csv("mnist_loss.csv")
# result.to_csv("cifar_loss.csv")
#,mode='a',header=False

def plot_loss(epoch_list,train_losses,train_losses_recon,train_losses_kl,test_losses):

    fig = plt.figure()
    plt.plot(epoch_list, train_losses_recon)
    plt.xlabel("epoch",size = 'x-large')
    plt.ylabel('Binary Cross Entropy',size = 'x-large')
    plt.title('VAE_MNIST_Reconstruction_Loss',size = 'x-large')
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig(f"./loss/vae_recons.png")
    plt.close()

    fig = plt.figure()
    plt.plot(epoch_list, train_losses_kl)
    plt.xlabel("epoch",size = 'x-large')
    plt.ylabel(' KL Divergence',size = 'x-large')
    plt.title('VAE_MNIST_KL_Divergence',size = 'x-large')
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig(f"./loss/vae_kl.png")
    plt.close()
    
    fig = plt.figure()
    plt.plot(epoch_list,train_losses,label= "train loss")
    plt.plot(epoch_list, test_losses,label = "test loss")
    plt.xlabel("epoch",size = 'x-large')
    plt.ylabel("loss ",size = 'x-large')
    #plt.title("layer=4 latent dimesion=64 \n(loss= Reconstruction Loss + KL Divergence)",size ='x-large')
    #plt.title("layer=4 latent dimesion=3 \n(loss= Reconstruction Loss + KL Divergence)",size ='x-large')
    plt.title("layer=2 latent dimesion=64 \n(loss= Reconstruction Loss + KL Divergence)",size ='x-large')
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig(f"./loss/total_loss.png")
    plt.close()
   
plot_loss(epoch_list,train_losses,train_losses_recon,train_losses_kl,test_losses)