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
train_dataset = datasets.CIFAR10(root='./DATA_CIFAR10',train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.CIFAR10(root='./DATA_CIFAR10',train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

#con3+2linear 
class VAE(nn.Module):
    def __init__(self,h_dim1,z_dim):
        super(VAE, self).__init__()

        #Encoding layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(8 * 8 * 16, h_dim1)
        self.fc_bn1 = nn.BatchNorm1d(h_dim1)
        self.fc21 = nn.Linear(h_dim1, z_dim)
        self.fc22 = nn.Linear(h_dim1, z_dim)

        #Decoding layers
        self.fc3 = nn.Linear(z_dim, h_dim1)
        self.fc_bn3 = nn.BatchNorm1d(h_dim1)
        self.fc4 = nn.Linear(h_dim1, 8 * 8 * 16)
        self.fc_bn4 = nn.BatchNorm1d(8 * 8 * 16)

        self.conv4 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def encoder(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2))).view(-1, 8 * 8 * 16)
        
        fc1 = self.relu(self.fc_bn1(self.fc1(conv3)))
        #mu, log_var
        return self.fc21(fc1), self.fc22(fc1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) 
        eps = torch.randn_like(std)
        sample = mu + eps * log_var
        return sample

    def decoder(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 8, 8)

        conv4 = self.relu(self.bn4(self.conv4(fc4)))
        conv5 = self.relu(self.bn5(self.conv5(conv4)))
        
        return self.conv6(conv5).view(-1, 3, 32, 32)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

#conv3+3linear
class VAE(nn.Module):
    def __init__(self, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        #Encoding layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(8 * 8 * 16, h_dim1)
        self.fc_bn1 = nn.BatchNorm1d(h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc_bn2 = nn.BatchNorm1d(h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)

        #Decoding layers
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc_bn4 = nn.BatchNorm1d(h_dim2)
        self.fc5 = nn.Linear(h_dim2,h_dim1)
        self.fc_bn5 = nn.BatchNorm1d(h_dim1)
        self.fc6 = nn.Linear(h_dim1, 8 * 8 * 16)
        self.fc_bn6 = nn.BatchNorm1d(8 * 8 * 16)

        self.conv4 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def encoder(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2))).view(-1, 8 * 8 * 16)
        
        fc1 = self.relu(self.fc_bn1(self.fc1(conv3)))
        fc2 = self.relu(self.fc_bn2(self.fc2(fc1)))
        #mu, log_var
        return self.fc31(fc2), self.fc32(fc2)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) 
        eps = torch.randn_like(std)
        sample = mu + eps * log_var
        return sample

    def decoder(self, z):
        fc4 = self.relu(self.fc_bn4(self.fc4(z)))
        fc5 = self.relu(self.fc_bn5(self.fc5(fc4)))
        fc6 = self.relu(self.fc_bn6(self.fc6(fc5))).view(-1, 16, 8, 8)

        conv4 = self.relu(self.bn4(self.conv4(fc6)))
        conv5 = self.relu(self.bn5(self.conv5(conv4)))
        
        return self.conv6(conv5).view(-1, 3, 32, 32)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
#conv4 
class VAE(nn.Module):
    def __init__(self,z_dim):
        super(VAE, self).__init__()

        #Encoding layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(8 * 8 * 16, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc21 = nn.Linear(512, z_dim)
        self.fc22 = nn.Linear(512, z_dim)

        #Decoding layers
        self.fc3 = nn.Linear(z_dim, 512)
        self.fc_bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 8 * 8 * 16)
        self.fc_bn4 = nn.BatchNorm1d(8 * 8 * 16)

        self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def encoder(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 8 * 8 * 16)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        #mu, log_var
        return self.fc21(fc1), self.fc22(fc1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) 
        eps = torch.randn_like(std)
        sample = mu + eps * log_var
        return sample

    def decoder(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 8, 8)

        conv4 = self.relu(self.bn4(self.conv4(fc4)))
        conv5 = self.relu(self.bn5(self.conv5(conv4)))
        
        return self.conv6(conv5).view(-1, 3, 32, 32)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var


kl_loss_fun = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
recon_loss_fun= lambda recon_x, x: F.mse_loss(recon_x, x, size_average=False)

def trainmodel():
    model.train()
    running_loss, running_loss_recon, running_loss_kl = 0.0, 0.0, 0.0
    train_num = len(train_loader.dataset)

    for idx, (x, _) in enumerate(train_loader):
        x  = x.to(device)
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
                    recon_img = recon_x.view(batch_size, 3,32,32)
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
            img = img.to(device)
            recon_x, mu, logvar = model(img)
            recon_loss = recon_loss_fun(recon_x, img)
            kl_loss = kl_loss_fun(mu, logvar)
            loss = recon_loss + kl_loss
            running_loss += loss.item()
    test_loss = running_loss / test_num
    print('test_loss',test_loss)
    return test_loss

#cifar

model=VAE(h_dim1=512,z_dim=256).to(device)    #con3+2linear  z_dim=256
#model=VAE(h_dim1=512, z_dim=512).to(device)  #con3+2linear  z_dim=512
#model=VAE(h_dim1=512, h_dim2=256, z_dim=512).to(device)  #con3+3linear  z_dim=512
 
optimizer = optim.Adam(model.parameters(), lr=0.001)

Epoch = 100
train_losses, train_losses_recon, train_losses_kl, test_losses = [],[],[],[]
epoch_list =[]
layer_latent =[]
for epoch in range(Epoch):
    epoch_list.append(epoch+1)
    layer_latent.append('ly2_dim256')
    #layer_latent.append('ly2_dim512')
    #layer_latent.append('ly3_dim512')
    train_loss,train_loss_recon,train_loss_kl = trainmodel()
    train_losses.append(train_loss)
    train_losses_recon.append(train_loss_recon)
    train_losses_kl.append(train_loss_kl)

    test_loss= testmodel()
    test_losses.append(test_loss)

result = pd.DataFrame(list(zip(layer_latent,epoch_list,train_losses,train_losses_recon,train_losses_kl,test_losses)), columns=['layer_latent','epoch','loss', 'loss_recon', 'loss_kl','test_loss'])
result.to_csv("cifar_loss.csv")
#,mode='a',header=False


def plot_loss(epoch_list,train_losses,train_losses_recon,train_losses_kl,test_losses):

    fig = plt.figure()
    plt.plot(epoch_list, train_losses_recon)
    plt.xlabel("epoch",size = 'x-large')
    plt.ylabel('Binary Cross Entropy',size = 'x-large')
    plt.title('VAE_CIFAR_Reconstruction_Loss',size = 'x-large')
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig(f"./loss/vae_recons.png")
    plt.close()

    fig = plt.figure()
    plt.plot(epoch_list, train_losses_kl)
    plt.xlabel("epoch",size = 'x-large')
    plt.ylabel(' KL Divergence',size = 'x-large')
    plt.title('VAE_CIFAR_KL_Divergence',size = 'x-large')
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig(f"./loss/vae_kl.png")
    plt.close()
    fig = plt.figure()
    
    plt.plot(epoch_list,train_losses,label= "train loss")
    plt.plot(epoch_list, test_losses,label = "test loss")
    plt.xlabel("epoch",size = 'x-large')
    plt.ylabel("loss ",size = 'x-large')
    plt.legend(loc='best', prop={'size': 10})
    plt.title(" cifar10 cov3+2Linear lat_dim=256\n",size ='x-large')
    #plt.title(" cifar10 cov3+2Linear lat_dim=512\n",size ='x-large')
    #plt.title(" cifar10 cov3+3Linear lat_dim=512\n",size ='x-large')
    plt.savefig(f"./loss/total_loss.png")
    plt.close()
    
plot_loss(epoch_list,train_losses,train_losses_recon,train_losses_kl,test_losses)

