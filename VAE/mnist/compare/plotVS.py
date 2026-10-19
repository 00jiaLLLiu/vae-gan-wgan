import matplotlib.pyplot as plt
import pandas as pd



results = pd.read_csv("5340 HW4/VAE/mnist/compare/lossVS.csv",index_col=0) 

head=['layer_latent','epoch','loss', 'loss_recon', 'loss_kl','test_loss']  
layer_latent =['ly2_dim64','ly4_dim3','ly4_dim64']

# Creating figure to plot
fig,axes = plt.subplots(nrows=1,ncols=3,figsize = (17,6))
fig.suptitle("VAE_mnnist influenced by model complexity/latent dimension",size = 'xx-large')

model1=results[(results['layer_latent'] ==layer_latent[0])]
model2=results[(results['layer_latent'] ==layer_latent[1])]
model3=results[(results['layer_latent'] ==layer_latent[2])]

axes[0].plot(model1['epoch'],model1['loss'],label='ly2_dim64')
axes[0].plot(model1['epoch'],model2['loss'],label='ly4_dim3')
axes[0].plot(model1['epoch'],model3['loss'],label='ly4_dim64')
axes[0].set_xlabel("epoch",size = 'x-large')
axes[0].set_title("train loss of VAE",size = 'x-large')
axes[0].set_ylim( min(results['loss']) - 2 ,  max(results['loss'] + 2 ) )
axes[0].legend()

axes[1].plot(model1['epoch'],model1['loss_recon'],label='ly2_dim64')
axes[1].plot(model1['epoch'],model2['loss_recon'],label='ly4_dim3')
axes[1].plot(model1['epoch'],model3['loss_recon'],label='ly4_dim64')
axes[1].set_xlabel("epoch",size = 'x-large')
axes[1].set_title("reconstruction loss of VAE",size = 'x-large')
axes[1].set_ylim( min(results['loss_recon']) - 2 ,  max(results['loss_recon'] + 2 ) )
axes[1].legend()

axes[2].plot(model1['epoch'],model1['loss_kl'],label='ly2_dim64')
axes[2].plot(model1['epoch'],model2['loss_kl'],label='ly4_dim3')
axes[2].plot(model1['epoch'],model3['loss_kl'],label='ly4_dim64')
axes[2].set_xlabel("epoch",size = 'x-large')
axes[2].set_title("KL loss of VAE",size = 'x-large')
axes[2].set_ylim( min(results['loss_kl']) - 0.1 ,  max(results['loss_kl'] + 0.1 ) )
axes[2].legend()

fig.tight_layout()
plt.show()



