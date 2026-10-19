import matplotlib.pyplot as plt
import pandas as pd



results = pd.read_csv("5340 HW4/WGAN/cifar/compare/lossVS.csv",index_col=0) 

head=['layer_latent','epoch', 'loss_G', 'loss_D','emd']  
layer_latent =['ly4_dim64','ly4_dim128','ly5_dim64']

# Creating figure to plot
fig,axes = plt.subplots(nrows=1,ncols=1,figsize = (10,6))
fig.suptitle("WGAN_cifar influenced by model complexity/latent dimension",size = 'xx-large')

model1=results[(results['layer_latent'] ==layer_latent[0])]
model2=results[(results['layer_latent'] ==layer_latent[1])]
model3=results[(results['layer_latent'] ==layer_latent[2])]

axes.plot(model1['epoch'],model1['emd'],label='ly4_dim64')
axes.plot(model1['epoch'],model2['emd'],label='ly4_dim128')
axes.plot(model3['epoch'],model3['emd'],label='ly5_dim64')
axes.set_xlabel("epoch",size = 'x-large')
axes.set_title("EMD of WGAN",size = 'x-large')
axes.set_ylim( min(results['emd']) - 0.01 ,  max(results['emd'] + 0.01 ) )
axes.legend()

fig.tight_layout()
plt.show()



