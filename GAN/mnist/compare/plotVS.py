import matplotlib.pyplot as plt
import pandas as pd



results = pd.read_csv("5340 HW4/GAN/mnist/compare/lossVS.csv",index_col=0) 

head=['layer_latent','epoch', 'loss_G', 'loss_D','loss_jsd']  
layer_latent =['ly3_dim64','ly3_dim128','ly4_dim64']

# Creating figure to plot
fig,axes = plt.subplots(nrows=1,ncols=1,figsize = (6,6))
fig.suptitle("GAN_mnist influenced by model complexity/latent dimension",size = 'xx-large')

model1=results[(results['layer_latent'] ==layer_latent[0])]
model2=results[(results['layer_latent'] ==layer_latent[1])]
model3=results[(results['layer_latent'] ==layer_latent[2])]

axes.plot(model1['epoch'],model1['loss_jsd'],label='ly3_dim64')
axes.plot(model1['epoch'],model2['loss_jsd'],label='ly3_dim128')
axes.plot(model1['epoch'],model3['loss_jsd'],label='ly4_dim64')
axes.set_xlabel("epoch",size = 'x-large')
axes.set_title("JSD of GAN",size = 'x-large')
axes.set_ylim( min(results['loss_jsd']) - 0.1 ,  max(results['loss_jsd'] + 0.1 ) )
axes.legend()
fig.tight_layout()
plt.show()