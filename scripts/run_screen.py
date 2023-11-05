from minimal_entropy_generator import *
import torch

### GENERAL PARAMETERS ###

t0=0
t1=1

k=1

# target and base distribution parameters
m1=10
sigma1=0.1
m0=3.6787944117144233 # true values along energetic geodesic for k=1 and time=1
sigma0=0.9306009185444389
# m0=0
# sigma0=1

#model parameters
in_dim=2
inner_nodes=256
out_dim=1
device='cpu'
nb_epochs=10
epsilon=0.8 # fraction of loss given to score loss
learning_rate=3e-4

#define model
model = DrivingModel(m1,sigma1,m0,sigma0,t0,t1,in_dim,inner_nodes,out_dim,k,device,learning_rate)

#train and save model
model.train(nb_epochs,epsilon=epsilon)
network=model.driving_network
# torch.save(network.state_dict(), '../models/rg_OU_eps_{epsilon}')

# compute total entropy protdutcion
# and KL divergence
nbsamples=1000
with torch.no_grad():
    q1=model.compute_mean_heat(nbsamples,100)
    delta_s_sys_1=model.compute_system_entropy_change(nbsamples)
    delta_s_tot_1=delta_s_sys_1 - q1 / k
    KL = model.kullback_leibler(nbsamples)

# print or save results (depending on what we want to do)
print(f'DeltaS_tot : {delta_s_tot_1}')
print(f'KL divergence with target : {KL}')