import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from torch.utils.tensorboard import SummaryWriter



class DrivingForce(nn.Module):
    def __init__(self,inner_nodes=128,in_dim=2,out_dim=1):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.inner_nodes=inner_nodes

        self.relu_stack=nn.Sequential(
            nn.Linear(self.in_dim,self.inner_nodes),
            nn.ReLU(),
            nn.Linear(self.inner_nodes,self.inner_nodes),
            nn.ReLU(),
            nn.Linear(self.inner_nodes,self.out_dim)
        )

    def forward(self,x,t):
        state=torch.cat((x,t),dim=1)
        return self.relu_stack(state)

# time 1 will always be the target
# time 0 corresponds to the noise
class DrivingModel():
    def __init__(self,m1=10,sigma1=0.1,m0=0,sigma0=1,t0=0,t1=1,in_dim=2,inner_nodes=256,out_dim=1,k=1,device='cpu',learning_rate=1e-3):
        self.t0=t0
        self.t1=t1
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.inner_nodes=inner_nodes
        self.m1=m1
        self.sigma1=sigma1
        self.m0=m0
        self.sigma0=sigma0
        self.device=device
        self.k=k

        self.driving_network=DrivingForce(self.inner_nodes,self.in_dim,self.out_dim).to(self.device)

        self.probability_flow=ProbabiltyFlow(self.driving_network,self.k)
        self.true_probability_flow=TrueProbabiltyFlow(self.m1,self.sigma1,self.k)
        self.HeatGeneration=HeatGeneration(self.driving_network,self.k,self.sigma0,self.m0)
        self.optimizer=optim.Adam(self.driving_network.parameters(),lr=learning_rate)

    def compute_score_loss(self,nb_samples):
        ts=torch.rand(nb_samples).unsqueeze(-1) # uniform sampling
        mt=torch.exp(- self.k * ts) * self.m1
        vart = (- 1 + torch.exp( 2 * self.k * ts) + self.sigma1 ** 2) * torch.exp(-2 * self.k * ts)
        xs=torch.randn_like(ts) * torch.sqrt(vart) + mt
        scores = - (xs - mt) / vart

        #watch out for time reversal here
        forces=self.driving_network(xs,1-ts)
        loss=(forces - scores) ** 2
        return torch.mean(loss)

    def compute_mean_heat(self,nb_samples,nb_dynamic_steps):
        return self.HeatGeneration.compute_heat(nb_samples,nb_dynamic_steps)



    # runs the CNF augmented ODE to get samples and log probabilities
    def compute_log_prob(self,samples):
        p_z0 = torch.distributions.Normal(loc=torch.ones(1) * self.m0, scale=torch.ones(1) * self.sigma0)
        z_t0 = p_z0.sample((samples,)).to(self.device)
        logp_t0 = p_z0.log_prob(z_t0).to(self.device)
            #integrate ODE from t0 to t1
        z_t, logp_t = odeint(
                self.probability_flow,
                (z_t0, logp_t0),
                torch.tensor([self.t0, self.t1]).type(torch.float32).to(self.device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )
        return  z_t0, z_t, logp_t0,logp_t

    # computes system entropy change associated to trained model
    def compute_system_entropy_change(self,samples):
        _ ,_, logp_t0,logp_t = self.compute_log_prob(self,samples)
        return  torch.mean(logp_t0 - logp_t[1])

    # compute KL divergence associated to trained model
    def kullback_leibler(self,samples):
        with torch.no_grad():
            _ ,z_t1, logp_t0,logp_t = self.compute_log_prob(self,samples)
            p_z1 = torch.distributions.Normal(loc=torch.ones(1) * self.m1, scale=torch.ones(1) * self.sigma1)
            logp_target = p_z1.log_prob(z_t1).to(self.device)
        return  torch.mean(logp_t - logp_target)


    # using the true score
    def compute_true_system_entropy_change(self,samples):
        p_z0 = torch.distributions.Normal(loc=torch.ones(1) * self.m0, scale=torch.ones(1) * self.sigma0)
        z_t0 = p_z0.sample((samples,)).to(self.device)
        logp_t0 = p_z0.log_prob(z_t0).to(self.device)
            #integrate ODE from t0 to t1
        _, logp_t = odeint(
                self.true_probability_flow,
                (z_t0, logp_t0),
                torch.tensor([self.t0, self.t1]).type(torch.float32).to(self.device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )

        return  torch.mean(logp_t0 - logp_t[1])

    def loss(self,epsilon):
        # loss associated to the diference with the score
        score_loss_samples=64
        score_loss=self.compute_score_loss(score_loss_samples)

        #compute system entropy difference
        entropy_nb_samples=64
        sys_entropy_change=self.compute_system_entropy_change(entropy_nb_samples)

        #compute heat along trajectory
        heat_nb_samples=64
        heat_nb_dynamic_steps=100
        Q=self.compute_mean_heat(heat_nb_samples,heat_nb_dynamic_steps)
        total_entropy_change = sys_entropy_change - Q / self.k
        return (1-epsilon) * total_entropy_change + epsilon * score_loss

    def train(self,nb_epochs,epsilon,logdir='./logs'):
        writer = SummaryWriter(log_dir=logdir)
        for epoch in range(nb_epochs):
            self.optimizer.zero_grad()
            loss=self.loss(epsilon)
            loss.backward()
            self.optimizer.step()
            writer.add_scalar("Minimal Total Entropy Matching Loss", loss.item(), epoch)
        #     if epoch%100==0:
        #         print('epoch',epoch,'loss',loss.item())
        # print('epoch',epoch,'loss',loss.item())

    def generate_trajectories(self,nb_samples,nb_steps):
        ts = torch.linspace(0,1,nb_steps)
        dts = torch.diff(ts)

        # generate trajectories
        positions=torch.randn(nb_samples,nb_steps) * self.sigma0 + self.m0
        x0=positions[:,0].clone().unsqueeze(-1) #(dataset_size,1)
        for i,t in enumerate(ts[1:]):
            time = torch.ones_like(x0) * t
            drift =  self.k * x0 + 2 * self.k * self.driving_network(x0,time)
            noise = torch.randn_like(x0)
            x0 = x0 + drift * dts[i] + noise * torch.sqrt(2 * self.k * torch.abs(dts[i]))
            positions[:,i+1]=x0.squeeze()
        return ts,positions

    def generate_true_trajectories(self,nb_samples,nb_steps):
        ts = torch.linspace(0,1,nb_steps)
        dts = torch.diff(ts)

        # generate trajectories
        positions=torch.randn(nb_samples,nb_steps) * self.sigma0 + self.m0
        x0=positions[:,0].clone().unsqueeze(-1) #(dataset_size,1)
        for i,t in enumerate(ts[1:]):
            time = torch.ones_like(x0) * t
            mt=torch.exp(- self.k * (1-time)) * self.m1
            vart = (- 1 + torch.exp( 2 * self.k * (1-time)) + self.sigma1 ** 2) * torch.exp(-2 * self.k * (1-time))
            score = - ( x0 - mt ) / vart
            drift =  self.k * x0 + 2 * self.k * score
            noise = torch.randn_like(x0)
            x0 = x0 + drift * dts[i] + noise * torch.sqrt(2 * self.k * torch.abs(dts[i]))
            positions[:,i+1]=x0.squeeze()
        return ts,positions

    # via exact score driving
    def compute_true_mean_heat(self,nb_samples,nb_dynamic_steps):
        ts = torch.linspace(0,1,nb_dynamic_steps)
        dts = torch.diff(ts)


        # generate trajectories
        positions=torch.randn(nb_samples,nb_dynamic_steps) * self.sigma0 + self.m0
        x0=positions[:,0].clone().unsqueeze(-1) #(dataset_size,1)
        for i,t in enumerate(ts[1:]):

            time = torch.ones(nb_samples,1) * t
            mt=torch.exp(- self.k * (1-time)) * self.m1
            vart = (- 1 + torch.exp( 2 * self.k * (1-time)) + self.sigma1 ** 2) * torch.exp(-2 * self.k * (1-time))
            score = - ( x0 - mt ) / vart
            drift =  self.k * x0 + 2 * self.k * score
            noise = torch.randn_like(x0)
            x0 = x0 + drift * dts[i] + noise * torch.sqrt(2 * self.k * torch.abs(dts[i]))
            positions[:,i+1]=x0.squeeze()


        # compute infinitesimal heat along trajectories
        position_increments = torch.diff(positions,dim=1) #(dataset_size,nbsteps-1)
        midpoints = (positions[:,1:] + positions[:,:-1])/2 #(dataset_size,nbsteps-1)
        delta_heat = torch.zeros_like(position_increments)
        for i in range(len(delta_heat[0])):
            time = torch.ones(nb_samples,1) * ts[i]
            midpoint = midpoints[:,i].unsqueeze(-1)
            mt=torch.exp(- self.k * (1-time)) * self.m1
            vart = (- 1 + torch.exp( 2 * self.k * (1-time)) + self.sigma1 ** 2) * torch.exp(-2 * self.k * (1-time))
            score = - ( midpoint - mt ) / vart
            strato_force = - self.k * midpoint - 2 * self.k * score
            delta_heat[:,i] =  position_increments[:,i] * strato_force.squeeze()
        Q = torch.mean(torch.sum(delta_heat,1))
        return Q

    # via exact score driving from somewhere on the energetic geodesic
    def compute_true_system_entropy_change_v2(self):
        return np.log(self.sigma1 / self.sigma0)

class ProbabiltyFlow(nn.Module):
    def __init__(self, force_network : DrivingForce, k ):
        super().__init__()
        self.force_network = force_network
        self.k = k

    def forward(self, t, states):
        z = states[0]
        batchsize = z.shape[0]
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            #implement the ODE
            dz_dt = self.k * z + self.k * self.force_network(z,torch.ones_like(z) * t)
            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)
        return (dz_dt, dlogp_z_dt)

class TrueProbabiltyFlow(nn.Module):
    def __init__(self, m,sigma, k ):
        super().__init__()
        self.m=m
        self.sigma=sigma
        self.k = k

    def forward(self, t, states):
        z = states[0]
        # logp_z = states[1]
        batchsize = z.shape[0]
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            #implement the ODE
            mt=torch.exp(- self.k * (1-t)) * self.m
            vart = (- 1 + torch.exp( 2 * self.k * (1-t)) + self.sigma ** 2) * torch.exp(-2 * self.k * (1-t))
            score = - ( z - mt ) / vart
            dz_dt = self.k * z + self.k * score
            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)
        return (dz_dt, dlogp_z_dt)
# computes the ODE term for the log likelihood

def trace_df_dz(f, z):
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()

class HeatGeneration(nn.Module):
    def __init__(self, force_network : DrivingForce,k,sigma0,m0):
        super().__init__()
        self.force_network = force_network
        self.k = k
        self.sigma0=sigma0
        self.m0=m0

    #should we take the same trajectories as for the system entropy production? Probability at the level of means it does not matter
    def compute_heat(self,nb_samples,nb_dynamic_steps):
        ts = torch.linspace(0,1,nb_dynamic_steps)
        dts = torch.diff(ts)


        # generate trajectories
        positions=torch.randn(nb_samples,nb_dynamic_steps) * self.sigma0 + self.m0
        x0=positions[:,0].clone().unsqueeze(-1) #(dataset_size,1)
        for i,t in enumerate(ts[1:]):

            time = torch.ones(nb_samples,1) * t
            drift =  self.k * x0 + 2 * self.k * self.force_network(x0,time)
            noise = torch.randn_like(x0)
            x0 = x0 + drift * dts[i] + noise * torch.sqrt(2 * self.k * torch.abs(dts[i]))
            positions[:,i+1]=x0.squeeze()


        # compute infinitesimal heat along trajectories
        position_increments = torch.diff(positions,dim=1) #(dataset_size,nbsteps-1)
        midpoints = (positions[:,1:] + positions[:,:-1])/2 #(dataset_size,nbsteps-1)
        delta_heat = torch.zeros_like(position_increments)
        for i in range(len(delta_heat[0])):
            time = torch.ones(nb_samples,1) * ts[i]
            midpoint = midpoints[:,i].unsqueeze(-1)
            strato_force = - self.k * midpoint - 2 * self.k * self.force_network(midpoint,time)
            delta_heat[:,i] =  position_increments[:,i] * strato_force.squeeze()
        Q = torch.mean(torch.sum(delta_heat,1))

        return Q
