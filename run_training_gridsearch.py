

import os, sys, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

import mne

import matplotlib
import matplotlib.pyplot as plt
import time

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from tools.train import eval_model_cl
from tools.data import unfold_raw, rm_repeated_annotations
from tools.load_data import load_data

from models.eeg_encoder import EEGEncoder
from models.envelope_encoder import EnvelopeEncoder
from models.contrastive_eeg_speech import CLEE


# Read data
subj_ids = [1] #list(range(1, 20))
fs = 128
window_size = int(5 * fs)
stride_size_train, stride_size_val, stride_size_test = int(2.5 * fs), int(5 * fs), int(5 * fs)
batch_size = int(32)
lr = 0.001

n_channs = 129 # 128 for eeg, 1 for env
print('-------------------------------------')
print(f'window_size: {window_size}  stride_size_test: {stride_size_test}')

dataset_name = ['rochester_data', 'natural_speech']
outputs_path = f'../outputs/'
data_path = os.path.join(outputs_path, dataset_name[0], dataset_name[1])
after_ica_path = os.path.join(data_path, 'after_ica_raw')
print(f'data_path: {data_path}')


X = load_data(subj_ids, after_ica_path, window_size, 
              stride_size_train, stride_size_val, stride_size_test, n_channs)


# Create dataloaders
class MyDataset(Dataset):
    def __init__(self, eeg, env):
        self.eeg = eeg
        self.env = env
    
    def __getitem__(self, index):
        return self.eeg[index], self.env[index]
    
    def __len__(self):
        return len(self.eeg)
    

dataset_train = MyDataset(X['eegs_train'], X['envs_train'])
dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
dl_val = DataLoader(MyDataset(X["eegs_val"], X["envs_val"]), 
                    batch_size=batch_size, shuffle=True, drop_last=True)



# Create model
eeg_encoder = EEGEncoder()
env_encoder = EnvelopeEncoder()
model = CLEE(eeg_encoder, env_encoder)
model.to(device)

# Train model
models_dict = {'SIMPLE': model}
lossi = []
udri = [] # update / data ratio 
ud = []



for name, model in models_dict.items():

    # Reset for the new model in the loop
    print(f"+--------------New model: {name}----------------------+")
    writer = SummaryWriter(log_dir=f"runs/{name}_{time.strftime('%Y%m%d_%H%M%S')}")
    model.to(device)
    optimizer = optim.NAdam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.7)
    cnt = 0
    loss_batches = []


    for epoch in range(1, 50):

        print(f"====== Epoch: {epoch}")

        model.train()
        for ix_batch, (Xb_eeg, Xb_env) in enumerate(dl_train):

            # send to device
            Xb_eeg = Xb_eeg.to(device)
            Xb_env = Xb_env.to(device)

            # Zero out gradients
            optimizer.zero_grad()

            # forward pass
            eeg_features, env_features, logit_scale = model(Xb_eeg, Xb_env) 


            # normalize features
            eeg_features_n = eeg_features / eeg_features.norm(dim=1, keepdim=True)
            env_features_n = env_features / env_features.norm(dim=1, keepdim=True)

            # logits
            logits_per_eeg = logit_scale * eeg_features_n @ env_features_n.t()
            logits_per_env = logits_per_eeg.t()

            #loss function
            labels = torch.arange(batch_size).to(device)
            loss_eeg = F.cross_entropy(logits_per_eeg, labels)
            loss_env = F.cross_entropy(logits_per_env, labels)
            loss   = (loss_eeg + loss_env)/2

            # backward pass
            loss.backward()
            optimizer.step()

            loss_batches.append(loss.item())
            cnt += 1

            with torch.no_grad():
                #ud = {f"p{ix}":(lr*p.grad.std() / p.data.std()).log10().item() for ix, p in enumerate(model.parameters()) if p.ndim==4 }
                #writer.add_scalars('UpdateOData/ud', ud, cnt)
                writer.add_scalar('Loss/train_batch', loss.item(), cnt)

            # normalize weights
            with torch.no_grad():
                model.eeg_encoder.normalize_weights()
            
            #break   

        loss_epoch = loss_batches[-(ix_batch + 1):]  # mean loss across batches
        loss_epoch = sum(loss_epoch) / len(loss_epoch)
        writer.add_scalar('Loss/train_epoch', loss_epoch, epoch)
        #for pname, p in model.named_parameters():
        #writer.add_histogram(f'Params/{pname}', p, epoch)
        #writer.add_histogram(f'Grads/{pname}', p.grad, epoch)

        loss_val, *_ = eval_model_cl(dl_val, model, device=device)
        writer.add_scalar('Loss/val_epoch', loss_val, epoch)

        

        model.train()

        # Update learning rate based on epoch
        scheduler.step()
            
    #break   





