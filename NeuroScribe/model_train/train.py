
import matplotlib
from torch.optim import lr_scheduler, Adam
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import os
import numpy as np
import argparse
import json
import sys
sys.path.append("..") 
from nets.model import NeuroScribe
from load.load.data.data_loader import fif_txt_Loader

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='cnn-NeuroScribe-il')
parser.add_argument('--tasks', type=str, default='signature')
args = parser.parse_args()


dataset_name = 'SMR_BCI'

data_path1 = '../data/S1_Sessions1_3.fif'
#data_path2 = '../data/S1_S.fif'
x_path='../data/x3.txt'
y_path='../data/x3.txt'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if_sep=0
raw,  or_tr = fif_txt_Loader.load_data([data_path1], x_path, y_path, if_sep)
for i in range(748):
    or_tr[i]=or_tr[i]*20



raw_data = raw.get_data()
input_size = raw_data.shape[1] * raw_data.shape[2]

inds = np.arange(748)

test_inds = inds[700:]
train_inds = inds[:700]
X = torch.Tensor(raw_data[:748]).float()
Y = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[:748]
X, Y = X.to(device), Y.to(device)
time = str(datetime.now())
time = time.replace(' ', '_')
time = time.replace(':', '_')
time = time.replace('-', '_')
time = time.replace('.', '_')
model_save_path = '../EEG_NeuroScribe_models/' + args.name


k = 1
T = 199 / k
N = 200
learning_rate = 0.001
num_epochs = 300
batch_size = 10

param_str = "T" + str(T) + "_K" + str(k) + "_N" + str(N) + "_L" + str(learning_rate) + "_E" + str(num_epochs) + "_B" + str(batch_size)
model_save_path = model_save_path+'_'+args.tasks + '_' + '(' + dataset_name + ')_(' + param_str + ')_(' + time + ')'
os.mkdir(model_save_path)

image_save_path = model_save_path + '/images'
os.mkdir(image_save_path)

# data sets
Y = Y[:, ::k, :]
X_train = X[train_inds]
Y_train = Y[train_inds]
X_test = X[test_inds]
Y_test = Y[test_inds]

# the NeuroScribe model
NeuroScribe = NeuroScribe(T=T, l=1, N=N, state_index=np.arange(2))

print(NeuroScribe)
NeuroScribe.to(device)

optimizer = torch.optim.Adam(NeuroScribe.parameters(), lr=learning_rate)

loss_values = []
# training process
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=29999, eta_min=0.000001)


def euclidean_distance(p1, p2):

    return np.sqrt(np.sum((p1 - p2) ** 2))

def dtw(traj1, traj2, dist=euclidean_distance):
    """
    Calculate the DTW distance between two trajectories.

    Parameters:
    traj1, traj2 - numpy array of shape (n, 2) representing two trajectories, where n is the number of points and 2 is the xy coordinates.
    dist - a function that calculates the distance between two points, default to Euclidean distance.

    Back:
    dtw_distance - The DTW distance between two trajectories.
    """
    n, m = len(traj1), len(traj2)
    dtw_matrix = np.zeros((n+1, m+1))
    

    for i in range(n+1):
        dtw_matrix[i, 0] = float('inf')
    for j in range(m+1):
        dtw_matrix[0, j] = float('inf')
    dtw_matrix[0, 0] = 0
    

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dist(traj1[i-1], traj2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                         dtw_matrix[i, j-1],
                                         dtw_matrix[i-1, j-1])

    return dtw_matrix[n, m]

for epoch in range(num_epochs):
    inds = np.arange(X_train.shape[0])
    #np.random.shuffle(inds)
    NeuroScribe.train()
    #getgnet.train()
    count=0
    for ind in np.split(inds, len(inds) // batch_size):


        y_h = NeuroScribe(X_train[ind], Y_train[ind, 0, :])
        loss_display = (y_h - Y_train[ind]) ** 2
        
        abs_diff = torch.abs(y_h - Y_train[ind])
        #abs_diff[:,:,1]=0
        abs_diff=torch.mean(abs_diff)
        loss = torch.where(abs_diff < 1, 0.5 * abs_diff ** 2, abs_diff - 0.5)

        loss.backward()
        optimizer.step()
        
        scheduler.step()
        optimizer.zero_grad()
        
        
        count+=1
        print('Epoch: '+str(epoch)+'    '+'Data: '+str(count)+'    '+'Loss: '+str(float(loss_display.mean(1).mean(1).mean().item()))+'    '+'Lr: '+str(scheduler.get_lr()))



    if (epoch+1)%10==0:
        NeuroScribe.eval()
        


        test_sample_indices = test_inds[:1]
        train_sample_indices=train_inds[100:110]

        y_ht = NeuroScribe(X[test_sample_indices], Y[test_sample_indices, 0, :])
        y_rt = Y[test_sample_indices]

        y_he = NeuroScribe(X[train_sample_indices], Y[train_sample_indices, 0, :])
        y_re = Y[train_sample_indices]
        for i in range(0, len(test_sample_indices)):
            plt.figure(figsize=(8, 8))
            plt.tight_layout()

            plt.plot(y_rt[i, :, 0].detach().cpu().numpy(), y_rt[i, :, 1].detach().cpu().numpy(), c='#DB70DB',alpha=0.7, linewidth=5,label='Original Trajectory')
            plt.plot(y_ht[i, :, 0].detach().cpu().numpy(), y_ht[i, :, 1].detach().cpu().numpy(), c='blue',alpha=0.7, linewidth=5,label='Generated Trajectory')
            # plt.xticks([ (i - 1) * 0.05 for i in range(1, 21)])
            # plt.yticks([ (i - 1) * 0.05 for i in range(1, 21)])
            # plt.xticks(range(1,20,1))
            # plt.yticks(range(1,20,1))
            plt.xlabel('X')
            plt.ylabel('Y')
    
            #plt.show()
            plt.savefig(image_save_path + '/valid_img_' + str(epoch) + '_' + str(i) + '.png')
            plt.close('all')
        for i in range(0, len(train_sample_indices)):
            plt.figure(figsize=(8, 8))
            plt.tight_layout()

            plt.plot(y_re[i, :, 0].detach().cpu().numpy(), y_re[i, :, 1].detach().cpu().numpy(), c='#DB70DB',alpha=0.5, linewidth=5,label='Original Trajectory')
            if (epoch+1)%200==0:
                with open(model_save_path + '/X.txt', 'a') as f:

                    for j in list(y_he[i, :, 0].detach().cpu().numpy()):
            
                        f.write(str(j) + ' ')
                    f.write('\n')
                    f.close()
                with open(model_save_path + '/Y.txt', 'a') as f:

                    for j in list(y_he[i, :, 1].detach().cpu().numpy()):
            
                        f.write(str(j) + ' ')
                    f.write('\n')
                    f.close()
                with open(model_save_path + '/DWT.txt', 'a') as f:
    
            
                    f.write(str(dtw(y_he[i, :,:].detach().cpu().numpy(), y_re[i, :, :].detach().cpu().numpy())) )
                    f.write('\n')
                    f.close()
            plt.plot(y_he[i, :, 0].detach().cpu().numpy(), y_he[i, :, 1].detach().cpu().numpy(), c='blue',alpha=0.5, linewidth=5,label='Generated Trajectory')
            # plt.xticks(range(1,20,1))
            # plt.yticks(range(1,20,1))
            plt.legend()
            plt.xlabel('X')
            plt.ylabel('Y')
    
            #plt.show()
            plt.savefig(image_save_path + '/train_img_' + str(epoch) + '_' + str(i) + '.png')
            plt.close('all')


    # # if epoch % 2 == 0:
    x_test = X_train[np.arange(20)]
    y_test = Y_train[np.arange(20)]
    #goal_fe=getgnet(x_test)
    y_htest = NeuroScribe(x_test, y_test[:, 0, :])

    test = ((y_htest - y_test) ** 2).mean(1).mean(1)
    # loss = torch.mean((y_h - Y_train[ind]) ** 2)  # loss value
    print('Epoch: ' + str(epoch) + ', Test Error: ' + str(test.mean().item()))
    loss_values.append(str(test.mean().item()))
    torch.save(NeuroScribe, model_save_path + '/cnn-model.pt')

# write value to file
with open(model_save_path + '/test_loss.txt', 'w') as f:
    f.write(json.dumps(loss_values))
