# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:19:04 2022

@author: p.tung; z.rao
"""

import random  # 添加 random 模块导入
from torch.utils.data import Dataset, DataLoader  # 导入 DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class FeatureDataset(Dataset): #from numpy to tensor (pytroch-readable)
    '''
    Args: x is a 2D numpy array [x_size, x_features]
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.y[idx])
    
class AttributeDataset(Dataset): # this is for classifier 
    '''
    Args: x is a 2D numpy array [x_size, x_features]
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return torch.Tensor(self.x[idx]), torch.Tensor(self.y[idx])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def same_seeds(seed): #fix np & torch seed to the same.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_latents(model, dataset, device=None):  #from dataset to altten, 添加 device 参数
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 默认 device
        
    model.to(device).eval()  # training model or evaluation mode, eval means setting the model to its evaluation mode (gradient fixed)    
    latents = []
    with torch.no_grad(): # fix the gradient, assure that the model parameters are fixed
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        for i, data in enumerate(dataloader):
            x = data[0].to(device)
            recon_x, z = model(x)
            latents.append(z.detach().cpu().numpy())
    return np.concatenate(latents, axis=0)

def imq_kernel(X: torch.Tensor, Y: torch.Tensor, h_dim: int, device=None):  # 添加 device 参数
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 默认 device

    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t()).to(device)  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x  # mm matrix multiplication

    norms_y = Y.pow(2).sum(1, keepdim=True).to(device)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t()).to(device)  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:  # need more study on this
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).to(device)) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats
#plotting functions
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle=angle, **kwargs))
        
        #                     angle, **kwargs))
#        ellipse = Ellipse(position, nsig * width, nsig * height, angle)  # 只传递必要的参数
        # 手动设置支持的属性
#        if 'alpha' in kwargs:
#            ellipse.set_alpha(kwargs['alpha'])
#        if 'facecolor' in kwargs:
#            ellipse.set_facecolor(kwargs['facecolor'])
#        if 'zorder' in kwargs:
#            ellipse.set_zorder(kwargs['zorder'])
#        ax.add_patch(ellipse)
        
def plot_gmm(gm, latents, raw_x, raw_y, label=True, ax=None):  # 添加 raw_x 和 raw_y 参数
    X = latents
    fig, axs = plt.subplots(1,1,figsize=(2,2),dpi=200)
    ax = axs or plt.gca()
    labels = gm.fit(X).predict(X)
    if label:
        low_cu = raw_x[:,5] < 0.05  # 使用传入的 raw_x 参数
        low_cu_latent = latents[low_cu]
        low_cu_color = raw_y[:][low_cu]  # 使用传入的 raw_y 参数

        high_cu = raw_x[:,5] >= 0.05
        high_cu_latent = latents[high_cu]
        high_cu_color = raw_y[:][high_cu]

        scatter1 = axs.scatter(low_cu_latent[:,0], low_cu_latent[:,1], c=low_cu_color, alpha=.65, s=8, linewidths=0, cmap='viridis')
        scatter2 = axs.scatter(high_cu_latent[:,0], high_cu_latent[:,1], c=high_cu_color, alpha=.65, s=14, linewidths=0, cmap='Reds', marker='^')
        #scatter3 = axs.scatter(latents[698:,0], latents[698:,1], alpha=1., s=10, linewidths=.75, edgecolors='k', facecolors='none')
    else:
        ax.scatter(X[:, 0], X[:, 1], s=5, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gm.weights_.max()
    for pos, covar, w in zip(gm.means_, gm.covariances_, gm.weights_):
        draw_ellipse(pos, covar, alpha= 0.75 * w * w_factor, facecolor='slategrey', zorder=-10)
