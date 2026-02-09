
#import numpy as np
#import torch
#
#def generate_place_cell_centers(n_place_cells, env_size=(1.0, 1.0),env_centers=(0.0,0.0)):#acrually the env origin):
    #side_len = int(np.sqrt(n_place_cells))
    #assert side_len**2 == n_place_cells, "Place cells must form a square grid (e.g., 16x16)"
    #x = np.linspace(env_centers[0], env_centers[0]+env_size[0], side_len)
    #y = np.linspace(env_centers[1], env_centers[1]+env_size[1], side_len)
    #grid_x, grid_y = np.meshgrid(x, y)
    #centers = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    #return torch.tensor(centers, dtype=torch.float32)
#
#def generate_head_cell_centers(n_head_cells):
    #return torch.linspace(-np.pi, np.pi, n_head_cells)
#
#
#
#def make_place_cell_soft_targets (xy, cell_centers, sigma=0.1):
    #if xy.ndim == 1:
        #xy = xy.unsqueeze(0)
    #xy = xy.unsqueeze(1)
    #centers = cell_centers.unsqueeze(0)
    #diff_squared = (xy - centers) ** 2
    #exponent = -torch.sum(diff_squared, dim=-1) / (2 * sigma ** 2)
    #numerator = torch.exp(exponent)
#
    #center_diff_squared = (centers - centers.transpose(0, 1)) ** 2
    #center_exponent = -torch.sum(center_diff_squared, dim=-1) / (2 * sigma ** 2)
    #denominator = torch.sum(torch.exp(center_exponent), dim=1)
    #denominator = denominator.unsqueeze(0).repeat(xy.shape[0], 1)
#
    #distribution = numerator / (denominator + 1e-8)
    #return torch.softmax(distribution,dim=-1) #used softmax
#
#
#
#
#"""def make_place_cell_soft_targets(true_xy, centers, sigma=0.1):
    #dists = torch.cdist(true_xy, centers)
    #logits = -0.5 * (dists / sigma) ** 2
    #return torch.softmax(logits, dim=-1)
#"""
#
#
#def make_head_cell_soft_targets(theta, cell_angles, kappa=20.0):
    #if theta.dim() == 1:
        #theta = theta.unsqueeze(1)
    #centers = cell_angles.view(1, -1)
    #numerator = torch.exp(kappa * torch.cos(theta - centers))
    #denominator = torch.sum(torch.exp(kappa * torch.cos(centers - centers.transpose(0, 1))), dim=-1)
    #denominator = denominator.unsqueeze(0).repeat(theta.shape[0], 1)
    #distribution = numerator / (denominator + 1e-8)
    #return torch.softmax(distribution,dim=-1)
#
#"""
#
#def make_head_cell_soft_targets(true_theta, mu_angles, kappa=20.0):
    #B = true_theta.shape[0]
    #theta = true_theta.unsqueeze(1)
    #mu = mu_angles.unsqueeze(0)
    #logits = kappa * torch.cos(theta - mu)
    #return torch.softmax(logits, dim=-1)
#"""

import torch
import math

def generate_place_cell_centers(n_cells):
    """
    Generate place cell centers on a uniform grid in a 2D square [0, 2] x [0, 2].
    
    """
    side = int(n_cells ** 0.5)
    assert side ** 2 == n_cells, "n_cells must be a perfect square"
    
    x = torch.linspace(0, 2, side)
    y = torch.linspace(0, 2, side)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')  # (side, side)
    centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (n_cells, 2)
    return centers

def generate_head_cell_centers(n_cells):
    return torch.linspace(0, 2 * math.pi, steps=n_cells + 1)[:-1]



def compute_place_cell_targets(position, centers, sigma=0.1):
    """
    Create soft target distribution for place cells using Gaussian tuning.
    """
    diff = position.unsqueeze(-2) - centers  # (..., n_cells, 2)
    dist_squared = (diff ** 2).sum(-1)  # (..., n_cells)
    logits = -dist_squared / (2 * sigma ** 2)
    probs = torch.softmax(logits, dim=-1)
    return probs


def compute_head_cell_targets(angle, centers, kappa=10.0):
    """
    Create soft target distribution for head direction cells using von Mises tuning.
    """
    angle = angle.unsqueeze(-1)  # (..., 1)
    diff = angle - centers  # (..., n_cells)
    logits = kappa * torch.cos(diff)
    probs = torch.softmax(logits, dim=-1)
    return probs
