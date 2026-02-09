import os
import torch
from torch.utils.data import Dataset
from glob import glob
from target_utils import generate_place_cell_centers, generate_head_cell_centers

class GridCellPTDataset(Dataset):
    def __init__(self, pt_dir, place_cells=256, head_cells=36):
        self.pt_files = sorted(glob(os.path.join(pt_dir, '*.pt')))
        if not self.pt_files:
            raise ValueError(f"No .pt files found in {pt_dir}")

        # Generate shared place/head centers
        self.place_cell_centers = generate_place_cell_centers(place_cells)
        self.head_cell_centers = generate_head_cell_centers(head_cells)

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        data = torch.load(self.pt_files[idx])

        required_keys = ['rgb', 'velocity', 'position', 'angle_sin_cos', 'place_targets', 'head_targets']
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing key '{key}' in file: {self.pt_files[idx]}")

        #Batch dictionary 
        data['placecenters'] = self.place_cell_centers
        data['headcenters'] = self.head_cell_centers

        return {
            'rgb': data['rgb'],
            'velocity': data['velocity'],
            'position': data['position'],
            'angle_sin_cos': data['angle_sin_cos'],
            'place_targets': data['place_targets'],
            'head_targets': data['head_targets'],
            'placecenters': data['placecenters'],
            'headcenters': data['headcenters'],
        }
