import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionEncoder(nn.Module):
    def __init__(self):
        super(VisionEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 512),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class PoseDecoder(nn.Module):
    def __init__(self, place_dim=256, head_dim=36):
        super(PoseDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(place_dim + head_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, place_dist, head_dist):
        x = torch.cat([place_dist, head_dist], dim=-1)
        return self.net(x)

class GridCellNetworkPlus(nn.Module):
    def __init__(self, n_place=256, n_head=36, dropout_prob=0.5):
        super(GridCellNetworkPlus, self).__init__()
        self.vision_encoder = VisionEncoder()
        self.lstm = nn.LSTM(input_size=514, hidden_size=128, batch_first=True)
        self.hidden_to_grid = nn.Sequential(
            nn.Linear(128, 512),
            nn.Dropout(p=dropout_prob)
        )
        self.grid_to_place = nn.Linear(512, n_place)
        self.grid_to_head = nn.Linear(512, n_head)
        self.pose_decoder = PoseDecoder(n_place, n_head)

        self.h_init = nn.Linear(n_place + n_head, 128)
        self.c_init = nn.Linear(n_place + n_head, 128)

    def forward(self, image_seq, velocity, place_targets, head_targets, hidden_state=None):
        B, S, C, H, W = image_seq.shape
        flat_input = image_seq.view(B * S, C, H, W)
        flat_output = self.vision_encoder(flat_input)
        vis_feats = flat_output.view(B, S, -1)

        combined = torch.cat([vis_feats, velocity], dim=-1)

        if hidden_state is None:
            place_0 = place_targets[:, 0]
            head_0 = head_targets[:, 0]
            combined_0 = torch.cat([place_0, head_0], dim=-1)
            h0 = self.h_init(combined_0).unsqueeze(0)
            c0 = self.c_init(combined_0).unsqueeze(0)
            hidden_state = (h0, c0)

        lstm_out, hidden = self.lstm(combined, hidden_state)
        grid_code = self.hidden_to_grid(lstm_out)

        place_dist = F.softmax(self.grid_to_place(grid_code), dim=-1)
        head_dist = F.softmax(self.grid_to_head(grid_code), dim=-1)

        poses = torch.stack([
            self.pose_decoder(place_dist[:, t], head_dist[:, t]) for t in range(S)
        ], dim=1)

        #return lstm_out for gridness analysis
        return place_dist, head_dist, poses, hidden, grid_code
