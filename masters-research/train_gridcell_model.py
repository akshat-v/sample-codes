
import os
import csv
import torch
import argparse
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from gridcell_pt_dataset import GridCellPTDataset
from gridcell_learnable_decoder import GridCellNetworkPlus
from gridcell_population_decoder import GridCellNetworkPlus2
from gridcell_losses import compute_total_loss_learnable, compute_total_loss_population
from debug_logger import DebugLogger
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with preprocessed .pt train/val subfolders')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model', type=str, choices=['learnable', 'population'], default='learnable')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--load_model_path', type=str, default=None)
    return parser.parse_args()

def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    debug_log = DebugLogger(log_path=os.path.join(args.log_dir, 'training_debug.log'), enable_console=True)

    train_dataset = GridCellPTDataset(os.path.join(args.data_dir, 'train_pt'))
    val_dataset   = GridCellPTDataset(os.path.join(args.data_dir, 'val_pt'))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=12, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    model = GridCellNetworkPlus() if args.model == 'learnable' else GridCellNetworkPlus2()
    model.to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    start_epoch = 0
    if args.load_model_path and os.path.exists(args.load_model_path):
        print("Loading model from:", args.load_model_path)
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))

    os.makedirs(args.log_dir, exist_ok=True)
    loss_log_path = os.path.join(args.log_dir, 'loss_log.csv')
    if not os.path.exists(loss_log_path):
        with open(loss_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'val_loss',
                'train_pose_loss', 'train_ce_place', 'train_ce_head',
                'val_pose_loss', 'val_ce_place', 'val_ce_head'
            ])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = total_pose_loss = total_ce_place = total_ce_head = 0.0
        running_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            rgb = batch['rgb'].to(device, non_blocking=True)
            velocity = batch['velocity'].to(device, non_blocking=True)
            position = batch['position'].to(device, non_blocking=True)
            angle = batch['angle_sin_cos'].to(device, non_blocking=True)
            place_targets = batch['place_targets'].to(device, non_blocking=True)
            head_targets = batch['head_targets'].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                outputs = model(rgb, velocity, place_targets, head_targets)
                place_probs, head_probs, poses, _,_ = outputs

                if args.model == 'learnable':
                    loss, loss_ce_place, loss_ce_head, loss_pose = compute_total_loss_learnable(
                        place_probs, head_probs, poses,
                        place_targets, head_targets,
                        position, angle
                    )
                else:
                    loss, loss_ce_place, loss_ce_head, loss_pose = compute_total_loss_population(
                        place_probs, head_probs, poses,
                        place_targets, head_targets,
                        position, angle,
                        batch['placecenters'].to(device),
                        batch['headcenters'].to(device),
                        poses
                    )

                #debug logs
                pred_theta = poses[:, :, 2]
                debug_log.log_tensor_stats("Pred θ (radians)", pred_theta)
                debug_log.log_tensor_stats("GT angle vec [sin, cos]", angle)
                debug_log.log_tensor_stats("GT Position", position)
                debug_log.log_tensor_stats("Head Predictions (probs)", head_probs)
                debug_log.log_tensor_stats("Head Centers", batch['headcenters'].to(device))
                debug_log.log_tensor_stats("Head Targets", head_targets)
                if 'headcenters' in batch:
                    debug_log.log_tensor_stats("Head Centers", batch['headcenters'].to(device))

            scaler.scale(loss).backward()
            debug_log.log_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_pose_loss += loss_pose.item()
            total_ce_place += loss_ce_place.item()
            total_ce_head += loss_ce_head.item()
            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                tqdm.write(f"[Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)}] "
                           f"Running Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        avg_train_loss = total_loss / len(train_loader)
        avg_pose_loss = total_pose_loss / len(train_loader)
        avg_ce_place = total_ce_place / len(train_loader)
        avg_ce_head = total_ce_head / len(train_loader)
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_total_loss = val_pose_loss = val_ce_place = val_ce_head = 0.0
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device, non_blocking=True)
                velocity = batch['velocity'].to(device, non_blocking=True)
                position = batch['position'].to(device, non_blocking=True)
                angle = batch['angle_sin_cos'].to(device, non_blocking=True)
                place_targets = batch['place_targets'].to(device, non_blocking=True)
                head_targets = batch['head_targets'].to(device, non_blocking=True)

                with autocast():
                    outputs = model(rgb, velocity, place_targets, head_targets)
                    place_probs, head_probs, poses, _ ,_= outputs

                    if args.model == 'learnable':
                        loss, loss_ce_place, loss_ce_head, loss_pose = compute_total_loss_learnable(
                            place_probs, head_probs, poses,
                            place_targets, head_targets,
                            position, angle
                        )
                    else:
                        loss, loss_ce_place, loss_ce_head, loss_pose = compute_total_loss_population(
                            place_probs, head_probs, poses,
                            place_targets, head_targets,
                            position, angle,
                            batch['placecenters'].to(device),
                            batch['headcenters'].to(device),
                            poses
                        )

                val_total_loss += loss.item()
                val_pose_loss += loss_pose.item()
                val_ce_place += loss_ce_place.item()
                val_ce_head += loss_ce_head.item()

        avg_val_loss = val_total_loss / len(val_loader)
        avg_val_pose = val_pose_loss / len(val_loader)
        avg_val_ce_place = val_ce_place / len(val_loader)
        avg_val_ce_head = val_ce_head / len(val_loader)

        print(f"Validation Loss: {avg_val_loss:.4f}")

        #csv, remove for runs when testing
        with open(loss_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                avg_train_loss, avg_val_loss,
                avg_pose_loss, avg_ce_place, avg_ce_head,
                avg_val_pose, avg_val_ce_place, avg_val_ce_head
            ])

        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.log_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")

if __name__ == '__main__':
    train()
