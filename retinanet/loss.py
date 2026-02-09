
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection._utils import Matcher, BoxCoder
from torchvision.ops import box_iou

def collate_fn(batch):
  
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

class RetinaNetLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        #use pytorch inbuilt matcher 
        self.matcher = Matcher(
            high_threshold=0.5, 
            low_threshold=0.4, 
            allow_low_quality_matches=True
        )
        #pytorch boxcoder
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.reg_loss_fn = nn.SmoothL1Loss(reduction='sum')

    def focal_loss(self, logits, targets):
        alpha = 0.25
        gamma = 2.0
        
        num_classes = logits.shape[1]
        t = F.one_hot(targets, num_classes + 1).float()
        t = t[:, 1:] 

        p = torch.sigmoid(logits)
        pt = p * t + (1 - p) * (1 - t)
        
        alpha_factor = alpha * t + (1 - alpha) * (1 - t)
        modulating_factor = (1 - pt).pow(gamma)
        
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction='none')
        loss = alpha_factor * modulating_factor * bce
        
        return loss.sum()

    def forward(self, cls_preds, bbox_preds, anchors, targets):
        classification_losses = []
        regression_losses = []
        
        for i in range(len(targets)):
            target_boxes = targets[i]["boxes"]
            target_labels = targets[i]["labels"]
            img_anchors = anchors[i]
            
            
            if target_boxes.numel() == 0:
                matched_idxs = torch.full((len(img_anchors),), -1, dtype=torch.long, device=img_anchors.device)
            else:
                iou_matrix = box_iou(target_boxes, img_anchors)
                matched_idxs = self.matcher(iou_matrix)

            
            cls_targets_per_img = torch.zeros(len(img_anchors), dtype=torch.long, device=img_anchors.device)
            
            pos_mask = matched_idxs >= 0
            ignore_mask = matched_idxs == -2
            
            if pos_mask.any():
                matched_gt_indices = matched_idxs[pos_mask]
                cls_targets_per_img[pos_mask] = target_labels[matched_gt_indices]
            
            
            img_cls_preds = cls_preds[i]  
            
          
            valid_mask = ~ignore_mask 
            
            if valid_mask.any():
                
                loss_cls = self.focal_loss(
                    img_cls_preds[valid_mask], 
                    cls_targets_per_img[valid_mask]
                )
                classification_losses.append(loss_cls)

            if pos_mask.any():
                pos_bbox_preds = bbox_preds[i][pos_mask]
                matched_gt_boxes = target_boxes[matched_gt_indices]
                pos_anchors = img_anchors[pos_mask]
                
                target_deltas = self.box_coder.encode(
                    reference_boxes=[matched_gt_boxes], 
                    proposals=[pos_anchors]
                )
                
                if isinstance(target_deltas, tuple):
                    target_deltas = torch.cat(target_deltas, dim=0)

                loss_bbox = self.reg_loss_fn(pos_bbox_preds, target_deltas)
                regression_losses.append(loss_bbox)
            else:
                regression_losses.append(torch.tensor(0.0, device=bbox_preds.device))

     
        num_pos_anchors = 0
        for i in range(len(targets)):
            if len(targets[i]["boxes"]) > 0:
                iou = box_iou(targets[i]["boxes"], anchors[i])
                matches = self.matcher(iou)
                num_pos_anchors += (matches >= 0).sum()
        
        num_pos_anchors = max(1.0, float(num_pos_anchors))
        
        return torch.stack(classification_losses).sum() / num_pos_anchors, \
               torch.stack(regression_losses).sum() / num_pos_anchors