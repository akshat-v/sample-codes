
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection._utils import Matcher, BoxCoder
from torchvision.ops import box_iou
#Code for retina-net style object detector 

class ObjectDetectionModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.num_classes = params["num_classes"]
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.num_anchors = len(anchor_sizes[0]) * len(aspect_ratios[0]) 
        self.fpn_channels = params.get("fpn_channels", 256)

        self.backbone = self.prepare_backbone()
        self.neck = self.prepare_neck()
        self.cls_head, self.bbox_head = self.prepare_heads()
        
        #anchor generator is incorrect. The anchor sizes and number is incorrectly
        #initialized. As a result the anchors are suboptimal. Fix needs rewrite of
        #object detection model init and loss function/ training script
        self.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        ) 

    def forward(self, x):
       
        c3, c4, c5 = self.backbone(x)
        pyramid_features = self.neck((c3, c4, c5))

        
        img_sizes = [(x.shape[-2], x.shape[-1]) for _ in range(x.shape[0])]
        image_list_obj = ImageList(x, img_sizes)
        anchors = self.anchor_generator(image_list_obj, pyramid_features)

        
        cls_outputs = []
        bbox_outputs = []

        for feature in pyramid_features:
            cls_out = self.cls_head(feature)
            bbox_out = self.bbox_head(feature)
            #uses custom process_head_output function to fix dimensions for loss
            cls_outputs.append(process_head_output(cls_out, self.num_anchors))
            bbox_outputs.append(process_head_output(bbox_out, self.num_anchors))

        cls_outputs = torch.cat(cls_outputs, dim=1)
        bbox_outputs = torch.cat(bbox_outputs, dim=1)

        return cls_outputs, bbox_outputs, anchors

    def prepare_backbone(self):
        return ResNetBackbone()

    def prepare_neck(self):
        return FPN(in_channels_list=[512, 1024, 2048],
                    out_channels=self.fpn_channels)

    def prepare_heads(self):
        #create classification and bbox heads
        cls_head = ClassificationHead(
            in_channels=self.fpn_channels,
            num_anchors=self.num_anchors, 
            num_classes=self.num_classes
        )
        bbox_head = BBoxHead(
            in_channels=self.fpn_channels,
            num_anchors=self.num_anchors
        )
        return cls_head, bbox_head


class ResNetBackbone(nn.Module):
    #use pretrained resnet weights
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights="IMAGENET1K_V1")

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1  # C2
        self.layer2 = resnet.layer2  # C3
        self.layer3 = resnet.layer3  # C4
        self.layer4 = resnet.layer4  # C5

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        #create Feature pyramid network with 5 levels
        super().__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

        self.p6 = nn.Conv2d(in_channels_list[-1], out_channels, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, pyramid_features):
        c3, c4, c5 = pyramid_features

        p5 = self.lateral_convs[2](c5)
        p4 = self.lateral_convs[1](c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lateral_convs[0](c3) + F.interpolate(p4, scale_factor=2, mode="nearest")

        p3 = self.output_convs[0](p3)
        p4 = self.output_convs[1](p4)
        p5 = self.output_convs[2](p5)

        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))

        return [p3, p4, p5, p6, p7]


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)
        self.output = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        #should initialize priors here something like this.
        """ self.output = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        prior = 0.01
        fill_value = -math.log((1 - prior) / prior)
        self.output.bias.data.fill_(fill_value)"""
    def forward(self, x):
        x = self.conv(x)
        return self.output(x)


class BBoxHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)
        self.output = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return self.output(x)


def process_head_output(head_output, num_anchors):
    #returns reshaped outputs so that it aligns with inbuilt mapper in pytorch for loss calc
    N, _, H, W = head_output.shape
    C = head_output.shape[1] // num_anchors
    head_output = head_output.view(N, num_anchors, C, H, W)
    head_output = head_output.permute(0, 3, 4, 1, 2)
    return head_output.reshape(N, -1, C)
