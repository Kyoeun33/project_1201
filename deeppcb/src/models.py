"""Four-channel patch CNN and shared-backbone, two-input Faster R-CNN."""

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


class FastPatchCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers = []
        in_channels = 4
        for out_channels in (16, 32, 64, 128):
            layers.extend((nn.Conv2d(in_channels, out_channels, 3, padding=1),
                           nn.GroupNorm(4, out_channels), nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)))
            in_channels = out_channels
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(128, 7)  # 0 background, 1..6 official IDs

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(patches).flatten(1))


class SharedSiameseFPN(nn.Module):
    """Same ResNet18 weights encode test and template independently at four scales."""

    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()
        encoder = resnet18(weights=None)  # never downloads pretrained parameters
        encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4)
        channels = (64, 128, 256, 512)
        self.project = nn.ModuleList(nn.Conv2d(n * 3, out_channels, 1) for n in channels)
        self.fpn = FeaturePyramidNetwork([out_channels] * 4, out_channels,
                                         extra_blocks=LastLevelMaxPool())
        self.out_channels = out_channels

    def _encode(self, image: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(image)
        x1 = self.layer1(x)  # stride 4: 160 x 160 on a 640 image
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]

    def forward(self, pair: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        test = self._encode(pair[:, 0:1])
        template = self._encode(pair[:, 1:2])
        fused = OrderedDict()
        for i, (a, b, projection) in enumerate(zip(test, template, self.project)):
            fused[str(i)] = projection(torch.cat((a, b, torch.abs(a - b)), dim=1))
        return self.fpn(fused)


def make_accurate_model(config: dict) -> FasterRCNN:
    model_cfg = config["model"]
    backbone = SharedSiameseFPN(int(model_cfg["fpn_channels"]))
    anchors = tuple(tuple(int(v) for v in level) for level in model_cfg["anchors"])
    if len(anchors) != 5:
        raise ValueError("4개 FPN 레벨 + pool 레벨의 앵커 크기 5개 묶음 필요")
    generator = AnchorGenerator(sizes=anchors, aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    roi = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
    return FasterRCNN(
        backbone, num_classes=7, min_size=640, max_size=640,
        image_mean=[0.0, 0.0], image_std=[1.0, 1.0],
        rpn_anchor_generator=generator, box_roi_pool=roi,
        box_score_thresh=0.0, box_nms_thresh=float(model_cfg["nms_iou"]),
        box_detections_per_img=300,
        rpn_pre_nms_top_n_train=1000, rpn_post_nms_top_n_train=500,
        rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_test=500,
    )
