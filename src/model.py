import torch
import torch.nn as nn
import torchvision.models as models


class RoadDamageClassifier(nn.Module):
    """
    A modular wrapper class that encapsulates a pre-trained computer vision
    backbone and attaches a custom classification head.
    """

    def __init__(self, backbone, in_features, num_classes=5):
        super(RoadDamageClassifier, self).__init__()

        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)

        return logits


def create_model(
    model_name="resnet18",
    num_classes=5,
    freeze_backbone="partial"
):
    """
    Factory function to instantiate, configure, and prepare
    pre-trained models for transfer learning.
    """

    if model_name == "resnet18":
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif model_name == "mobilenet_v3_small":
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()

    elif model_name == "efficientnet_b0":
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

    else:
        raise ValueError(
            f"Backbone '{model_name}' is not supported."
        )
    # freezingg logic
    if freeze_backbone is True or freeze_backbone == "full":
        for param in backbone.parameters():
            param.requires_grad = False

    elif freeze_backbone == "partial":
        for param in backbone.parameters():
            param.requires_grad = False
        if model_name == "resnet18":
            for param in backbone.layer4.parameters():
                param.requires_grad = True
        elif model_name == "mobilenet_v3_small":
            for param in backbone.features[-2:].parameters():
                param.requires_grad = True
        elif model_name == "efficientnet_b0":
            for param in backbone.features[-1].parameters():
                param.requires_grad = True
    elif freeze_backbone is False or freeze_backbone == "none":
        for param in backbone.parameters():
            param.requires_grad = True

    model = RoadDamageClassifier(
        backbone=backbone,
        in_features=in_features,
        num_classes=num_classes
    )

    return model


if __name__ == "__main__":
    print("---Running Component Architecture Validation---")
    model = create_model(
        model_name="resnet18",
        num_classes=5,
        freeze_backbone="partial"
    )
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")