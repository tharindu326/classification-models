import torch.nn as nn
import torchvision.models as models

class ResNetModel(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=10, pretrained=True):
        super(ResNetModel, self).__init__()
        # Load specified ResNet model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            self.model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        # Freeze all layers for transfer learning
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the classifier head for transfer learning
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


