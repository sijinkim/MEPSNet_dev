import torch
from torchvision.models import resnet34


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = resnet34(pretrained=True)

        # Drop the last fc layers. B x 512 x H x W
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-2])

        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)

    def __repr__(self):
        return f"{self.__module__.split('.')[-1].upper()} " \
            f"<{self.__class__.__name__}>"

