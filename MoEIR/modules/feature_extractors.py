class ResNet(Module):
    def __init__(self):
        from  torchvision.models import resnet152
        self.resnet = resnet152(pretrained=True)
        
        # TODO: Adjust last layers to output 64 dimensions.

        for param in self.resnet.features.parameters():
            param.requires_grad = False

    def forward(self,x):
        # 64 x H x W
        return self.resnet(x)
