import torch
import torch.nn as nn
import torchvision


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG16(SimpleCNN):
    def __init__(self):
        super().__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.features = torch.nn.Sequential(*list(vgg16.children())[:-1])
        for param in self.features.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(224 * 224 // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1),
        )


############################################ HELPERS FOR DENSEBLOCKNETWORK ############################################
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=growthRate, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer):
        super().__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = nn.Conv2d(in_channels=in_channels_, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        return out


class Downsample(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.downsample(x)
        return x


########################################################################################################################


class DenseBlockNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(
            Downsample(3, 16),
            Downsample(16, 16),
        )
        self.dense_blocks = nn.Sequential(
            DenseBlock(16, 16, 3),
            DenseBlock(16, 16, 3),
            DenseBlock(16, 16, 3),
            DenseBlock(16, 16, 3),
            DenseBlock(16, 16, 3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 56 * 56, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.dense_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
