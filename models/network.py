import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3+1, 64, kernel_size=4, stride=2, padding=1),  # 3 for RGB + 1 for mask
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, mask):
        inp = torch.cat([x, mask], dim=1)
        return self.model(inp)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # Using layers up to relu4_1
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(2)])   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(2,7)]) # relu2_2
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(7,12)])# relu3_2
        self.slice4 = nn.Sequential(*[vgg[x] for x in range(12,21)])# relu4_2
        self.resize = resize

        # Normalization for VGG
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        # Convert from [-1,1] to [0,1]
        pred = (pred + 1)/2
        target = (target + 1)/2

        # Normalize to VGG input
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        if self.resize:
            pred = F.interpolate(pred, size=(224,224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224,224), mode='bilinear', align_corners=False)

        pred_feats1 = self.slice1(pred)
        pred_feats2 = self.slice2(pred_feats1)
        pred_feats3 = self.slice3(pred_feats2)
        pred_feats4 = self.slice4(pred_feats3)

        target_feats1 = self.slice1(target)
        target_feats2 = self.slice2(target_feats1)
        target_feats3 = self.slice3(target_feats2)
        target_feats4 = self.slice4(target_feats3)

        loss = (F.l1_loss(pred_feats1, target_feats1) +
                F.l1_loss(pred_feats2, target_feats2) +
                F.l1_loss(pred_feats3, target_feats3) +
                F.l1_loss(pred_feats4, target_feats4))
        return loss
