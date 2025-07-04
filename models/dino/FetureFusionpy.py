import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttentionModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(TemporalAttentionModule, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        #b, t, c, h, w = x.size()
        # y = self.avg_pool(x).view(b, t, c)  # Pool and flatten the spatial dimensions
        # y = y.mean(dim=1)  # Average over the temporal dimension to get a single vector per batch
        # y = self.fc(y).view(b, 1, c, 1, 1)
        # return x * y.expand_as(x)

        b, t, c, h, w = x.size()
        y = self.avg_pool(x.view(b * t, c, h, w)).view(b * t, c)  # Flatten and applypool
        #y = self.fc(y).view(b, t, 1, 1, 1)
        y = self.fc(y).view(b, t, c, 1, 1)
        return x * y.expand_as(x)+x


class FeatureFusion(nn.Module):
    def __init__(self, channels):
        super(FeatureFusion, self).__init__()
        self.temporal_attention = TemporalAttentionModule(channels)

    def forward(self, x):
        # x: [b, t, c, h, w]
        # Apply attention across temporal dimension
        x = self.temporal_attention(x)

        # Aggregate across the temporal dimension
        #x = x.sum(dim=1)  # Sum or you might also use mean
        x = x.mean(dim=1)  # Sum or you might also use mean
        return x
