import torch
from torch import nn
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Subject(object):
    """
    Instance holds information about a subject.
    """

    def __init__(self, t1, t2, label):
        """
        Create a subject

        Args:
            t1: ndarray/None
                array of T1w cerebellar image (after cropping)
            t2: ndarray/None
                array of T2w cerebellar image (after cropping)
            label: ndarray/None
                array of label cerebellar image (after cropping)
        """

        self.t1 = t1
        self.t2 = t2
        self.label = label

    def get_data(self):
        """
        access data of subject. (0 paddings for None)

        Returns:
            t1_data: (Tensor)
                tensor of T1w cerebellar image (after cropping)
            t2_data: (Tensor)
                tensor of T2w cerebellar image (after cropping)
            label_data: (Tensor)
                tensor of label cerebellar image (after cropping)

        """
        if self.t1 is None:
            t1_data = torch.zeros((128, 128, 128), dtype=torch.float)
        else:
            t1_data = torch.tensor(self.t1, dtype=torch.float)
            t1_data = (t1_data - t1_data.mean()) / t1_data.std()
        if self.t2 is None:
            t2_data = torch.zeros((128, 128, 128), dtype=torch.float)
        else:
            t2_data = torch.tensor(self.t2, dtype=torch.float)
            t2_data = (t2_data - t2_data.mean()) / t2_data.std()

        if self.label is None:
            label_data = torch.zeros((128, 128, 128), dtype=torch.float)
        else:
            label_data = torch.tensor(self.label.numpy(), dtype=torch.float)
        return t1_data, t2_data, label_data

class _ConvNet(nn.Module):
    """
    convolution block in Unet
    """

    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        """

        Args:
            in_channels: (int)
                the number of input channels
            out_channels: (int)
                the number of output channels
            dropout_rate: (float)
                reserved
        """
        super(_ConvNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(),
            nn.Dropout3d(dropout_rate)
        )

    def forward(self, x):
        """
        calculation

        Args:
            x: (Tensor)
                input

        Returns:
            y: (Tensor)
                output

        """
        return self.layer(x)


class _DownSample(nn.Module):
    """
    downsample block in unet
    """

    def __init__(self, channels):
        """

        Args:
            channels: (int)
                the number of input channels
        """
        super(_DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(channels, affine=True, track_running_stats=True),
            nn.LeakyReLU()
        )

    def forward(self, x):
        """
        calculation

        Args:
            x: (Tensor)
                input

        Returns:
            y: (Tensor)
                output

        """
        return self.layer(x)


class _UpSample(nn.Module):
    """
    upsample block in unet
    """

    def __init__(self, channels):
        """

        Args:
            channels: (int)
                the number of input channels
        """
        super(_UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2, padding=0),
            nn.InstanceNorm3d(channels, affine=True, track_running_stats=True),
            nn.LeakyReLU()
        )
        self.layer = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels, affine=True, track_running_stats=True),
            nn.LeakyReLU()
        )

    def forward(self, x, feature_map):
        """
        calculation

        Args:
            x: (Tensor)
                input
            feature_map: (Tensor)
                The feature map form the corresponding skip connection

        Returns:
            y: (Tensor)
                output

        """
        up = self.up(x)
        out = self.layer(torch.cat((up, feature_map), dim=1))
        return out


class UNet(nn.Module):
    """
    Unet model architecture.
    """

    def __init__(self, init_features=16, dropout_rate=0.0):
        """

        Args:
            init_features: int
                Number of filters in the first convolutional block.
            dropout_rate: float
                Reserved for dropout (0.0 by default which has no effect)
        """
        super(UNet, self).__init__()

        self.enc1 = _ConvNet(in_channels=2, out_channels=init_features, dropout_rate=dropout_rate)
        self.down1 = _DownSample(init_features)
        self.enc2 = _ConvNet(in_channels=init_features, out_channels=init_features * 2, dropout_rate=dropout_rate)
        self.down2 = _DownSample(init_features * 2)
        self.enc3 = _ConvNet(in_channels=init_features * 2, out_channels=init_features * 4, dropout_rate=dropout_rate)
        self.down3 = _DownSample(init_features * 4)
        self.enc4 = _ConvNet(in_channels=init_features * 4, out_channels=init_features * 8, dropout_rate=dropout_rate)
        self.down4 = _DownSample(init_features * 8)
        self.bottleneck = _ConvNet(in_channels=init_features * 8, out_channels=init_features * 8)
        self.up4 = _UpSample(init_features * 8)
        self.dec4 = _ConvNet(in_channels=init_features * 8, out_channels=init_features * 4)
        self.up3 = _UpSample(init_features * 4)
        self.dec3 = _ConvNet(in_channels=init_features * 4, out_channels=init_features * 2)
        self.up2 = _UpSample(init_features * 2)
        self.dec2 = _ConvNet(in_channels=init_features * 2, out_channels=init_features)
        self.up1 = _UpSample(init_features)
        self.dec1 = _ConvNet(in_channels=init_features, out_channels=2)
        self.out = nn.Sequential(
            nn.Softmax(dim=1)
        )

    def forward(self, t1, t2, age=25):
        """
        Unet calculation

        Args:
            t1: (Tensor)
                the cropped T1w image
            t2: (Tensor)
                the cropped T2w image
            age: (float)
                reserved

        Returns:
            pm: (Tensor)
                the cerebellar probability map

        """

        result_ini = torch.cat((t1, t2), dim=1)

        result_1 = self.enc1(result_ini)
        result_2 = self.enc2(self.down1(result_1))
        result_3 = self.enc3(self.down2(result_2))
        result_4 = self.enc4(self.down3(result_3))
        result_5 = self.bottleneck(self.down4(result_4))
        out_1 = self.dec4(self.up4(result_5, result_4))
        out_2 = self.dec3(self.up3(out_1, result_3))
        out_3 = self.dec2(self.up2(out_2, result_2))
        out_4 = self.dec1(self.up1(out_3, result_1))

        return self.out(out_4)

def _load_model(model, params_file):
    """
    load model with pretrained weights

    Args:
        model: (Unet)
            Unet model
        params_file: (string)
            path to the pretrained weights

    Returns:
        net: (Unet)
            the pretrained model
    """
    net = model.to(device)
    net.eval()
    if os.path.exists(params_file):
        net.load_state_dict(torch.load(params_file, weights_only=True, map_location=device))
    else:
        print('fail to load weights')
        exit(0)
    return net


def predict(model, params_file, t1=None, t2=None):
    """
    Run a prediction on a single subject using a trained UNet model

    Args:
        model: (Unet)
            Unet model
        params_file: (string)
            filename of the pretrained weights
        t1: (Tensor)
            tensor of T1w cerebellar image (after cropping)
        t2: (Tensor)
            tensor of T2w cerebellar image (after cropping)

    Returns:
        mask: (ndarray)
            the 3D numpy array of predicted mask (template space)

    """
    net = _load_model(model, params_file)
    if t1 is None:
        t1 = torch.zeros((128, 128, 128), dtype=torch.float)
    if t2 is None:
        t2 = torch.zeros((128, 128, 128), dtype=torch.float)
    t1, t2 = t1.unsqueeze(0).unsqueeze(0), t2.unsqueeze(0).unsqueeze(0)
    t1, t2 = t1.to(device), t2.to(device)
    mask = net(t1, t2)
    mask = mask.cpu().detach().numpy()
    return mask[0][0]