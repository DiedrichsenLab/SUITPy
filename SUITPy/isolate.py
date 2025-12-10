"""
Cerebellar Isolation using a Unet model
authors: Yao Li, Carlos Hernandez-Castillo, Joern Diedrichsen
"""

import sys
import argparse
import os
import nibabel as nib
import ants
import numpy as np
from tempfile import mkstemp
import nitools
from typing import Tuple, Union
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class _Conv3dN:
    """
    numpy implementation of 3D convolution layer
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int, int]],
            stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
            padding: Union[int, Tuple[int, int, int]] = (0, 0, 0),
            dilation: Union[int, Tuple[int, int, int]] = (1, 1, 1),
            bias: bool = True,
            padding_mode: str = 'zeros',
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3
        self.bias = bias
        self.padding_mode = padding_mode

        self.weight = np.random.randn(out_channels, in_channels, *self.kernel_size)

        if bias:
            self.bias_term = np.zeros(out_channels)
        else:
            self.bias_term = None

    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        if all(p == 0 for p in self.padding):
            return x

        pd, ph, pw = self.padding
        if self.padding_mode == 'zeros':
            return np.pad(x,
                          ((0, 0), (0, 0),
                           (pd, pd), (ph, ph), (pw, pw)),
                          mode='constant')
        elif self.padding_mode == 'reflect':
            return np.pad(x,
                          ((0, 0), (0, 0),
                           (pd, pd), (ph, ph), (pw, pw)),
                          mode='reflect')
        else:
            raise NotImplementedError(f"Padding mode {self.padding_mode} not implemented")

    def _im2col(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        batch_size, in_channels, depth, height, width = x.shape
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding
        dd, dh, dw = self.dilation

        x_padded = self._pad_input(x)

        out_depth = (depth + 2 * pd - dd * (kd - 1) - 1) // sd + 1
        out_height = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        out_width = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1

        cols = np.zeros((batch_size, in_channels, kd, kh, kw, out_depth, out_height, out_width))

        for d in range(kd):
            for h in range(kh):
                for w in range(kw):
                    d_start = d * dd
                    h_start = h * dh
                    w_start = w * dw

                    d_slice = slice(d_start, d_start + out_depth * sd, sd)
                    h_slice = slice(h_start, h_start + out_height * sh, sh)
                    w_slice = slice(w_start, w_start + out_width * sw, sw)

                    cols[:, :, d, h, w, :, :, :] = x_padded[:, :, d_slice, h_slice, w_slice]

        # Reshape for matrix multiplication (batch, out_d, out_h, out_w, in_ch, kd, kh, kw)
        cols = cols.transpose(0, 5, 6, 7, 1, 2, 3, 4)
        cols = cols.reshape(batch_size * out_depth * out_height * out_width, -1)

        return cols, (out_depth, out_height, out_width)

    def load_state(self, weight: np.ndarray, bias_term: np.ndarray):
        self.weight = weight
        self.bias_term = bias_term

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 5:
            raise ValueError(f"Input must have 5 dimensions (N, C, D, H, W), got {x.ndim}")

        batch_size, in_channels, depth, height, width = x.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {in_channels}")

        cols, (out_depth, out_height, out_width) = self._im2col(x)

        # Reshape weights for matrix multiplication (out_ch, in_ch * kd * kh * kw)
        weight_flat = self.weight.reshape(self.out_channels, -1)

        # Perform convolution via matrix multiplication (batch * out_d * out_h * out_w, out_ch)
        output_flat = cols @ weight_flat.T

        output = output_flat.reshape(batch_size, out_depth, out_height, out_width, self.out_channels)
        output = output.transpose(0, 4, 1, 2, 3)  # (batch, out_ch, out_d, out_h, out_w)

        if self.bias:
            output += self.bias_term.reshape(1, -1, 1, 1, 1)

        return output

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class _ConvTranspose3dN:
    """
    numpy implementation of 3D transpose convolution layer
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int, int]],
            stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
            padding: Union[int, Tuple[int, int, int]] = (0, 0, 0),
            output_padding: Union[int, Tuple[int, int, int]] = (0, 0, 0),
            dilation: Union[int, Tuple[int, int, int]] = (1, 1, 1),
            bias: bool = True,
            padding_mode: str = "zeros",
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding,) * 3
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3
        self.bias = bias
        self.padding_mode = padding_mode

        self.weight = np.random.randn(in_channels, out_channels, *self.kernel_size)
        if bias:
            self.bias_term = np.zeros(out_channels)
        else:
            self.bias_term = None


    def load_state(self, weight: np.ndarray, bias_term: np.ndarray):
        self.weight = weight
        self.bias_term = bias_term

    def _dilate_input(self, x: np.ndarray) -> np.ndarray:
        batch_size, in_channels, depth, height, width = x.shape
        sd, sh, sw = self.stride

        dilated_depth = (depth - 1) * sd + 1
        dilated_height = (height - 1) * sh + 1
        dilated_width = (width - 1) * sw + 1

        dilated_x = np.zeros((batch_size, in_channels, dilated_depth, dilated_height, dilated_width))

        # Insert original values at strided positions
        dilated_x[:, :, ::sd, ::sh, ::sw] = x

        return dilated_x

    def _apply_padding(self, x: np.ndarray) -> np.ndarray:
        pd, ph, pw = self.padding
        kd, kh, kw = self.kernel_size

        pad_d_before = kd - 1 - pd
        pad_d_after = kd - 1 - pd + self.output_padding[0]

        pad_h_before = kh - 1 - ph
        pad_h_after = kh - 1 - ph + self.output_padding[1]

        pad_w_before = kw - 1 - pw
        pad_w_after = kw - 1 - pw + self.output_padding[2]

        padding = ((0, 0), (0, 0),
                   (pad_d_before, pad_d_after),
                   (pad_h_before, pad_h_after),
                   (pad_w_before, pad_w_after))
        if self.padding_mode == "zeros":
            return np.pad(x, padding, mode='constant')
        else:
            raise NotImplementedError(f"Padding mode {self.padding_mode} not implemented")

    def _im2col(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        batch_size, in_channels, depth, height, width = x.shape
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding
        opd, oph, opw = self.output_padding
        dd, dh, dw = self.dilation

        dilated_x = self._dilate_input(x)

        padded_x = self._apply_padding(dilated_x)

        # Output dimensions
        out_depth = (depth - 1) * sd - 2 * pd + dd * (kd - 1) + opd + 1
        out_height = (height - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
        out_width = (width - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1

        cols = np.zeros((batch_size, in_channels, kd, kh, kw, out_depth, out_height, out_width))

        for d in range(kd):
            for h in range(kh):
                for w in range(kw):
                    d_start = d * dd
                    h_start = h * dh
                    w_start = w * dw

                    d_slice = slice(d_start, d_start + out_depth)
                    h_slice = slice(h_start, h_start + out_height)
                    w_slice = slice(w_start, w_start + out_width)

                    cols[:, :, d, h, w, :, :, :] = padded_x[:, :, d_slice, h_slice, w_slice]

        # Reshape for matrix multiplication (batch, out_d, out_h, out_w, in_ch, kd, kh, kw)
        cols = cols.transpose(0, 5, 6, 7, 1, 2, 3, 4)
        cols = cols.reshape(batch_size * out_depth * out_height * out_width, -1)

        return cols, (out_depth, out_height, out_width)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 5:
            raise ValueError(f"Input must have 5 dimensions (N, C, D, H, W), got {x.ndim}")

        batch_size, in_channels, depth, height, width = x.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {in_channels}")

        cols, (out_depth, out_height, out_width) = self._im2col(x)

        # Reshape weights for matrix multiplication (out_ch, in_ch * kd * kh * kw)
        weight = self.weight.transpose(1, 0, 2, 3, 4)
        weight_flip = np.flip(weight, (2, 3, 4))
        weight_flat = weight_flip.reshape(self.out_channels, -1)

        # Perform convolution via matrix multiplication (batch * out_d * out_h * out_w, out_ch)
        output_flat = cols @ weight_flat.T

        output = output_flat.reshape(batch_size, out_depth, out_height, out_width, self.out_channels)
        output = output.transpose(0, 4, 1, 2, 3)  # (batch, out_ch, out_d, out_h, out_w)

        if self.bias:
            output += self.bias_term.reshape(1, -1, 1, 1, 1)

        return output

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class _InstanceNorm3dN:
    """
    numpy implementation of 3D instance normalization layer
    """
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = False,
            track_running_stats: bool = False
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = np.ones(num_features)
            self.bias = np.zeros(num_features)
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            raise RuntimeError('track_running_stats currently not supported.')
            # self.running_mean = np.zeros(num_features)
            # self.running_var = np.ones(num_features)
            # self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def _check_input_dim(self, x: np.ndarray) -> None:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (got {x.ndim}D input)")
        if x.shape[1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} channels in input (got {x.shape[1]} channels)")

    def _compute_instance_stats(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and variance per instance per channel across spatial dimensions (D, H, W)"""

        n, c, d, h, w = x.shape
        x_reshaped = x.reshape(n, c, -1)

        mean = np.mean(x_reshaped, axis=2)
        var = np.var(x_reshaped, axis=2)

        return mean, var

    def _normalize(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        """Normalize input using computed statistics"""

        n, c, d, h, w = x.shape

        mean = mean.reshape(n, c, 1, 1, 1)
        var = var.reshape(n, c, 1, 1, 1)

        x_normalized = (x - mean) / np.sqrt(var + self.eps)

        return x_normalized

    def forward(self, x: np.ndarray) -> np.ndarray:

        self._check_input_dim(x)

        if self.track_running_stats:
            n, c, d, h, w = x.shape
            mean = self.running_mean.reshape(1, c, 1, 1, 1)
            var = self.running_var.reshape(1, c, 1, 1, 1)
        else:
            # If not tracking running stats, compute instance stats even during inference
            mean, var = self._compute_instance_stats(x)

        if not self.track_running_stats:
            x_normalized = self._normalize(x, mean, var)
        else:
            x_normalized = (x - mean) / np.sqrt(var + self.eps)

        if self.affine:
            weight = self.weight.reshape(1, -1, 1, 1, 1)
            bias = self.bias.reshape(1, -1, 1, 1, 1)
            x_normalized = x_normalized * weight + bias

        return x_normalized

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


    def load_state_dict(self, state_dict: dict) -> None:

        if self.affine:
            self.weight = state_dict['weight']
            self.bias = state_dict['bias']

        if self.track_running_stats:
            self.running_mean = state_dict['running_mean']
            self.running_var = state_dict['running_var']
            self.num_batches_tracked = state_dict['num_batches_tracked']


class _LeakyReLUN:
    """
    numpy implementation of Leaky ReLUN layer
    """
    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def forward(self, x: np.ndarray) -> np.ndarray:

        return np.maximum(0, x) + self.negative_slope * np.minimum(0, x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class _SoftmaxN:
    """
    numpy implementation of Softmax layer
    """
    def __init__(self, dim: int = -1,):
        self.dim = dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        dim = self.dim

        x_shifted = x - np.max(x, axis=dim, keepdims=True)

        exp_x = np.exp(x_shifted)

        softmax_output = exp_x / np.sum(exp_x, axis=dim, keepdims=True)

        return softmax_output

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class _ConvNetN:
    """
    convolution block in Unet
    """
    def __init__(self, in_channels: int, out_channels: int):
        """

        Args:
            in_channels: (int)
                the number of input channels
            out_channels: (int)
                the number of output channels
        """
        self.conv1 = _Conv3dN(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm1 = _InstanceNorm3dN(num_features=out_channels, affine=True, track_running_stats=False)
        self.activate1= _LeakyReLUN()

        self.conv2 = _Conv3dN(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm2 = _InstanceNorm3dN(num_features=out_channels, affine=True, track_running_stats=False)
        self.activate2 = _LeakyReLUN()


    def load_state(
            self,
            c_weight1: np.ndarray,
            c_bias1: np.ndarray,
            n_state_dict1: dict,
            c_weight2: np.ndarray,
            c_bias2: np.ndarray,
            n_state_dict2: dict,
    ):
        """
        load the previously trained parameters
        Args:
            c_weight1: (ndarray)
                the weight of the first convolution layer
            c_bias1: (ndarray)
                the bias of the first convolution layer
            n_state_dict1: (dict)
                the state dictionary of the first instance normalization layer
            c_weight2: (ndarray)
                the weight of the second convolution layer
            c_bias2: (ndarray)
                the bias of the second convolution layer
            n_state_dict2: (dict)
                the state dictionary of the second instance normalization layer

        Returns:

        """
        self.conv1.load_state(weight=c_weight1, bias_term=c_bias1)
        self.norm1.load_state_dict(state_dict=n_state_dict1)
        self.conv2.load_state(weight=c_weight2, bias_term=c_bias2)
        self.norm2.load_state_dict(state_dict=n_state_dict2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        calculation

        Args:
            x: (ndarray)
                input

        Returns:
            y: (ndarray)
                output

        """
        result1 = self.conv1(x)
        result2 = self.norm1(result1)
        result3 = self.activate1(result2)
        result4 = self.conv2(result3)
        result5 = self.norm2(result4)
        result6 = self.activate2(result5)

        return result6

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class _DownSampleN:
    """
    downsample block in unet
    """
    def __init__(self, channels: int):
        """

        Args:
            channels: (int)
                the number of input channels
        """
        self.conv1 = _Conv3dN(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)
        self.norm1 = _InstanceNorm3dN(num_features=channels, affine=True, track_running_stats=False)
        self.activate1 = _LeakyReLUN()

    def load_state(
            self,
            c_weight: np.ndarray,
            c_bias: np.ndarray,
            n_state_dict: dict
    ):
        """
        load the previously trained parameters
        Args:
            c_weight: (ndarray)
                the weight of the convolution layer
            c_bias: (ndarray)
                the bias of the convolution layer
            n_state_dict: (dict)
                the state dictionary of the instance normalization layer

        """
        self.conv1.load_state(weight=c_weight, bias_term=c_bias)
        self.norm1.load_state_dict(state_dict=n_state_dict)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        calculation

        Args:
            x: (ndarray)
                input

        Returns:
            y: (ndarray)
                output

        """
        result1 = self.conv1(x)
        result2 = self.norm1(result1)
        result3 = self.activate1(result2)

        return result3

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class _UpSampleN:
    """
    upsample block in unet
    """
    def __init__(self, channels: int):
        """

        Args:
            channels: (int)
                the number of input channels
        """
        self.conv1 = _ConvTranspose3dN(in_channels=channels, out_channels=channels, kernel_size=2, stride=2, padding=0)
        self.norm1 = _InstanceNorm3dN(num_features=channels, affine=True, track_running_stats=False)
        self.activate1 = _LeakyReLUN()

        self.conv2 = _Conv3dN(in_channels=channels * 2, out_channels=channels, kernel_size=3, padding=1)
        self.norm2 = _InstanceNorm3dN(num_features=channels, affine=True, track_running_stats=False)
        self.activate2 = _LeakyReLUN()

    def load_state(
            self,
            c_weight1: np.ndarray,
            c_bias1: np.ndarray,
            n_state_dict1: dict,
            c_weight2: np.ndarray,
            c_bias2: np.ndarray,
            n_state_dict2: dict
    ):
        """
        load the previously trained parameters
        Args:
            c_weight1: (ndarray)
                the weight of the transpose convolution layer
            c_bias1: (ndarray)
                the bias of the transpose convolution layer
            n_state_dict1: (dict)
                the state dictionary of the first instance normalization layer
            c_weight2: (ndarray)
                the weight of the convolution layer
            c_bias2: (ndarray)
                the bias of the convolution layer
            n_state_dict2: (dict)
                the state dictionary of the second instance normalization layer
        Returns:

        """
        self.conv1.load_state(weight=c_weight1, bias_term=c_bias1)
        self.norm1.load_state_dict(state_dict=n_state_dict1)
        self.conv2.load_state(weight=c_weight2, bias_term=c_bias2)
        self.norm2.load_state_dict(state_dict=n_state_dict2)

    def forward(self, x: np.ndarray, feature_map: np.ndarray) -> np.ndarray:
        """
        calculation

        Args:
            x: (ndarray)
                input
            feature_map: (ndarray)
                The feature map form the corresponding skip connection

        Returns:
            y: (ndarray)
                output

        """
        result1 = self.conv1(x)
        result2 = self.norm1(result1)
        result3 = self.activate1(result2)
        result3 = np.concat([result3, feature_map], axis=1)
        result4 = self.conv2(result3)
        result5 = self.norm2(result4)
        result6 = self.activate2(result5)

        return result6

    def __call__(self, x: np.ndarray, feature_map: np.ndarray) -> np.ndarray:
        return self.forward(x, feature_map)


class UNetN:
    """
    Unet model architecture.
    """

    def __init__(self, init_features: int = 16):
        """

        Args:
            init_features: int
                Number of filters in the first convolutional block.
        """
        self.enc1 = _ConvNetN(in_channels=2, out_channels=init_features)
        self.down1 = _DownSampleN(channels=init_features)
        self.enc2 = _ConvNetN(in_channels=init_features, out_channels=init_features * 2)
        self.down2 = _DownSampleN(init_features * 2)
        self.enc3 = _ConvNetN(in_channels=init_features * 2, out_channels=init_features * 4)
        self.down3 = _DownSampleN(init_features * 4)
        self.enc4 = _ConvNetN(in_channels=init_features * 4, out_channels=init_features * 8)
        self.down4 = _DownSampleN(init_features * 8)
        self.bottleneck = _ConvNetN(in_channels=init_features * 8, out_channels=init_features * 8)
        self.up4 = _UpSampleN(init_features * 8)
        self.dec4 = _ConvNetN(in_channels=init_features * 8, out_channels=init_features * 4)
        self.up3 = _UpSampleN(init_features * 4)
        self.dec3 = _ConvNetN(in_channels=init_features * 4, out_channels=init_features * 2)
        self.up2 = _UpSampleN(init_features * 2)
        self.dec2 = _ConvNetN(in_channels=init_features * 2, out_channels=init_features)
        self.up1 = _UpSampleN(init_features)
        self.dec1 = _ConvNetN(in_channels=init_features, out_channels=2)
        self.out = _SoftmaxN(dim=1)

    def load_state_dict(self, state_dict: dict):
        """
        Load the state dict of the Unet model.
        Args:
            state_dict: (dict)
                The state dict of the Unet model

        """

        convs = {'enc1': self.enc1, 'enc2': self.enc2, 'enc3': self.enc3, 'enc4': self.enc4, 'bottleneck': self.bottleneck, 'dec1': self.dec1, 'dec2': self.dec2, 'dec3': self.dec3, 'dec4': self.dec4}
        downs = {'down1': self.down1, 'down2': self.down2, 'down3': self.down3, 'down4': self.down4}
        ups = {'up1': self.up1, 'up2': self.up2, 'up3': self.up3, 'up4': self.up4}

        for c in convs:
            c_weight1 = state_dict[f'{c}.layer.0.weight']
            c_bias1 = state_dict[f'{c}.layer.0.bias']
            n1 = {
                'weight': state_dict[f'{c}.layer.1.weight'],
                'bias': state_dict[f'{c}.layer.1.bias'],
                # 'running_mean': state_dict[f'{c}.layer.1.running_mean'],
                # 'running_var': state_dict[f'{c}.layer.1.running_var'],
                # 'num_batches_tracked': state_dict[f'{c}.layer.1.num_batches_tracked']
            }
            c_weight2 = state_dict[f'{c}.layer.3.weight']
            c_bias2 = state_dict[f'{c}.layer.3.bias']
            n2 = {
                'weight': state_dict[f'{c}.layer.4.weight'],
                'bias': state_dict[f'{c}.layer.4.bias'],
                # 'running_mean': state_dict[f'{c}.layer.4.running_mean'],
                # 'running_var': state_dict[f'{c}.layer.4.running_var'],
                # 'num_batches_tracked': state_dict[f'{c}.layer.4.num_batches_tracked']
            }
            convs[c].load_state(c_weight1=c_weight1, c_bias1=c_bias1, n_state_dict1=n1, c_weight2=c_weight2, c_bias2=c_bias2, n_state_dict2=n2)

        for d in downs:
            c_weight = state_dict[f'{d}.layer.0.weight']
            c_bias = state_dict[f'{d}.layer.0.bias']
            n = {
                'weight': state_dict[f'{d}.layer.1.weight'],
                'bias': state_dict[f'{d}.layer.1.bias'],
                # 'running_mean': state_dict[f'{d}.layer.1.running_mean'],
                # 'running_var': state_dict[f'{d}.layer.1.running_var'],
                # 'num_batches_tracked': state_dict[f'{d}.layer.1.num_batches_tracked']
            }
            downs[d].load_state(c_weight=c_weight, c_bias=c_bias, n_state_dict=n)

        for u in ups:
            c_weight1 = state_dict[f'{u}.up.0.weight']
            c_bias1 = state_dict[f'{u}.up.0.bias']
            n1 = {
                'weight': state_dict[f'{u}.up.1.weight'],
                'bias': state_dict[f'{u}.up.1.bias'],
                # 'running_mean': state_dict[f'{u}.up.1.running_mean'],
                # 'running_var': state_dict[f'{u}.up.1.running_var'],
                # 'num_batches_tracked': state_dict[f'{u}.up.1.num_batches_tracked']
            }
            c_weight2 = state_dict[f'{u}.layer.0.weight']
            c_bias2 = state_dict[f'{u}.layer.0.bias']
            n2 = {
                'weight': state_dict[f'{u}.layer.1.weight'],
                'bias': state_dict[f'{u}.layer.1.bias'],
                # 'running_mean': state_dict[f'{u}.layer.1.running_mean'],
                # 'running_var': state_dict[f'{u}.layer.1.running_var'],
                # 'num_batches_tracked': state_dict[f'{u}.layer.1.num_batches_tracked']
            }
            ups[u].load_state(c_weight1=c_weight1, c_bias1=c_bias1, n_state_dict1=n1, c_weight2=c_weight2, c_bias2=c_bias2, n_state_dict2=n2)

    def forward(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """
        Unet calculation

        Args:
            t1: (ndarray)
                the cropped T1w image
            t2: (ndarray)
                the cropped T2w image

        Returns:
            pm: (ndarray)
                the cerebellar probability map

        """
        result_ini = np.concatenate((t1, t2), axis=1)

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

    def __call__(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        return self.forward(t1, t2)

def _load_model(params_file: str):
    """
    load model with pretrained weights

    Args:
        params_file: (string)
            path to the pretrained weights

    Returns:
        net: (Unet)
            the pretrained model
    """
    net = UNetN()
    if os.path.exists(params_file):
        with open(params_file, "rb") as f:
            params = pickle.load(f)
        net.load_state_dict(params)
    else:
        raise RuntimeError('fail to load pre-trained parameters')
    return net


def predict(params_file: str, t1: np.ndarray = None, t2: np.ndarray = None) -> np.ndarray:
    """
    Run a prediction on a single subject using a trained UNet model

    Args:
        params_file: (string)
            filename of the pretrained weights
        t1: (ndarray)
            Numpy array of T1w cerebellar image (after cropping)
        t2: (ndarray)
            Numpy array of T2w cerebellar image (after cropping)

    Returns:
        mask: (ndarray)
            the 3D numpy array of predicted mask (template space)

    """
    net = _load_model(params_file)
    if t1 is None:
        t1 = np.zeros((128, 128, 128))
    else:
        t1 = (t1 - t1.mean()) / t1.std()
    if t2 is None:
        t2 = np.zeros((128, 128, 128))
    else:
        t2 = (t2 - t2.mean()) / t2.std()
    t1, t2 = t1.reshape((1, 1, 128, 128, 128)), t2.reshape((1, 1, 128, 128, 128))
    mask = net(t1, t2)
    return mask[0][0]

def from_nibabel(nib_image: nib.Nifti1Image) -> ants.ANTsImage:
    """
    Converts a given Nifti image into an ANTsPy image
    (https://antspy.readthedocs.io/en/latest/index.html)

    Args:
        img: NiftiImage

    Returns:
        ants_image: ANTsImage
    """
    fd, tmpfile = mkstemp(suffix=".nii.gz")
    nib_image.to_filename(tmpfile)
    new_img = ants.image_read(tmpfile)
    os.close(fd)
    os.remove(tmpfile)
    return new_img


def img_read(path: str) -> ants.ANTsImage:
    """
    basic function to read a nifti image
    Args:
        path: (string)
            image path

    Returns:
        img (ANTs image)
            An ANTs image

    """

    nib_img = nib.load(path)
    # set q form identical to s form to avoid misalignment from ANTs
    nib_img.set_qform(nib_img.get_sform())
    new_img = from_nibabel(nib_img)
    return new_img


class TemplateCerebellarBoundingBox:
    """
        Basic cerebellar bounding box class, which defines the cropped area.
        All other template implementations should be registered to this template.
    """

    def __init__(self, name: str ='MNI152NLin6Asym', bounding_box: np.ndarray = None, cerebellar_center: np.ndarray = None, cropped_size: np.ndarray = None):
        """
        Create a bounding box

        Args:
            name: (string)
                The name of template used. It uses MNI152NLin6Asym by default.
            bounding_box: (ndarray)
                cerebellar bounding box in MNI space (in mm) (reserved for future development)
            cerebellar_center: (ndarray)
                (reserved for future development)
            cropped_size: (ndarray)
                (reserved for future development)
        """
        self.cropped_size = (128, 128, 128)
        if bounding_box is not None:
            self.bounding_box = bounding_box
        else:
            if cerebellar_center is not None and cropped_size is not None:
                # reserved for future development
                pass
            else:
                self.bounding_box = np.array([[64, -114, -88], [-64, 14, 40]])

        self.lowerleft = self.bounding_box[0]
        self.upperright = self.bounding_box[1]

        base_dir = os.path.dirname(__file__)
        self.template = ants.image_read(os.path.join(base_dir, f'templates/tpl-{name}_T1w.nii.gz'))
        self.nib_template = nib.load(os.path.join(base_dir, f'templates/tpl-{name}_T1w.nii.gz'))
        self.brainmask = ants.image_read(os.path.join(base_dir, f'templates/tpl-{name}-brain_mask.nii.gz'))
        self.brain = ants.mask_image(self.template, self.brainmask)
        self.affine = nib.load(os.path.join(base_dir, f'templates/tpl-{name}_T1w.nii.gz')).affine

    def get_crop_indices(self) -> np.ndarray:
        """
        calculate the lower left and upper right indices of the cropped area (in voxels).

        Returns:
            indices: (ndarray)
                a 2 * 3 ndarray consists of two vertices defining the bounding box
        """
        return nitools.coords_to_voxelidxs(self.bounding_box.T, self.nib_template).T

    def get_cropped_affine(self) -> np.ndarray:
        """
        get the cropped area affine

        Returns:
            affine: (ndarray)
                The affine form for the cropped image
        """
        # This function needs to fix. It will fail if the template affine is not diagonal
        affine = np.diag([self.affine[0, 0], self.affine[1, 1], self.affine[2, 2], 1])
        affine[0, 3] = abs(affine[0, 0]) * self.lowerleft[0]
        affine[1, 3] = abs(affine[1, 1]) * self.lowerleft[1]
        affine[2, 3] = abs(affine[2, 2]) * self.lowerleft[2]

        return affine

    def registration(self, img: ants.ANTsImage, type_of_transform: str = 'Similarity') -> ants.ANTsTransform:
        """
        register the image to this template

        Args:
            img: (ANTsImage)
                image to be registered
            type_of_transform: (string)
                transform type (Affine by default, check ANTsPY[https://antspy.readthedocs.io/en/latest/registration.html] for details)

        Returns:
            trans: (ANTsTransform)
                the transformation from the subject space to the template space

        """
        result = ants.registration(fixed=self.template, moving=img, type_of_transform=type_of_transform)
        trans = ants.read_transform(result['fwdtransforms'][0])

        return trans

    def registration_brain(self, img: ants.ANTsImage, type_of_transform: str = 'Similarity') -> ants.ANTsTransform:
        """
        register the image to this template using the brain. The input image should be brain only.

        Args:
            img: (ANTsImage)
                image to be registered
            type_of_transform: (string)
                transform type (Affine by default, check ANTsPY[https://antspy.readthedocs.io/en/latest/registration.html] for details)

        Returns:
            trans: (ANTsTransform)
                the transformation from the subject space to the template space

        """
        result = ants.registration(fixed=self.brain, moving=img, type_of_transform=type_of_transform)
        trans = ants.read_transform(result['fwdtransforms'][0])

        return trans

    def crop(self, img: ants.ANTsImage, trans: ants.ANTsTransform = None) -> Tuple[ants.ANTsImage, ants.ANTsImage]:
        """
        Crop the cerebellar area using the bounding box.

         Args:
             img: (ANTsImage)
                 image to be cropped
             trans: (ANTsTransform)
                 transformation matrix from the image space to the template space (only use it if img is not in the MNI template)

         Returns:
             cropped_img: (ANTsImage)
                cropped image
             img : (ANTsImage)
                the whole transformed image

         """
        start_indices, end_indices = self.get_crop_indices()
        if trans is not None:
            img = ants.apply_ants_transform_to_image(trans, img, self.template)
        return ants.crop_indices(img, tuple(start_indices.astype(int)), tuple(end_indices.astype(int))), img

    def template2subject(self, img: ants.ANTsImage, trans: ants.ANTsTransform, ref: ants.ANTsImage) -> ants.ANTsImage:
        """
        transform the image from template space to the subject space

        Args:
            img: (ANTsImage)
                the image to be transformed
            trans: (ANTs transformation)
                transformation matrix (from subject space to template space)
            ref: (ANTsImage)
                reference image

        Returns:
            img: (ANTsImage)
                the transformed image in subject space

        """
        trans_inv = ants.invert_ants_transform(trans)
        result = ants.apply_ants_transform_to_image(trans_inv, img, ref)

        return result


def subject_preprocess(t1_file: str = None, t2_file: str = None, brain_mask_file: str = None, label_file: str = None,
                       BoundingBox: TemplateCerebellarBoundingBox = TemplateCerebellarBoundingBox(),
                       type_of_transform: str = 'Similarity') -> Tuple[ants.ANTsTransform, ants.ANTsImage, ants.ANTsImage, ants.ANTsImage, ants.ANTsImage, ants.ANTsImage]:
    """
    function to preprocess a single subject.
    1. Transform the image from subject space to the template space
    2. Using a pre-defined bounding box to crop the image in the template space

    Args:
        t1_file: (string)
            file to T1w image
        t2_file: (string)
            file to T2w image
        brain_mask_file: (string)
            file to the brain mask image (can be used to improve affine registration)
        label_file: (string)
            file to label image (Optional, this image will be transformed into the template space using the same transformation.)
        BoundingBox: (TemplateCerebellarBoundingBox)
            the bounding box
        type_of_transform: (string)
            reserved for future use (see ANTspy)

    Returns:
        trans: (ANTsTransform)
            transformation from subject space to template space
        t1_crop: (ANTsImage)
            cropped cerebellar area from transformed T1w image
        t2_crop: (ANTsImage)
            cropped cerebellar area from transformed T2w image
        label_crop: (ANTsImage)
            cropped cerebellar area from transformed label image
        t1_whole: (ANTsImage)
            whole transformed T1w image
        t2_whole: (ANTsImage)
            whole transformed T2w image

    """

    if t1_file is not None:
        t1 = img_read(t1_file)
    else:
        t1 = None
    if t2_file is not None:
        t2 = img_read(t2_file)
    else:
        t2 = None

    # Read additional images
    if label_file is not None:
        label = img_read(label_file)
    else:
        label = None

    # If T1 and T2 are both given, but not aligned, align T2 to T1 first
    if t2 is not None and t1 is not None:
        if ants.get_spacing(t1) != ants.get_spacing(t2):  # JD: is this a bullet-proof  way to check alignment?
            t2 = ants.registration(fixed=t1, moving=t2, type_of_transform='Rigid')['warpedmovout']

    # Apply the brain mask if provided    
    if brain_mask_file is not None:
        brain_mask = img_read(brain_mask_file)
        if t1 is not None:
            brain = ants.mask_image(image=t1, mask=brain_mask)
        else:
            brain = ants.mask_image(image=t2, mask=brain_mask)
        trans = BoundingBox.registration_brain(brain, type_of_transform=type_of_transform)
    else:
        if t1 is not None:
            trans = BoundingBox.registration(t1, type_of_transform=type_of_transform)
        else:
            trans = BoundingBox.registration(t2, type_of_transform=type_of_transform)

    if t1 is not None:
        t1_crop, t1_whole = BoundingBox.crop(t1, trans)
    else:
        t1_crop = None
        t1_whole = None

    if t2 is not None:
        t2_crop, t2_whole = BoundingBox.crop(t2, trans)
    else:
        t2_crop = None
        t2_whole = None

    if label is not None:
        label_crop, _ = BoundingBox.crop(label, trans)
    else:
        label_crop = None

    return trans, t1_crop, t2_crop, label_crop, t1_whole, t2_whole


def threshold(img: ants.ANTsImage, lower: float = 0.5, upper: float = 1.0) -> ants.ANTsImage:
    """
    remove all other values from the image

    Args:
        img: (ANTsImage)
            the input image
        lower: (float)
            lower threshold
        upper: (float)
            upper threshold

    Returns:
        image : (ANTsImage)
            the thresholded image
    """
    img[img < lower] = 0
    img[img > upper] = 0
    return img


def remove_islands(img: ants.ANTsImage) -> ants.ANTsImage:
    """ Removes parts of the mask that is not connected to the largest cluster
    
    Args:
        img (ANTsImage): the input image
    Returns:
        mask (ANTsImage): Image containing the largest connected component
    """
    clusters = ants.image_to_cluster_images(img)

    mask = None
    voxels = 0
    for temp in clusters:
        if temp.numpy().sum() > voxels:
            mask = temp
            voxels = temp.numpy().sum()

    return mask


def subject_postprocess(mask: ants.ANTsImage, trans: ants.ANTsTransform, BoundingBox: TemplateCerebellarBoundingBox, ref: ants.ANTsImage) -> ants.ANTsImage:
    """
    transform the predicted cerebellum mask to the original space
    Args:
        mask: (ANTsImage)
            the predicted cerebellum mask from the template space
        trans: (ANTsTransform)
            the transformation from subject space to template space
        BoundingBox: (TemplateCerebellarBoundingBox)
            the bounding box
        ref: (ANTsImage)
            the reference image

    Returns:
        result: (ANTsImage)
            the final cerebellum mask from the subject space

    """

    result = BoundingBox.template2subject(mask, trans, ref)
    # threshold and binarize the image
    result = threshold(result)
    result[result != 0] = 1

    result = remove_islands(result)
    return result


def isolate(t1_file: str = None, t2_file: str = None, 
            brain_mask_file: str = None, 
            label_file: str = None, 
            result_folder: str = None,
            template: str = 'MNI152NLin6Asym',
            type_of_transform: str = 'Similarity', 
            params: str = 'pre_trained_numpy.pkl', 
            save_cropped_files: bool = False,
            verbose: bool = True) -> ants.ANTsImage:
    """
    main function for cerebellum isolation

    Args:
        t1_file: (string)
            filename and path to T1w image, optional
        t2_file: (string)
            filename and path to T2w image, optional
        brain_mask_file: (string)
            filename and path to brain mask, optional
        label_file: (string)
            filename and path to label image, optional (reserved, currently has no effect)
        result_folder: (string)
            path to output folder (optional, otherwise it is saved to input image folder)
        mask_name: (string)
            name of the output mask (optinal, defaults to '<t1_file>_dseg.nii.gz')
        template: (string)
            template to use (reserved)
        type_of_transform: (string)
            reserved for future use (see ANTspy)
        params: (string)
            path to params file (reserved)
        save_cropped_files: bool
            set to True to save files cropped to window (only works if result_folder is specified)
        verbose: bool
            whether to print out status information during processing
    Returns:
        mask: (ANTsImage)
            predicted cerebellum mask

    """

    if t1_file is not None:
        result_folder = os.path.dirname(os.path.abspath(t1_file)) if result_folder is None else result_folder
        basename = os.path.splitext(os.path.basename(t1_file))
    elif t2_file is not None:
        result_folder = os.path.dirname(os.path.abspath(t2_file)) if result_folder is None else result_folder
        basename = os.path.splitext(os.path.basename(t2_file))
    else:
        raise RuntimeError('Must specify either t1_file or t2_file')

    # Strip .nii or .nii.gz extension 
    if basename[1] == '.gz':
        basename = os.path.splitext(basename[0])
    basename = basename[0]

    # find paramter file and template bounding box 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    params_file = os.path.join(base_dir, 'parameters', params)
    BoundingBox = TemplateCerebellarBoundingBox(name=template)

    # Crop the images to the Unet input window
    if verbose:
        print("preprocessing")
    trans, t1_crop, t2_crop, label_crop, _, _ = subject_preprocess(t1_file=t1_file,
                                                                   t2_file=t2_file,
                                                                   brain_mask_file=brain_mask_file,
                                                                   label_file=label_file,
                                                                   BoundingBox=BoundingBox,
                                                                   type_of_transform=type_of_transform)
    if isinstance(t1_crop, ants.core.ants_image.ANTsImage):
        t1_crop_data = t1_crop.numpy()
    else:
        t1_crop_data = t1_crop
    if isinstance(t2_crop, ants.core.ants_image.ANTsImage):
        t2_crop_data = t2_crop.numpy()
    else:
        t2_crop_data = t2_crop
    if isinstance(label_crop, ants.core.ants_image.ANTsImage):
        label_crop_data = label_crop.numpy()
    else:
        label_crop_data = label_crop


    # Do a forward pass through the Unet model
    if verbose:
        print('isolating cerebellum using UNet model')
    mask = predict(params_file=params_file, t1=t1_crop_data, t2=t2_crop_data)
    mask = nib.Nifti1Image(mask, BoundingBox.get_cropped_affine())
    mask = from_nibabel(mask)

    # Postprocess and transform the mask back to subject space
    if verbose:
        print('postprocessing')
    if t1_file is not None:
        result = subject_postprocess(mask=mask, trans=trans, BoundingBox=BoundingBox, ref=img_read(t1_file))
    else:
        result = subject_postprocess(mask=mask, trans=trans, BoundingBox=BoundingBox, ref=img_read(t2_file))
    if result_folder is not None:
        os.makedirs(result_folder, exist_ok=True)
        if verbose:
            print(f"saving results into {result_folder}")
        ants.image_write(result, os.path.join(result_folder, f'{basename}_cerebellum_dseg.nii.gz'))

        if save_cropped_files:
            if t1_crop is not None:
                ants.image_write(t1_crop, os.path.join(result_folder, f'{basename}_crop.nii.gz'))
            else:
                ants.image_write(t2_crop, os.path.join(result_folder, f'{basename}_crop.nii.gz'))
            ants.image_write(mask, os.path.join(result_folder, f'{basename}_cerebellum_crop_dseg.nii.gz'))
            ants.write_transform(trans, os.path.join(result_folder, f'{basename}_trans.mat'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--T1', type=str, help='path to T1w image')
    parser.add_argument('--T2', type=str, help='path to T2w image')
    parser.add_argument('--brain_mask', type=str, help='path to brain mask image')
    parser.add_argument('--label', type=str, help='path to label image')
    parser.add_argument('--result_folder', type=str, help='path to save the isolation image (results will be saved to '
                                                          'T1w image folder (or T2w image folder if no T1w image is '
                                                          'specified))')
    parser.add_argument('--template', type=str, default='MNI152NLin6Asym',
                        help='template for registration (MNI152NLin6Asym by '
                             'default)')
    parser.add_argument('--params', type=str, default='pre_trained.pkl', help='pretrained parameter file')
    parser.add_argument('--save_cropped_files', action='store_true', help='Save files cropped to UNet input window')

    args = parser.parse_args()

    if args.T1 is None and args.T2 is None:
        raise RuntimeError('Must specify either t1_file or t2_file')

    if args.result_folder is None:
        if args.T1 is None:
            args.result_folder = os.path.dirname(os.path.abspath(args.T2))
        else:
            args.result_folder = os.path.dirname(os.path.abspath(args.T1))

    result = isolate(t1_file=args.T1,
                     t2_file=args.T2,
                     brain_mask_file=args.brain_mask,
                     label_file=args.label,
                     result_folder=args.result_folder,
                     template=args.template,
                     params=args.params,
                     save_cropped_files=args.save_cropped_files,)
