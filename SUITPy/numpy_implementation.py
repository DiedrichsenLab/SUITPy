import numpy as np
from typing import Tuple, Union
import pickle
import os


class _Conv3dN:
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int, int]],
            stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
            padding: Union[int, Tuple[int, int, int]] = (0, 0, 0),
            dilation: Union[int, Tuple[int, int, int]] = (1, 1, 1),
            bias: bool = True
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3
        self.bias = bias

        self.weight = np.random.randn(out_channels, in_channels, *self.kernel_size)

        if bias:
            self.bias_term = np.zeros(out_channels)
        else:
            self.bias_term = None

    def _im2col(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        batch_size, in_channels, depth, height, width = x.shape
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding
        dd, dh, dw = self.dilation

        if any(p != 0 for p in [pd, ph, pw]):
            x_padded = np.pad(x, ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)), mode='constant')
        else:
            x_padded = x

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
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int, int]],
            stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
            padding: Union[int, Tuple[int, int, int]] = (0, 0, 0),
            output_padding: Union[int, Tuple[int, int, int]] = (0, 0, 0),
            dilation: Union[int, Tuple[int, int, int]] = (1, 1, 1),
            bias: bool = True
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding,) * 3
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3
        self.bias = bias

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

        return np.pad(x, padding, mode='constant')

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
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True
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
            self.running_mean = np.zeros(num_features)
            self.running_var = np.ones(num_features)
            self.num_batches_tracked = 0
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
    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def forward(self, x: np.ndarray) -> np.ndarray:

        return np.maximum(0, x) + self.negative_slope * np.minimum(0, x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class _SoftmaxN:
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
    def __init__(self, in_channels: int, out_channels: int):
        self.conv1 = _Conv3dN(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm1 = _InstanceNorm3dN(num_features=out_channels, affine=True, track_running_stats=True)
        self.activate1= _LeakyReLUN()

        self.conv2 = _Conv3dN(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm2 = _InstanceNorm3dN(num_features=out_channels, affine=True, track_running_stats=True)
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
        self.conv1.load_state(weight=c_weight1, bias_term=c_bias1)
        self.norm1.load_state_dict(state_dict=n_state_dict1)
        self.conv2.load_state(weight=c_weight2, bias_term=c_bias2)
        self.norm2.load_state_dict(state_dict=n_state_dict2)

    def forward(self, x: np.ndarray) -> np.ndarray:
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
    def __init__(self, channels: int):
        self.conv1 = _Conv3dN(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)
        self.norm1 = _InstanceNorm3dN(num_features=channels, affine=True, track_running_stats=True)
        self.activate1 = _LeakyReLUN()

    def load_state(
            self,
            c_weight: np.ndarray,
            c_bias: np.ndarray,
            n_state_dict: dict
    ):
        self.conv1.load_state(weight=c_weight, bias_term=c_bias)
        self.norm1.load_state_dict(state_dict=n_state_dict)

    def forward(self, x: np.ndarray) -> np.ndarray:
        result1 = self.conv1(x)
        result2 = self.norm1(result1)
        result3 = self.activate1(result2)

        return result3

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class _UpSampleN:
    def __init__(self, channels: int):
        self.conv1 = _ConvTranspose3dN(in_channels=channels, out_channels=channels, kernel_size=2, stride=2, padding=0)
        self.norm1 = _InstanceNorm3dN(num_features=channels, affine=True, track_running_stats=True)
        self.activate1 = _LeakyReLUN()

        self.conv2 = _Conv3dN(in_channels=channels * 2, out_channels=channels, kernel_size=3, padding=1)
        self.norm2 = _InstanceNorm3dN(num_features=channels, affine=True, track_running_stats=True)
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
        self.conv1.load_state(weight=c_weight1, bias_term=c_bias1)
        self.norm1.load_state_dict(state_dict=n_state_dict1)
        self.conv2.load_state(weight=c_weight2, bias_term=c_bias2)
        self.norm2.load_state_dict(state_dict=n_state_dict2)

    def forward(self, x: np.ndarray, feature_map: np.ndarray) -> np.ndarray:
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
                'running_mean': state_dict[f'{c}.layer.1.running_mean'],
                'running_var': state_dict[f'{c}.layer.1.running_var'],
                'num_batches_tracked': state_dict[f'{c}.layer.1.num_batches_tracked']
            }
            c_weight2 = state_dict[f'{c}.layer.3.weight']
            c_bias2 = state_dict[f'{c}.layer.3.bias']
            n2 = {
                'weight': state_dict[f'{c}.layer.4.weight'],
                'bias': state_dict[f'{c}.layer.4.bias'],
                'running_mean': state_dict[f'{c}.layer.4.running_mean'],
                'running_var': state_dict[f'{c}.layer.4.running_var'],
                'num_batches_tracked': state_dict[f'{c}.layer.4.num_batches_tracked']
            }
            convs[c].load_state(c_weight1=c_weight1, c_bias1=c_bias1, n_state_dict1=n1, c_weight2=c_weight2, c_bias2=c_bias2, n_state_dict2=n2)

        for d in downs:
            c_weight = state_dict[f'{d}.layer.0.weight']
            c_bias = state_dict[f'{d}.layer.0.bias']
            n = {
                'weight': state_dict[f'{d}.layer.1.weight'],
                'bias': state_dict[f'{d}.layer.1.bias'],
                'running_mean': state_dict[f'{d}.layer.1.running_mean'],
                'running_var': state_dict[f'{d}.layer.1.running_var'],
                'num_batches_tracked': state_dict[f'{d}.layer.1.num_batches_tracked']
            }
            downs[d].load_state(c_weight=c_weight, c_bias=c_bias, n_state_dict=n)

        for u in ups:
            c_weight1 = state_dict[f'{u}.up.0.weight']
            c_bias1 = state_dict[f'{u}.up.0.bias']
            n1 = {
                'weight': state_dict[f'{u}.up.1.weight'],
                'bias': state_dict[f'{u}.up.1.bias'],
                'running_mean': state_dict[f'{u}.up.1.running_mean'],
                'running_var': state_dict[f'{u}.up.1.running_var'],
                'num_batches_tracked': state_dict[f'{u}.up.1.num_batches_tracked']
            }
            c_weight2 = state_dict[f'{u}.layer.0.weight']
            c_bias2 = state_dict[f'{u}.layer.0.bias']
            n2 = {
                'weight': state_dict[f'{u}.layer.1.weight'],
                'bias': state_dict[f'{u}.layer.1.bias'],
                'running_mean': state_dict[f'{u}.layer.1.running_mean'],
                'running_var': state_dict[f'{u}.layer.1.running_var'],
                'num_batches_tracked': state_dict[f'{u}.layer.1.num_batches_tracked']
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

def _load_model(model, params_file: str):
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
    net = model
    if os.path.exists(params_file):
        with open(params_file, "rb") as f:
            params = pickle.load(f)
        net.load_state_dict(params)
    else:
        print('fail to load weights')
        exit(0)
    return net


def predict(model, params_file: str, t1: np.ndarray = None, t2: np.ndarray = None) -> np.ndarray:
    """
    Run a prediction on a single subject using a trained UNet model

    Args:
        model: (Unet)
            Unet model
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
    net = _load_model(model, params_file)
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

