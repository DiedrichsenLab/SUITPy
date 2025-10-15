"""
Cerebellar Isolation using a Unet model
authors: Yao Li, Carlos Hernandez-Castillo, Joern Diedrichsen
"""

import sys
import argparse
import os
import nibabel as nib
import ants
import torch
from torch import nn
from tempfile import mkstemp
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def from_nibabel(nib_image):
    """
    Convert a nibabel image to an ANTsImage (https://antspy.readthedocs.io/en/v0.3.8/index.html)
    """
    fd, tmpfile = mkstemp(suffix=".nii.gz")
    nib_image.to_filename(tmpfile)
    new_img = ants.image_read(tmpfile)
    os.close(fd)
    os.remove(tmpfile)
    return new_img


def img_read(path):

    nib_img = nib.load(path)
    header = nib_img.header
    header.set_qform(None)
    new_img = from_nibabel(nib.Nifti1Image(nib_img.dataobj, nib_img.affine, header))
    return new_img


def standardize(x):
    """
    Standardizes input tensor x to have mean 0 and standard deviation 1.
    Args:
    ----------
        x: tensor
            the tensor to be standardized

    Returns:
    ----------
        data: tensor
            standardized tensor

    """
    x_mean, x_std = x.mean(), x.std()
    return (x - x_mean) / x_std


class _MNITemplate(object):
    """
        Basic MNI template class, which defines the cropped area. 
        All other template implementations should be registered to this template. (should not be instantiated directly)
    """
    def __init__(self):
        # location and size of the cropped area (changing it might result worse performance as the model was trained on it)
        self.cropped_size = (128, 128, 128)
        self.cereb_center = [0, -50, -24]

        base_dir = os.path.dirname(__file__)
        self.MNI_template = os.path.join(base_dir, f'templates/MNI_2009.nii.gz')
        self.mni_affine = nib.load(self.MNI_template).affine

    def get_cropped_coordinates(self):
        """
        calculate the coordinates of the cropped area.
        """
        start = (self.cereb_center[0] - self.cropped_size[0] / 2 * self.mni_affine[0, 0],
                 self.cereb_center[1] - self.cropped_size[1] / 2 * self.mni_affine[1, 1],
                 self.cereb_center[2] - self.cropped_size[2] / 2 * self.mni_affine[2, 2])
        end = (self.cereb_center[0] + self.cropped_size[0] / 2 * self.mni_affine[0, 0],
               self.cereb_center[1] + self.cropped_size[1] / 2 * self.mni_affine[1, 1],
               self.cereb_center[2] + self.cropped_size[2] / 2 * self.mni_affine[2, 2])
        return start, end

    def get_mni_template(self):
        """
        get the MNI template image
        """
        return ants.image_read(self.MNI_template)

    def get_mni_affine(self):
        """
        get the MNI template affine
        """
        return self.mni_affine

    def get_crop_indices(self):
        """
        calculate the indices of the cropped area.
        """
        start_coords, end_coords = self.get_cropped_coordinates()
        affine_inv = np.linalg.inv(self.mni_affine)
        start = affine_inv @ np.array([start_coords[0], start_coords[1], start_coords[2], 1])
        end = affine_inv @ np.array([end_coords[0], end_coords[1], end_coords[2], 1])

        return start[0:3].astype('int'), end[0:3].astype('int')

    def get_cropped_affine(self):
        """
        get the cropped area affine
        """
        start_coords, end_coords = self.get_cropped_coordinates()
        affine = np.diag([self.mni_affine[0, 0], self.mni_affine[1, 1], self.mni_affine[2, 2], 1])
        affine[0, 3] = abs(affine[0, 0]) * start_coords[0]
        affine[1, 3] = abs(affine[1, 1]) * start_coords[1]
        affine[2, 3] = abs(affine[2, 2]) * start_coords[2]

        return affine

    # def crop(self, img, trans=None):
    #     """
    #
    #     Args:
    #     ----------
    #         img: ANTsImage
    #             image to be cropped
    #         trans: ANTsTransform
    #             transformation matrix from the image space to the MNI template space (only use it if img is not in the MNI template)
    #
    #     Returns:
    #     ----------
    #         cropped_img: ANTsImage
    #             cropped image
    #
    #     """
    #     start_indices, end_indices = self.get_crop_indices()
    #     if trans is not None:
    #         img = ants.apply_ants_transform_to_image(trans, img, self.get_mni_template())
    #     return img[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]]


class AdultTemplate(_MNITemplate):
    """
    adult template
    """
    def __init__(self):
        super().__init__()

        base_dir = os.path.dirname(__file__)
        self.template_head = os.path.join(base_dir, f'templates/adult/head.nii.gz')
        self.template_brain = os.path.join(base_dir, f'templates/adult/brain.nii.gz')

    def get_template(self):
        """
        get the template head
        """
        return ants.image_read(self.template_head)

    def get_template_brain(self):
        """
        get the template brain
        """
        return ants.image_read(self.template_brain)

    def registration(self, img, type_of_transform='Affine'):
        """
        register the image to this template

        Args:
        ----------
            img: ANTsImage
                image to be registered
            type_of_transform: string, optional
                transform type (Affine by default, check ANTsPY[https://antspy.readthedocs.io/en/latest/registration.html] for details)

        Returns:
        ----------
            image: ANTsImage
                the registered image
            trans: ANTsTransform
                the transformation from the subject space to the template space

        """
        result = ants.registration(fixed=self.get_template(), moving=img, type_of_transform=type_of_transform)
        img = result['warpedmovout']
        trans = ants.read_transform(result['fwdtransforms'][0])

        return trans

    def registration_brain(self, img, type_of_transform='Affine'):
        """
        register the image to this template using the brain. The input image should be brain only.

        Args:
        ----------
            img: ANTsImage
                image to be registered
            type_of_transform: string, optional
                transform type (Affine by default, check ANTsPY[https://antspy.readthedocs.io/en/latest/registration.html] for details)

        Returns:
        ----------
            image: ANTsImage
                the registered image
            trans: ANTsTransform
                the transformation from the subject space to the template space

        """
        result = ants.registration(fixed=self.get_template_brain(), moving=img, type_of_transform=type_of_transform)
        img = result['warpedmovout']
        trans = ants.read_transform(result['fwdtransforms'][0])

        return trans


class Subject(object):
    """
    Subject holds information about a subject.
    """

    def __init__(self, t1, t2, label):
        """

        Args:
        ----------
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
        ----------
            t1_data: Tensor
                tensor of T1w cerebellar image (after cropping)
            t2_data: Tensor
                tensor of T2w cerebellar image (after cropping)
            label_data: Tensor
                tensor of label cerebellar image (after cropping)

        """
        if self.t1 is None:
            t1_data = torch.zeros((128, 128, 128), dtype=torch.float)
        else:
            t1_data = torch.tensor(self.t1, dtype=torch.float)
            t1_data = standardize(t1_data)
        if self.t2 is None:
            t2_data = torch.zeros((128, 128, 128), dtype=torch.float)
        else:
            t2_data = torch.tensor(self.t2, dtype=torch.float)
            t2_data = standardize(t2_data)

        if self.label is None:
            label_data = torch.zeros((128, 128, 128), dtype=torch.float)
        else:
            label_data = torch.tensor(self.label.numpy(), dtype=torch.float)
        return t1_data, t2_data, label_data


def subject_preprocess(t1_path=None, t2_path=None, brain_path=None, brain_mask_path=None, label_path=None, template=AdultTemplate(),
                       type_of_transform='Affine'):
    """
    function to preprocess a single subject

    Args:
    ----------
        t1_path: string
            Path to T1w image
        t2_path: string
            Path to T2w image
        brain_path: string, optional
            Path to brain image
        label_path: string, optional
            Path to label image (This image will be transformed into the template space using the same transformation.)
        template:
        type_of_transform: string, optional
            reserved for future use (see ANTspy)

    Returns:
    ----------
        trans: ANTsTransform
            transformation from subject space to template space
        t1_crop: ANTsImage
            cropped cerebellar area from transformed T1w image
        t2_crop: ANTsImage
            cropped cerebellar area from transformed T2w image
        label_crop: ANTsImage
            cropped cerebellar area from transformed label image
        t1_whole: ANTsImage
            whole transformed T1w image
        t2_whole: ANTsImage
            whole transformed T2w image

    """

    if t1_path is not None:
        t1 = img_read(t1_path)
    else:
        t1 = None
    if t2_path is not None:
        t2 = img_read(t2_path)
    else:
        t2 = None
    if brain_mask_path is not None:
        brain_mask = img_read(brain_mask_path)
        if t1 is not None:
            brain = ants.mask_image(image=t1, mask=brain_mask)
        else:
            brain = ants.mask_image(image=t2, mask=brain_mask)
    elif brain_path is not None:
        brain = img_read(brain_path)
    else:
        brain = None
    if label_path is not None:
        label = img_read(label_path)
    else:
        label = None

    if t2 is not None and t1 is not None:
        if ants.get_spacing(t1) != ants.get_spacing(t2):
            t2 = ants.registration(fixed=t1, moving=t2, type_of_transform='Rigid')['warpedmovout']

    if brain is not None:
        trans = template.registration_brain(brain, type_of_transform=type_of_transform)
    else:
        if t1 is not None:
            trans = template.registration(t1, type_of_transform=type_of_transform)
        else:
            trans = template.registration(t2, type_of_transform=type_of_transform)

    if t1 is not None:
        t1_crop, t1_whole = template.crop(t1, trans)
        # t1_whole = ants.apply_ants_transform_to_image(trans, t1, template.get_mni_template())
    else:
        t1_crop = None
        t1_whole = None

    if t2 is not None:
        t2_crop, t2_whole = template.crop(t2, trans)
        # t2_whole = ants.apply_ants_transform_to_image(trans, t2, template.get_mni_template())
    else:
        t2_crop = None
        t2_whole = None

    if label is not None:
        label_crop, _ = template.crop(label, trans)
    else:
        label_crop = None

    return trans, t1_crop, t2_crop, label_crop, t1_whole, t2_whole


def threshold(img, lower=0.95, upper=1.0):
    """
    remove all other values from the image

    Args:
    ----------
        img: ANTsImage
            the input image
        lower: float
            lower threshold
        upper: float
            upper threshold

    Returns:
    ----------
        image : ANTsImage
            the thresholded image
    """
    img[img < lower] = 0
    img[img > upper] = 0
    return img


def binarize(img):
    """
    binarize the image

    Args:
    ----------
        img: ANTsImage
            the input image

    Returns:
    ----------
        image : ANTsImage
            the binarized image

    """
    img[img != 0] = 1
    # img.to(np.uint8)
    # return img.where(img != 0, 1, 0).to(np.uint8)
    return img


def denoise(img):
    clusters = ants.image_to_cluster_images(img)

    mask = None
    voxels = 0
    for temp in clusters:
        if temp.numpy().sum() > voxels:
            mask = temp
            voxels = temp.numpy().sum()

    return mask


def subject_postprocess(mask, trans, template, ref):
    """
    transform the predicted cerebellum mask to the original space
    Args:
    ----------
        mask: ANTsImage
            the predicted cerebellum mask from the template space
        trans: ANTsTransform
            the transformation from subject space to template space
        ref: ANTsImage
            the reference image

    Returns:
        result: ANTsImage
            the final cerebellum mask from the subject space

    """

    result = template.warp_to_MNI(mask, trans, ref)
    result = binarize(threshold(result))
    result = denoise(result)
    return result






# convolution block in Unet
class _ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
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
        return self.layer(x)


class _DownSample(nn.Module):
    def __init__(self, channels):
        super(_DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(channels, affine=True, track_running_stats=True),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class _UpSample(nn.Module):
    def __init__(self, channels):
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
        ----------
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




def _load_model(model, params_path):
    """
    load model with pretrained weights

    Args:
    ----------
        model: Unet model
        params_path: string
            path to the pretrained weights

    Returns:
    ----------
        net: the pretrained model
    """
    net = model.to(device)
    if os.path.exists(params_path):
        net.load_state_dict(torch.load(params_path, weights_only=True, map_location=device))
    else:
        print('fail to load weights')
        exit(0)
    return net


def predict(model, params_path, t1=None, t2=None):
    """
    Run a prediction on a single subject using a trained UNet model

    Args:
    ----------
        model: Unet model
        params_path: string
            path to the pretrained weights
        t1: Tensor
            tensor of T1w cerebellar image (after cropping)
        t2: Tensor
            tensor of T2w cerebellar image (after cropping)

    Returns:
    ----------
        mask: ndarray
            the 3D numpy array of predicted mask (template space)

    """
    net = _load_model(model, params_path)
    if t1 is None:
        t1 = torch.zeros((128, 128, 128), dtype=torch.float)
    if t2 is None:
        t2 = torch.zeros((128, 128, 128), dtype=torch.float)
    t1, t2 = t1.unsqueeze(0).unsqueeze(0), t2.unsqueeze(0).unsqueeze(0)
    t1, t2 = t1.to(device), t2.to(device)
    mask = net(t1, t2)
    mask = mask.cpu().detach().numpy()
    return mask[0][0]





def isolate(t1_path=None, t2_path=None, brain_path=None, brain_mask_path=None, label_path=None, result_folder=None, template='adult',
            type_of_transform='Affine', model=UNet(), params='pre_trained.pkl', keepfiles=False):
    """
    main function for cerebellum isolation

    Args:
    ----------
        t1_path: string, optional
            path to T1w image
        t2_path: string, optional
            path to T2w image
        brain_path: string, optional
            path to brain image
        brain_mask_path: string, optional
            path to brain mask
        label_path: string, optional
            path to label image
        result_folder: string, optional
            path to output folder (nothing will be saved if None)
        template: string, optional
            template to use (reserved)
        type_of_transform: string, optional
            reserved for future use (see ANTspy)
        model: UNet, optional
            reserved
        params: string, optional
            path to params file
        keepfiles: bool, optional
            set to True to keep intermediate files, defaults to False (only works if result_folder is specified)

    Returns:
    ----------
        mask: ANTsImage
            predicted cerebellum mask

    """

    if t1_path is None and t2_path is None:
        print('No input images found')
        exit(0)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(base_dir, 'params', params)
    if template == 'adult':
        template = AdultTemplate()
    elif template == 'neonate':
        template = NeonateTemplate()
    elif template == 'elder':
        template = ElderTemplate()
    else:
        print('Unsupported template type')
        exit(0)
    trans, t1_crop, t2_crop, label_crop, t1_whole, t2_whole = subject_preprocess(t1_path=t1_path, t2_path=t2_path,
                                                                                 brain_path=brain_path,
                                                                                 brain_mask_path = brain_mask_path,
                                                                                 label_path=label_path,
                                                                                 template=template,
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

    sub = Subject(t1=t1_crop_data, t2=t2_crop_data, label=label_crop_data)
    t1, t2, label = sub.get_data()
    mask = predict(model=model, params_path=params_path, t1=t1, t2=t2)
    mask = nib.Nifti1Image(mask, template.get_cropped_affine())
    mask = from_nibabel(mask)

    if t1_path is not None:
        result = subject_postprocess(mask=mask, trans=trans, template=template, ref=img_read(t1_path))
    else:
        result = subject_postprocess(mask=mask, trans=trans, template=template, ref=img_read(t2_path))
    if result_folder is not None:
        os.makedirs(result_folder, exist_ok=True)
        ants.image_write(result, os.path.join(result_folder, 'cerebellum_Unet_dseg.nii.gz'))

        if keepfiles:
            if t1_crop is not None:
                ants.image_write(t1_crop, os.path.join(result_folder, 'T1w_crop.nii.gz'))
                ants.image_write(t1_whole, os.path.join(result_folder, 'T1w_whole.nii.gz'))
            if t2_crop is not None:
                ants.image_write(t2_crop, os.path.join(result_folder, 'T2w_crop.nii.gz'))
                ants.image_write(t2_whole, os.path.join(result_folder, 'T2w_whole.nii.gz'))
            if label_crop is not None:
                ants.image_write(label_crop, os.path.join(result_folder, 'label_crop.nii.gz'))
            ants.image_write(mask, os.path.join(result_folder, 'Unet_pm_MNI.nii.gz'))
            ants.write_transform(trans, os.path.join(result_folder, 'trans.mat'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--T1', type=str, help='path to T1w image')
    parser.add_argument('--T2', type=str, help='path to T2w image')
    parser.add_argument('--brain', type=str, help='path to brain image')
    parser.add_argument('--label', type=str, help='path to label image')
    parser.add_argument('--result_folder', type=str, help='path to save the isolation image (results will be saved to '
                                                          'T1w image folder (or T2w image folder if no T1w image is '
                                                          'specified))')
    parser.add_argument('--template', type=str, default='adult', help='template for registration (adult by '
                                                                      'default)')
    parser.add_argument('--params', type=str, default='best.pkl', help='pretrained parameter file')
    parser.add_argument('--keepfiles', action='store_true', help='keep intermediate files')

    args = parser.parse_args()

    if args.T1 is None and args.T2 is None:
        print('No input images found')
        exit(0)

    if args.result_folder is None:
        if args.T1 is None:
            args.result_folder = os.path.dirname(os.path.abspath(args.T2))
        else:
            args.result_folder = os.path.dirname(os.path.abspath(args.T1))

    result = isolate(t1_path=args.T1, t2_path=args.T2, brain_path=args.brain, label_path=args.label,
                     result_folder=args.result_folder, template=args.template, params=args.params)
