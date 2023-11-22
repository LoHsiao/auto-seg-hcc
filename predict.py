import os
from os.path import join, exists
import logging
import sys
from tqdm import tqdm

import torch
import monai
from monai.data import Dataset, CacheDataset, DataLoader, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    SpatialPadd,
    Spacingd,
    ConcatItemsd,
    NormalizeIntensityd,
    EnsureTyped,
    AsDiscreted,
    Invertd,
    SaveImaged,
)

from monai.networks.nets import DynUNet

torch.multiprocessing.set_sharing_strategy('file_system')  # for num_workers > 1
torch.backends.cudnn.benchmark = True

project_path = join(os.getcwd())

source_path = input_dir = sys.argv[1]
predict_path = sys.argv[2]
training_fold = sys.argv[3]  # 0, 1, 2, 3, 4
log_dir = join(project_path, 'runs/fold_'+training_fold)
pred_dir = join(predict_path, 'fold_'+training_fold)
npz_pred_dir = join(predict_path, 'npz_fold_'+training_fold)
if not exists(pred_dir):
    os.mkdir(pred_dir)
if not exists(npz_pred_dir):
    os.mkdir(npz_pred_dir)

num_workers = 4  # quadro of GPU number
image_shape = None
patch_shape = (128, 128, 64)

basic_channel = 32
n_deep_supervision = 0
pretrained_model = os.path.join(log_dir, 'model_best.pth')


def input_dird(input_dir, modalities):
    all_files = os.listdir(input_dir)
    all_files.sort()
    all_keys = []
    for each_file in all_files:
        if each_file.endswith('.nii.gz'):
            if each_file.split('_')[0] not in all_keys:
                all_keys.append(each_file.split('_')[0])
    
    all_paths = {each_key: {} for each_key in all_keys}
    for each_file in all_files:
        for i_mod in range(len(modalities)):
            if each_file.endswith('_'+str(i_mod).zfill(4)+'.nii.gz'):
                all_paths[each_file.split('_')[0]][modalities[i_mod]] = join(input_dir, each_file)

    test_files = []
    for each_key in all_paths.keys():
        test_files.append(all_paths[each_key]) 
    return None, test_files


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    PHASES = ["arterial", "venous", "delayed"]  
    train_files, eval_files = input_dird(input_dir, modalities=PHASES)

    # transforms of checking, training and validation dataset
    check_transforms = Compose(
        [
            LoadImaged(keys=PHASES, image_only=False),
            EnsureChannelFirstd(keys=PHASES),
            Spacingd(keys=PHASES, pixdim=(1.0, 1.0, 2.8)),
            SpatialPadd(keys=PHASES, spatial_size=patch_shape, method="end"),
            ConcatItemsd(keys=PHASES, name="volume", dim=0),
            EnsureTyped(keys=["volume"]),
        ]
    )

    eval_transforms = Compose(
        [
            LoadImaged(keys=PHASES, image_only=False),
            EnsureChannelFirstd(keys=PHASES),
            NormalizeIntensityd(keys=PHASES),
            Spacingd(keys=PHASES, pixdim=(1.0, 1.0, 2.8)),
            SpatialPadd(keys=PHASES, spatial_size=patch_shape, method="end"),
            ConcatItemsd(keys=PHASES, name="volume", dim=0),
            EnsureTyped(keys=["volume"]),
        ]
    )

    # define dataset, data loader
    check_ds = Dataset(data=eval_files, transform=check_transforms)
    # check images for network training
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["volume"].shape)

    # create a validation data loader
    eval_ds = CacheDataset(data=eval_files, transform=eval_transforms, cache_rate=0)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=list_data_collate
        )
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DynUNet_metadata = dict(
        spatial_dims=3, 
        in_channels=3, 
        out_channels=2, 
        strides=[
            [1,1,1], [2,2,1], [2,2,2],
            [2,2,2], [2,2,2], [2,2,1]
            ], 
        kernel_size=[
            [3,3,3], [3,3,3], [3,3,3],
            [3,3,3], [3,3,3], [3,3,3]
            ], 
        upsample_kernel_size=[             
            [2,2,1], [2,2,2],
            [2,2,2], [2,2,2], [2,2,1]
            ], 
        filters=[min(basic_channel * (2 ** i), 320) for i in range(6)], 
        dropout=None, 
        norm_name=('INSTANCE', {'affine': True}), 
        act_name=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}), 
        deep_supervision=(n_deep_supervision > 0), 
        deep_supr_num=n_deep_supervision, 
        res_block=False, 
        trans_bias=False
    )
    model = DynUNet(**DynUNet_metadata).to(device)
    if os.path.exists(pretrained_model):
        try:
            checkpoint = torch.load(pretrained_model, map_location=device)
            checkpoint_epoch = checkpoint['epoch']
            print('Loading pretrained model at epoch', checkpoint_epoch, 'from', pretrained_model, '...')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # no deep supervision 
            print('Loading pretrained model successfully.')  
        except:
            print('Loading pretrained model failed, please check the model path.')
            return EOFError

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    
    # define evaluation method for model
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=eval_transforms,
            orig_keys="arterial",
            meta_keys="pred_meta_dict",
            orig_meta_keys="arterial_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", 
                   output_dir=npz_pred_dir, output_postfix="npz", 
                   output_ext=".nii.gz", separate_folder=False, 
                   resample=False, print_log=False),
        AsDiscreted(keys="pred", argmax=True),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", 
                   output_dir=pred_dir, output_postfix="", 
                   output_ext=".nii.gz", separate_folder=False, 
                   resample=False, print_log=False),
    ])

    model.eval()
    with torch.no_grad():
        eval_images = None

        eval_loop = tqdm(eval_loader, total=len(eval_loader), leave=True)
        eval_loop.set_description(f'Inferece')
        for eval_data in eval_loop:
            eval_images = eval_data["volume"].to(device)
            roi_size = patch_shape  # (128, 128, 64)
            sw_batch_size = 2
            eval_data['pred'] = sliding_window_inference(eval_images, roi_size, sw_batch_size, model)
            eval_data = [post_transforms(i) for i in decollate_batch(eval_data)]
        for each in os.listdir(pred_dir):
            if each.endswith('_0000.nii.gz'):
                os.rename(join(pred_dir, each), join(pred_dir, each.replace('_0000.nii.gz', '.nii.gz')))
        for each in os.listdir(npz_pred_dir):
            if each.endswith('_0000_npz.nii.gz'):
                os.rename(join(npz_pred_dir, each), join(npz_pred_dir, each.replace('_0000_npz.nii.gz', '_npz.nii.gz')))


if __name__ == "__main__":
    main()
