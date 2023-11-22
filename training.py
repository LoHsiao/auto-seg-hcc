import os
from os.path import join
import json
import logging
import sys
from tqdm import tqdm

import torch
import aim

import monai
from monai.data import Dataset, CacheDataset, DataLoader
from monai.data import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.metrics import LossMetric

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    SpatialPadd,
    NormalizeIntensityd,
    Spacingd,
    ConcatItemsd,
    RandFlipd,
    RandRotated,
    RandCropByPosNegLabeld,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    EnsureTyped,
    EnsureType,
    AsDiscrete
)

from monai.networks.nets import DynUNet


torch.multiprocessing.set_sharing_strategy('file_system')  # for num_workers > 1
torch.backends.cudnn.benchmark = True

project_path = join(os.getcwd())
json_file = join(project_path, 'datalist/five_fold_dict.json')

training_fold = sys.argv[1]
log_dir = join(project_path, 'runs/fold_'+training_fold)
# log_dir = join(project_path, 'runs/check')
model_track_dir = log_dir
if not os.path.exists(model_track_dir):
    os.mkdir(model_track_dir)

num_workers = 4  # quadro of GPU number
training_or_test = 'training'
batch_size = 2

image_shape = None
patch_shape = (128, 128, 64)
num_samples = 2

one_hot_labels = 1
basic_channel = 32
n_epochs = 1000
initial_learning_rate = 0.01
n_deep_supervision = 3
w_deep_supervision = []
if n_deep_supervision == 1:
    w_deep_supervision = [2, 1]
elif n_deep_supervision == 2:
    w_deep_supervision = [4, 2, 1]
elif n_deep_supervision == 3:
    w_deep_supervision = [8, 4, 2, 1]
elif n_deep_supervision == 4:
    w_deep_supervision = [16, 8, 4, 2, 1]


try:
    aim_run_hash = sys.argv[2]
    if len(aim_run_hash) >= 10:
        continue_training = True
    else:
        continue_training = False
except:
    continue_training = False
if continue_training:
    pretrained_model = os.path.join(log_dir, 'model_latest.pth')


def input_json_datasetd(json_file, training_or_test='training', fold="0"):
    with open(json_file, 'r') as f:
        data_dict = json.load(f)
    
    modalities = data_dict['modalities']
    if training_or_test == 'training' and fold is not None:
        train_list = data_dict['fold'][fold]['train']
        train_files = []
        for i in range(len(train_list)):
            i_dict = {
                "label": join(data_dict['label path'], train_list[i]+'.nii.gz'),
            }
            for j in range(len(modalities)):
                i_dict[modalities[j]] = join(data_dict['image path'], train_list[i]+'_'+str(j).zfill(4)+'.nii.gz')    
            train_files.append(i_dict)
        
        eval_list = data_dict['fold'][fold]['eval']
        eval_files = []
        for i in range(len(eval_list)):
            i_dict = {
                "label": join(data_dict['label path'], train_list[i]+'.nii.gz'),
            }
            for j in range(len(modalities)):
                i_dict[modalities[j]] = join(data_dict['image path'], train_list[i]+'_'+str(j).zfill(4)+'.nii.gz')    
            eval_files.append(i_dict)    
        return train_files, eval_files


def adjust_learning_rate(optimizer, epoch, initial_lr, max_epochs):
    '''The learning rate is decayed throughout the training following the 'poly' learning rate policy.'''
    lr = initial_lr * (1 - epoch / max_epochs)**0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    train_files, eval_files = input_json_datasetd(json_file, training_or_test=training_or_test, fold=training_fold)
    PHASES = ["arterial", "venous", "delayed"]  
    # transforms of checking, training and validation dataset
    check_transforms = Compose(
        [
            LoadImaged(keys=PHASES+["label"], image_only=False),
            EnsureChannelFirstd(keys=PHASES+["label"]),
            Spacingd(keys=PHASES, pixdim=(1.0, 1.0, 2.8)),
            SpatialPadd(keys=PHASES+["label"], spatial_size=patch_shape, method="end"),
            RandCropByPosNegLabeld(keys=PHASES+["label"], label_key="label",  
                                   pos=1, neg=1, spatial_size=patch_shape, num_samples=num_samples),
            ConcatItemsd(keys=PHASES, name="volume", dim=0),
            EnsureTyped(keys=["volume", "label"]),
        ]
    )

    train_transforms = Compose(
        [
            LoadImaged(keys=PHASES+["label"], image_only=False),
            EnsureChannelFirstd(keys=PHASES+["label"]),
            NormalizeIntensityd(keys=PHASES),
            Spacingd(keys=PHASES, pixdim=(1.0, 1.0, 2.8)),
            SpatialPadd(keys=PHASES+["label"], spatial_size=patch_shape, method="end"),
            RandCropByPosNegLabeld(keys=PHASES+["label"], label_key="label",  
                                   pos=1, neg=1, spatial_size=patch_shape, num_samples=num_samples),
            RandRotated(keys=PHASES+["label"], prob=1, range_x=0.0, range_y=0.0, range_z=180.0, 
                        mode="nearest"),
            RandZoomd(keys=PHASES+["label"], prob=1, min_zoom=0.7, max_zoom=1.4, keep_size=True, padding_mode="constant",
                      mode=("trilinear", "trilinear", "trilinear", "nearest"), align_corners=(True, True, True, None)),
            RandGaussianNoised(keys=PHASES, prob=0.15, mean=0.0, std=0.1),
            RandGaussianSmoothd(keys=PHASES[0], prob=0.1, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
            RandGaussianSmoothd(keys=PHASES[1], prob=0.1, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
            RandGaussianSmoothd(keys=PHASES[2], prob=0.1, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
            RandScaleIntensityd(keys=PHASES, prob=0.15, factors=0.3),
            RandAdjustContrastd(keys=PHASES, prob=0.15, gamma=(0.7, 1.5)),
            RandFlipd(keys=PHASES+["label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=PHASES+["label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=PHASES+["label"], spatial_axis=2, prob=0.5),
            ConcatItemsd(keys=PHASES, name="volume", dim=0),
            EnsureTyped(keys=["volume", "label"]),
        ]
    )
    eval_transforms = Compose(
        [
            LoadImaged(keys=PHASES+["label"], image_only=False),
            EnsureChannelFirstd(keys=PHASES+["label"]),
            NormalizeIntensityd(keys=PHASES),
            Spacingd(keys=PHASES, pixdim=(1.0, 1.0, 2.8)),
            SpatialPadd(keys=PHASES+["label"], spatial_size=patch_shape, method="end"),
            ConcatItemsd(keys=PHASES, name="volume", dim=0),
            EnsureTyped(keys=["volume", "label"]),
        ]
    )

    # define dataset, data loader
    check_ds = Dataset(data=train_files, transform=check_transforms)
    # check images for network training
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["volume"].shape, check_data["label"].shape)

    # create a training data loader
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5)
    # use RandCropByPosNegLabeld to generate (batch_size x num_samples) images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available()
        )
    # create a validation data loader
    eval_ds = CacheDataset(data=eval_files, transform=eval_transforms, cache_rate=0.5)
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
        #upsample_kernel_size = upsample strides = strides[1:]
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
    # print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_learning_rate, nesterov=True, momentum=0.99)
    Optimizer_metadata = {}
    for ind, param_group in enumerate(optimizer.param_groups):
        optim_meta_keys = list(param_group.keys())
        Optimizer_metadata[f'param_group_{ind}'] = {key: value for (key, value) in param_group.items() if 'params' not in key}
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count:", pytorch_total_params)
    
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
    loss_type = "DiceCELoss"
    # start a typical PyTorch training
    eval_interval = 10
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    loss_metric = LossMetric(loss_fn=loss_function)
    
    # initialize a new Aim Run
    patch_shape_str = str(patch_shape[0])+"x"+str(patch_shape[1])+"x"+str(patch_shape[2])
    if continue_training:
        aim_run_hash = sys.argv[2]
        aim_run = aim.Run(run_hash=aim_run_hash, force_resume=True)
    else:
        aim_run = aim.Run(experiment="nnUnet"+"_patch_"+patch_shape_str+"_lr_"+str(initial_learning_rate)+"_SGD_"+loss_type+"_deep_superv_"+str(n_deep_supervision)+"_fold_"+training_fold)
        # log hyperparameters
        aim_run['batch_size'] = batch_size
        aim_run['patch_shape'] = patch_shape
        aim_run['num_samples'] = num_samples
        aim_run['initial_learning_rate'] = initial_learning_rate
        aim_run['deep_supervision'] = w_deep_supervision
        # log model metadata
        aim_run['DynUNet_meatdata'] = DynUNet_metadata
        # log optimizer metadata
        aim_run['Optimizer_metadata'] = Optimizer_metadata
        
    if continue_training:
        if not os.path.exists(pretrained_model):
            print('try to continue, but no model found!')
            return EOFError
        else:
            checkpoint = torch.load(pretrained_model, map_location=device)
            checkpoint_epoch = checkpoint['epoch']
            print('continue from epoch {}, loading latest model...'.format(checkpoint_epoch))
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        checkpoint_epoch = 0
        
    for epoch in range(checkpoint_epoch, n_epochs):   
        model.train()
        epoch_loss = 0
        step = 0
        adjust_learning_rate(optimizer, epoch, initial_learning_rate, n_epochs)
        
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        loop.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')

        for batch_data in loop:
            step += 1
            x, y = batch_data["volume"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(x)
            if n_deep_supervision > 0:
                loss = 0
                outputs = outputs.unbind(dim=1)
                for i in range(len(outputs)):
                    loss += loss_function(outputs[i], y) * w_deep_supervision[i] / sum(w_deep_supervision)
            else:
                loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            # track batch loss metric
            aim_run.track(loss.item(), name="batch_loss", context={'type':loss_type})
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        # track epoch loss metric
        aim_run.track(epoch_loss, name="epoch_loss", context={'type':loss_type}, epoch=epoch+1)

        if (epoch + 1) % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                eval_epoch_loss = 0
                eval_step = 0
                eval_images = None
                eval_labels = None
                eval_outputs = None
                eval_loop = tqdm(eval_loader, total=len(eval_loader), leave=True)
                eval_loop.set_description(f'Valid [{epoch + 1}/{n_epochs}]')
                for eval_data in eval_loop:
                    eval_step += 1
                    eval_images, eval_labels = eval_data["volume"].to(device), eval_data["label"].to(device)
                    roi_size = patch_shape  
                    sw_batch_size = num_samples
                    eval_outputs = sliding_window_inference(eval_images, roi_size, sw_batch_size, model, overlap=0.5)                  
                    
                    eval_outputs = [post_pred(i) for i in decollate_batch(eval_outputs)]
                    loss_metric(y_pred=eval_outputs, y=eval_labels)
                    eval_labels = [post_label(i) for i in decollate_batch(eval_labels)]
                    dice_metric(y_pred=eval_outputs, y=eval_labels)                    
                
                eval_epoch_loss = loss_metric.aggregate().item()
                loop.set_postfix(loss=eval_epoch_loss)
                loss_metric.reset()
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # track val metric
                aim_run.track(metric, name="val_metric", context={'type':"Dice"}, epoch=epoch+1)
                # reset the status for next validation round
                dice_metric.reset()
                
                metric_values.append(metric)

                torch.save(
                    {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),        
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': eval_epoch_loss,
                    },
                    os.path.join(model_track_dir, 'model_latest.pth'))
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    print(
                        "val: epoch: {} mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            epoch + 1, metric, best_metric, best_metric_epoch
                        )
                    )
                    torch.save(
                        {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),        
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': eval_epoch_loss
                        },
                        os.path.join(model_track_dir, 'model_best.pth'))
                message1 = f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                message2 = f"\nbest mean dice: {best_metric:.4f} "
                message3 = f"at epoch: {best_metric_epoch}"
    
                aim_run.track(aim.Text(message1 +"\n" + message2 + message3), name='epoch_summary', epoch=epoch + 1) 

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    # finalize Aim Run
    aim_run.close()


if __name__ == "__main__":
    main()
