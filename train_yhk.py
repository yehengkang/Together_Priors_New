import os
import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from nets_yhk.yolo import YoloBody
from functools import partial
from nets_yhk.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (get_classes, worker_init_fn)
from utils.utils_fit_yhk import fit_one_epoch


# 相较于原文件主要改的是 DDP 模式
if __name__ == "__main__":
    Cuda            = True
    classes_path    = 'model_data/rtts_classes.txt'
    # model_path      = 'model_data/yolox_s.pth'                 # Pretrained weights for better performance (COCO or VOC）
    model_path      = 'pretrained/yolox_s_Megvii.pth'            # No pretrained weights
    input_shape     = [640, 640]
    phi             = 's'
    mosaic              = False

    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 8

    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 8

    Freeze_Train        = False
    

    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01

    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4

    lr_decay_type       = "cos"

    save_period         = 10

    num_workers         = 4

    # train_annotation_path   = '2007_train_fog.txt'
    # val_annotation_path     = '2007_val_fog.txt'
    # clear_annotation_path = '2007_train.txt'
    # val_clear_annotation_path = '2007_val.txt'
    train_annotation_path   = 'datasets/yhk_train_fog.txt'
    val_annotation_path     = 'datasets/yhk_val_fog.txt'
    clear_annotation_path = 'datasets/yhk_train.txt'
    val_clear_annotation_path = 'datasets/yhk_val.txt'
    
    distributed = True
    #---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    #---------------------------------------------------------------------#
    sync_bn         = False
    ngpus_per_node  = torch.cuda.device_count()
    seed            = 11
    if distributed:
        # import ipdb; ipdb.set_trace()
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        
        torch.cuda.set_device(local_rank)
        
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    class_names, num_classes = get_classes(classes_path)

    model = YoloBody(num_classes, phi)
    weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        # pretrained_dict = torch.load(model_path, map_location = device)
        ckpt = torch.load(model_path, map_location=f"cuda:{local_rank}")
        if "model" in ckpt:
            pretrained_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            pretrained_dict = ckpt["state_dict"]
        else:
            pretrained_dict = ckpt
            
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    yolo_loss    = YOLOLoss(num_classes)
    # loss_history = LossHistory("logs/", model, input_shape=input_shape)
    if local_rank == 0:
        loss_history = LossHistory("logs/", model, input_shape=input_shape)
        # 这里是把训练文件复制到生成权重的路径里面，以便看训练条件
        shutil.copy("train_yhk.py", loss_history.log_dir)
    else:
        loss_history    = None

    model = model.to(device)
    if sync_bn and ngpus_per_node > 1 and distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")
    model_train = model.train()
    
    if Cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            # model_train = model_train.cuda(local_rank)
            # model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
            model_train = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    # import ipdb; ipdb.set_trace()
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    with open(clear_annotation_path, encoding='utf-8') as f:
        clear_lines = f.readlines()
    with open(val_clear_annotation_path, encoding='utf-8') as f:
        val_clear_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs         = 64
        Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
        Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)

        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset error!")

        train_dataset   = YoloDataset(train_lines, clear_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, mosaic=mosaic, train = True)
        val_dataset     = YoloDataset(val_lines, val_clear_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, mosaic=False, train = False)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True
        
        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs         = 64
                Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
                Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset error！")
                
                if distributed:
                    batch_size = batch_size // ngpus_per_node
                    
                gen     = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch
            
            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, save_period, local_rank)

            if distributed:
                dist.barrier()