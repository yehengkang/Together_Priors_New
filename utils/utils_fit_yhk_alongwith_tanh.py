import torch
from tqdm import tqdm
import torch.nn as nn

from utils.psnr import block_psnr
from utils.utils import get_lr
        

# 相较于原文件主要改的是 DDP 模式，跟train_yhk搭配        
def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, save_period, local_rank=0):
    loss        = 0
    val_loss    = 0
    Dehazy_loss = 0
    criterion = nn.MSELoss()

    # # ImageNet标准化参数，用于将GT转换到Tanh范围[-1, 1]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    if cuda:
        mean = mean.cuda(local_rank)
        std = std.cuda(local_rank)

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    
    model_train.train()
    print('Start Train')
    
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
        # if iteration >= 5:
            break

        images, targets, clearimgs, = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images  = torch.from_numpy(images).type(torch.FloatTensor).cuda(local_rank)
                targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda(local_rank) for ann in targets]
                clearimgs = torch.from_numpy(clearimgs).type(torch.FloatTensor).cuda(local_rank)
                # import ipdb; ipdb.set_trace()
                # hazy_and_clear = torch.cat([images, clearimgs], dim=0).cuda()
            else:
                images  = torch.from_numpy(images).type(torch.FloatTensor)
                targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                clearimgs = torch.from_numpy(clearimgs).type(torch.FloatTensor)

        optimizer.zero_grad()

        # 这个地方输入的清晰图像在前向传播过程中并没有用到，且按照split划分的时候会出现进一步地错误，所以注释掉，直接输入原始带雾图像
        # outputs = model_train(hazy_and_clear)    # 提示一下：这里的 model = YoloBody(num_classes, phi)
        
        # 这个是在深层进行psnr的计算，初步测试，效果不好，所以注释掉
        # with torch.no_grad():
        #     clear_features = model_train(clearimgs, mode="clear")
        # outputs = model_train(images, clear_features)
        
        # #####这个是在输入端进行psnr的计算
        # psnr_map = block_psnr(images, clearimgs)
        # outputs = model_train(images, psnr_map)
        
        # 这个是原始输入
        outputs = model_train(images)

        loss_value_all = 0
        
        loss_value     = yolo_loss(outputs[0], targets)

        # 将clearimgs从ImageNet标准化范围转换到Tanh范围[-1, 1]
        # 步骤: ImageNet标准化 -> [0, 1] -> [-1, 1]
        clearimgs_tanh = (clearimgs * std + mean) * 2.0 - 1.0
        loss_dehazy    = criterion(outputs[1], clearimgs_tanh)
        # loss_dehazy    = criterion(outputs[1], clearimgs)
        
        loss_value     = 0.5 * loss_value + 0.5 * loss_dehazy

        loss_value.backward()
        optimizer.step()

        loss += loss_value.item()
        Dehazy_loss += loss_dehazy.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1),
                                'Dehazy_loss': Dehazy_loss / (iteration + 1),
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    val_Dehazy_loss = 0
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
        # if iteration >= 3:
            break
        # images, targets = batch[0], batch[1]
        images, targets, clearimgs = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images  = torch.from_numpy(images).type(torch.FloatTensor).cuda(local_rank)
                targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda(local_rank) for ann in targets]
                clearimgs = torch.from_numpy(clearimgs).type(torch.FloatTensor).cuda(local_rank)
            else:
                images  = torch.from_numpy(images).type(torch.FloatTensor)
                targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                clearimgs = torch.from_numpy(clearimgs).type(torch.FloatTensor)

            # optimizer.zero_grad()

            outputs         = model_train(images)
            loss_value      = yolo_loss(outputs[0], targets)
            
            # 验证集也计算去雾损失
            clearimgs_tanh = (clearimgs * std + mean) * 2.0 - 1.0
            val_loss_dehazy = criterion(outputs[1], clearimgs_tanh)
            loss_value = 0.5 * loss_value + 0.5 * val_loss_dehazy

        val_loss += loss_value.item()
        val_Dehazy_loss += val_loss_dehazy.item()
        if local_rank == 0:
            # pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            # pbar.update(1)
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'val_Dehazy_loss': val_Dehazy_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
    
        # import ipdb; ipdb.set_trace()
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val, Dehazy_loss / epoch_step)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        # import ipdb; ipdb.set_trace()
        save_dir = loss_history.log_dir
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), save_dir+'/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
            # torch.save(model.state_dict(), '/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
