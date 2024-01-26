from tqdm import tqdm
import torch
import math
from torch import combinations, nn,optim
from torch.utils.data import DataLoader
from dataset import VideoDataset
from torch.nn.parallel import DataParallel
#from models import seed_everything
import models
import hydra
import numpy as np
from utils import *
import wandb
from torchsummary import summary
#from torchinfo import summary

#seed_everything()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('training on {}'.format(device))

Epoches = 100
lr = 0.0001


@hydra.main(config_path="config", config_name="config",version_base=None)
def train_model(args):
    model = models.RSTRM(args)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    class_criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    #reduction 参数的值是 "none" 时，损失函数的输出形状与输入数据的形状相同。当 reduction 参数的值是 "mean" 或 "sum" 时，损失函数的输出形状是一个标量。
    #regress_criterion = nn.MSELoss(reduction='sum')  # sum better than mean
    regress_criterion = nn.SmoothL1Loss(reduction='sum')
    #optimizer = optim.SGD(model.parameters(),lr=lr,weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.1)  # the scheduler divides the lr by 10 every 5 epochs

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Trainable Params: ", params)

    ## 级联损失 or 连接损失

    #model = nn.DataParallel(model, device_ids=[0,1])
    model.to(device)

    class_criterion.to(device)
    regress_criterion.to(device)

    if args.wandb:
        wandb.init(project=f"MTATF_{args.dataset}",name=f"MTATF_alld_{args.dataset_len}len_{args.shot}shot_{args.trans_linear_in_dim}dim_{args.c}c_{int(1-args.c)}r_{lr}lr_{Epoches}epoches")
        wandb.watch(model,regress_criterion,log="all",log_freq=10)
        wandb.log({'params': params})

    train_dataloader = DataLoader(VideoDataset(args), batch_size=1, shuffle=False, num_workers=2)

    mean_acc = []
    for epoch in range(Epoches):
        # reset the running loss and corrects
        running_loss = 0.0
        class_corrects = 0.0
        iou1_corrects = 0.0
        iou3_corrects = 0.0
        iou5_corrects = 0.0
        iou6_corrects = 0.0
        iou7_corrects = 0.0
        iou8_corrects = 0.0
        iou9_corrects = 0.0

        model.train()
        for inputs in tqdm(train_dataloader):
                # move inputs and labels to the device the training is taking place on
                support_feature = inputs['sf'][0].cuda()
                query_feature = inputs['qf'][0].cuda()
                class_label = inputs['vc'].cuda()
                segment_label = inputs['qsl'][0].cuda()
                #vt = inputs['vt']

                optimizer.zero_grad()

                logical,reg,_,_ = model(query_feature,support_feature)

                preds = nn.Softmax(dim=-1)(logical).argmax(dim=-1)

                start = nn.Softmax(dim=-1)(reg[0])
                end = nn.Softmax(dim=-1)(reg[1])
                reg = torch.stack((start,end),dim=0)
                reg = torch.tensor(postprocess(reg)).cuda()
                
                ious = []
                # compute the max iou between the predict and ground truth
                for i in range(len(reg)):
                    iou = iou_conculate(0.5,segment_label,reg[i],True)
                    ious.append(iou)
                    # get the max iou and its reg
                    if iou >= max(ious):
                        max_iou = iou
                        max_reg = reg[i]

                class_loss = class_criterion(logical.unsqueeze(0), class_label.long())
                reg_loss = regress_criterion(max_reg, segment_label.float())

                # add weight(param) to class_loss and reg_loss?????
                loss = args.c*class_loss + (1-args.c)*reg_loss
                # 512 37:78 , 73:77,79 , 55:73
                # 2048 55:75.5/76.9
                # reg = torch.round(reg)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                class_corrects += torch.sum(preds == class_label.data)
                iou1_corrects += np.sum(iou_conculate(0.1,segment_label,max_reg))
                iou3_corrects += np.sum(iou_conculate(0.3,segment_label,max_reg))
                iou5_corrects += np.sum(iou_conculate(0.5,segment_label,max_reg))
                iou6_corrects += np.sum(iou_conculate(0.6,segment_label,max_reg))
                iou7_corrects += np.sum(iou_conculate(0.7,segment_label,max_reg))
                iou8_corrects += np.sum(iou_conculate(0.8,segment_label,max_reg))
                iou9_corrects += np.sum(iou_conculate(0.9,segment_label,max_reg))

        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = class_corrects / len(train_dataloader.dataset)
        epoch_iou1reg = iou1_corrects / len(train_dataloader.dataset)
        epoch_iou3reg = iou3_corrects / len(train_dataloader.dataset)
        epoch_iou5reg = iou5_corrects / len(train_dataloader.dataset)
        epoch_iou6reg = iou6_corrects / len(train_dataloader.dataset)
        epoch_iou7reg = iou7_corrects / len(train_dataloader.dataset)
        epoch_iou8reg = iou8_corrects / len(train_dataloader.dataset)
        epoch_iou9reg = iou9_corrects / len(train_dataloader.dataset)
        reg_mean = np.mean([epoch_iou5reg,epoch_iou6reg,epoch_iou7reg,epoch_iou8reg,epoch_iou9reg])

        if args.wandb:
            wandb.log({"class_loss":epoch_loss,"class_acc":epoch_acc,"iou1":epoch_iou1reg,"iou3":epoch_iou3reg,"iou5":epoch_iou5reg,"iou6":epoch_iou6reg,"iou7":epoch_iou7reg,"iou8":epoch_iou8reg,"iou9":epoch_iou9reg,"reg_mean":reg_mean})

        if args.save_model:
            if len(mean_acc) == 0 or reg_mean > max(mean_acc):
                torch.save(model.state_dict(), 'best/2048/{}class{}reg{}.pth'.format(epoch,np.round(float(epoch_acc),4),np.round(reg_mean,4)))
                mean_acc.append(reg_mean)

        # if epoch+1 % 50 == 0 or epoch_acc >= 0.85:
        #     torch.save(model.state_dict(), '{}acc{}.pth'.format(epoch,np.round(float(epoch_acc),4)))
        print("[{}] Epoch: {}/{} Loss: {} Acc: {}, 0.1:{}, 0.3:{}, 0.5:{}, 0.6:{},0.7:{}, 0.8:{}, 0.9:{}, mean:{}".format('training', epoch+1, Epoches, epoch_loss, epoch_acc, epoch_iou1reg,epoch_iou3reg,epoch_iou5reg,epoch_iou6reg,epoch_iou7reg,epoch_iou8reg,epoch_iou9reg,reg_mean))
        scheduler.step()


# todo: how to conculate loss???  adjust model, input feature length to decoder, encoder(support feature self-attention), decoder(query and support feature cross-attention), 
# multi-task loss?????
# done: (remove < 30 fps feature),  lr , change len of dataloader, input trimmed support feature,
if __name__ == "__main__":
    #testing()
    train_model()
    #os.system('/root/upload.sh')