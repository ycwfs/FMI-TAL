from tqdm import tqdm
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from dataset import VideoDataset
from torch.nn.parallel import DataParallel
from models import seed_everything
import models
import hydra
import numpy as np

#seed_everything()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('training on {}'.format(device))

Epoches = 10000
lr = 0.0001

# compute iou
# def iou_conculate(thresh,gt,preds):
#     out = torch.floor(torch.max(preds[1],gt[1]) - torch.min(preds[0],gt[0]))
#     inter = torch.ceil(torch.min(preds[1],gt[1]) - torch.max(preds[0],gt[0]))
#     if inter/out >= thresh:
#         return 1
#     else:
#         return 0

def iou_conculate(thresh,gt,preds):
    out = torch.max(preds[1],gt[1]) - torch.min(preds[0],gt[0])
    inter = torch.min(preds[1],gt[1]) - torch.max(preds[0],gt[0])
    if inter/out >= thresh:
        return 1
    else:
        return 0

@hydra.main(config_path="config", config_name="config",version_base=None)
def train_model(args):
    model = models.RSTRM(args)

    class_criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    regress_criterion = nn.MSELoss(reduction='sum')  # standard crossentropy loss for classification  sum better than mean
    #optimizer = optim.SGD(model.parameters(),lr=lr,weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.1)  # the scheduler divides the lr by 10 every 5 epochs

    model.to(device)
    class_criterion.to(device)
    regress_criterion.to(device)

    train_dataloader = DataLoader(VideoDataset(args), batch_size=1, shuffle=False, num_workers=2)

    mean_acc = []
    for epoch in range(Epoches):
        # reset the running loss and corrects
        running_loss = 0.0
        class_corrects = 0.0
        iou1_corrects = 0.0
        iou3_corrects = 0.0
        iou5_corrects = 0.0
        iou7_corrects = 0.0
        iou9_corrects = 0.0

        model.train()
        for inputs in tqdm(train_dataloader):
                # move inputs and labels to the device the training is taking place on
                support_feature = inputs['sf'][0].cuda()
                query_feature = inputs['qf'][0].cuda()
                class_label = inputs['vc'].cuda()
                segment_label = inputs['qsl'][0].cuda()

                optimizer.zero_grad()

                logical,reg,_,_ = model(query_feature,support_feature)

                preds = nn.Softmax(dim=-1)(logical).argmax(dim=-1)

                class_loss = class_criterion(logical.unsqueeze(0), class_label.long())
                reg_loss = regress_criterion(reg, segment_label.float())

                loss = class_loss + reg_loss
                # ????????????
                # reg = torch.round(reg)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                class_corrects += torch.sum(preds == class_label.data)
                iou1_corrects += np.sum(iou_conculate(0.1,segment_label,reg))
                iou3_corrects += np.sum(iou_conculate(0.3,segment_label,reg))
                iou5_corrects += np.sum(iou_conculate(0.5,segment_label,reg))
                iou7_corrects += np.sum(iou_conculate(0.7,segment_label,reg))
                iou9_corrects += np.sum(iou_conculate(0.9,segment_label,reg))

        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = class_corrects / len(train_dataloader.dataset)
        epoch_iou1reg = iou1_corrects / len(train_dataloader.dataset)
        epoch_iou3reg = iou3_corrects / len(train_dataloader.dataset)
        epoch_iou5reg = iou5_corrects / len(train_dataloader.dataset)
        epoch_iou7reg = iou7_corrects / len(train_dataloader.dataset)
        epoch_iou9reg = iou9_corrects / len(train_dataloader.dataset)
        reg_mean = np.mean([epoch_iou1reg,epoch_iou3reg,epoch_iou5reg,epoch_iou7reg,epoch_iou9reg])

        # first loop mean_acc is empty,how to solve it?
        if len(mean_acc) == 0 or reg_mean > max(mean_acc):
            torch.save(model.state_dict(), './result/{}class{}reg{}.pth'.format(epoch,np.round(float(epoch_acc),4),np.round(reg_mean,4)))
            mean_acc.append(reg_mean)

        if epoch+1 % 50 == 0 or epoch_acc >= 0.85:
            torch.save(model.state_dict(), '{}acc{}.pth'.format(epoch,np.round(float(epoch_acc),4)))
        print("[{}] Epoch: {}/{} Loss: {} Acc: {}, 0.1:{}, 0.3:{}, 0.5:{}, 0.7:{}, 0.9:{}, mean:{}".format('training', epoch+1, Epoches, epoch_loss, epoch_acc, epoch_iou1reg,epoch_iou3reg,epoch_iou5reg,epoch_iou7reg,epoch_iou9reg,reg_mean))
        scheduler.step()

    torch.save(model.state_dict(), 'result/lastresult.pth')


# todo: how to conculate loss???  adjust model, input feature length to decoder, encoder(support feature self-attention), decoder(query and support feature cross-attention)
# done: (remove < 30 fps feature),  lr , change len of dataloader, input trimmed support feature,
if __name__ == "__main__":
    #testing()
    train_model()
    #os.system('/root/upload.sh')