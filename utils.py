import torch
import math
import numpy as np
    
# compute iou
# def iou_conculate(thresh,gt,preds):
#     out = torch.floor(torch.max(preds[1],gt[1]) - torch.min(preds[0],gt[0]))
#     inter = torch.ceil(torch.min(preds[1],gt[1]) - torch.max(preds[0],gt[0]))
#     if inter/out >= thresh:
#         return 1
#     else:
#         return 0

def spm(iou, mode='linear', sigma=0.3):
    # score penalty mechanism (soft-nms)

    if mode == 'linear':
        return 1 - iou
    elif mode == 'gaussian':
        return math.e ** (- (iou ** 2) / sigma)
    else:
        raise NotImplementedError

def iou_conculate(thresh,gt,preds,compute_only = False):
    out = torch.floor(torch.max(preds[1],gt[1]) - torch.min(preds[0],gt[0]))
    inter = torch.max(torch.tensor(0),torch.ceil(torch.min(preds[1],gt[1]) - torch.max(preds[0],gt[0])))
    if compute_only:
        return inter/out
    else:
        if inter/out >= thresh:
            return 1
        else:
            return 0
    
def np_iou_conculate(thresh,gt,preds,compute_only = False):
    out = np.floor(max(preds[1],gt[1]) - min(preds[0],gt[0]))
    inter = max(0,np.ceil(min(preds[1],gt[1]) - max(preds[0],gt[0])))
    if compute_only:
        return inter/out
    if inter/out >= thresh:
        return 1
    else:
        return 0


def postprocess(ss,se):
    # preds [2,T], mean the probability of the start and end points, compute top 50 combinations and use NMS to get the final 20 results
    
    # Get the probabilities of start and end points
    start_probs = ss
    end_probs = se
    
    # Compute the scores as the product of start and end probabilities
    scores = torch.outer(start_probs, end_probs).cpu()
    
    # Get the top 50 combinations based on scores and keep the score and index
    # get 1d indexs of top scores from dim[1156]  [474,449,......]
    len_scores = len(scores.flatten())
    if len_scores<50:
        topk_scores,top_indices = torch.topk(scores.flatten(), k=len_scores)
    else:
        topk_scores,top_indices = torch.topk(scores.flatten(), k=50)
    # get the indexs of the max element in a multi-dimension array dim[34,34], return combination indexs [13,13,....] [32,7.....] -> 13*34+32,13*34+7
    top_combinations = np.unravel_index(top_indices.cpu().numpy(), scores.shape)
    
    #combinations = np.stack((top_combinations[0],top_combinations[1],topk_scores.detach().numpy()))

    # Apply non-maximum suppression (NMS) to get the final results
    final_results = []
    for i in range(len(top_combinations[0])):
        start = top_combinations[0][i]
        end = top_combinations[1][i]
        score = topk_scores[i]
        # Check if the end is greater than the start
        if end > start:
            overlap = False
            for result in final_results:
                if np_iou_conculate(0.8, result, (start, end)):
                    overlap = True
                    break
            if not overlap:
                final_results.append((start, end))
            if len(final_results) == 20:
                break
    
    return final_results



# # Apply soft non-maximum suppression (NMS) to get the best results
def NMS(combination_scores, iou_thre, soft=False, soft_thre=0.001):
    # Non-Maximum Suppression [3,50]

    lists = sorted(combination_scores, key=lambda x: x[2], reverse=True)
    keep = []

    while lists:
        m = lists.pop(0)
        keep.append(m)
        for i, pred in enumerate(lists):
            _iou = np_iou_conculate(1,m, pred, compute_only=True)
            if _iou >= iou_thre:
                if soft:
                    pred[4] *= spm(_iou, mode='gaussian', sigma=0.3)
                    keep.append(lists.pop(i))
                else:
                    lists.pop(i)

    if soft:
        keep = list(filter(lambda x: x[4] >= soft_thre, keep))
        keep = sorted(keep, key=lambda x: x[4], reverse=True)

    return keep