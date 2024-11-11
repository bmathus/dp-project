import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from scipy.ndimage import zoom
from medpy import metric

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class KDLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = (
            F.kl_div(F.log_softmax(out_s / self.T, dim=1),F.softmax(out_t / self.T, dim=1), reduction="batchmean") # , reduction="batchmean"
            * self.T
            * self.T
        )
        return loss


def entropy_loss(p, C=4):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

def inference_msdnet(model,input):
    output = model(input)
    return output[0][0]

def inference_urpc(model,input):
    output, _, _, _ = model(input)
    return output

def inference_mtnet(model,input):
    output = model(input)
    if len(output)>1:
        output = output[0]
    return output

def test_single_volume_ds(image, label, model,device,cfg):
    if cfg.network == "urpc":
        inference = inference_urpc
    elif cfg.network == "msdnet":
        inference = inference_msdnet
    else:
        inference = inference_mtnet

    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    #(1, 10, 256, 256) squeeze-> (10,256,256) -> detach (gradient not longer computed for tensor)
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]): # iterate over channel index eg. 0 until 9
        slice = image[ind, :, :] # (256, 224)
        x, y = slice.shape[0], slice.shape[1] # 256, 224
        slice = zoom(slice, (cfg.patch_size / x, cfg.patch_size / y), order=0) # 1, 1.14 -> after zoom (256,256)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
        model.eval()
        with torch.no_grad():
            output = inference(model,input)
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / cfg.patch_size, y / cfg.patch_size), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, cfg.num_classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0