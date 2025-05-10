import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from scipy.ndimage import zoom
from medpy import metric
from config.run_config import Config


def inference_dbpnet(model,input):
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

def test_single_volume_ds(image, label, model,device,cfg: Config):
    if cfg.network == "urpc":
        inference = inference_urpc
    elif cfg.network == "dbpnet":
        inference = inference_dbpnet
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