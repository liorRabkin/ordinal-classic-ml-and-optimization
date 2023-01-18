import os, sys, pdb
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import itertools
# import deepdish as dd
# import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# My functions
from ordinal_classic_ml.utils.eval_util import ordinal_mse
from ordinal_classic_ml.utils.layer_util import extract_gap_layer, extract_vgg_fea_layer
from ordinal_classic_ml.utils.layer_util import gen_cam_visual
# from ordinal_classic_ml.utils.grad_cam import GradCam, show_cam_on_image
import ordinal_classic_ml.utils.utils_functions as uf


def eval_test(args, model, dset_loaders, dset_size, phase):
    labels_all = [] * dset_size[phase]
    preds_all = [] * dset_size[phase]
    outputs_all = [] * dset_size[phase]

    for data in dset_loaders[phase]:
        inputs, labels, _ = data
        inputs = Variable(inputs.cuda())
        with torch.no_grad():
            outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        labels_np = labels.numpy()
        labels = labels_np.tolist()
        labels_all.extend(labels)
        preds_cpu = preds.cpu()
        preds_np = preds_cpu.numpy()
        preds = preds_np.tolist()
        preds_all.extend(preds)

        softmax = nn.Softmax(dim=1)
        outputs_list = softmax(outputs).detach().cpu().numpy().tolist()
        outputs_all.extend(outputs_list)

    conf_matrix = confusion_matrix(labels_all, preds_all)
    acc = 1.0 * np.trace(conf_matrix) / np.sum(conf_matrix)
    mse = ordinal_mse(conf_matrix)
    mae = ordinal_mse(conf_matrix, poly_num=1)
    indices_dict = uf.indices(5, np.array(labels_all), np.array(preds_all), args.cost_matrix)
    cost = np.mean(indices_dict["cost"])

    return acc, mse, outputs_all, labels_all
