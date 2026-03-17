from __future__ import division, print_function

import numpy as np
import torch

from .utils.shap_utils import *


def gen_concept_masks(gen_model, target_img):
    return gen_model.generate(target_img)


### model: the target endtoend model you wish to explain
### img_numpy: image in numpy version, the same file you pass to the sam
### image_class: the class id you wish to explain
### concept_masks: concept_masks gen by sam
### fc: the fully-connect layer in the target endtoend model
### feat_exp: the feature extrator in the target endtoend model
### image_norm: transform the img_numpy into torch version with the target model normlize


def samshap(
    model,
    img_numpy,
    image_class,
    concept_masks,
    fc,
    feat_exp,
    image_norm=None,
    lr=0.008,
):
    feat, probs, losses, net, bin_x_torch = learn_PIE(
        feat_exp,
        model,
        concept_masks,
        img_numpy,
        image_class,
        fc,
        lr=lr,
        epochs=100,
        image_norm=image_norm,
    )  ### learning the PIE module
    feat_num = len(concept_masks)
    shap_val = []
    mc = 50000

    for i in range(feat_num):
        bin_x_tmp = np.random.binomial(
            1, 0.5, size=(mc, feat_num)
        )  ### mc shapley computing
        bin_x_tmp_sec = bin_x_tmp.copy()

        bin_x_tmp[:, i] = 1
        bin_x_tmp_sec[:, i] = 0

        bin_x_tmp = torch.from_numpy(bin_x_tmp).type(torch.float)
        bin_x_tmp_sec = torch.from_numpy(bin_x_tmp_sec).type(torch.float)

        pre_shap = (
            (
                feat_prob(fc, net.forward_feat(bin_x_tmp), image_class)
                - feat_prob(fc, net.forward_feat(bin_x_tmp_sec), image_class)
            )
            .detach()
            .cpu()
            .numpy()
        )

        shap_val.append(pre_shap.sum() / mc)
    ans = shap_val.index(max(shap_val))
    shap_list = shap_val

    shap_list = np.array(shap_list)
    shap_arg = np.argsort(-shap_list)
    auc_mask = np.expand_dims(
        np.array(
            [concept_masks[shap_arg[: i + 1]].sum(0) for i in range(len(shap_arg))]
        ).astype(bool),
        3,
    )

    return auc_mask, shap_list
