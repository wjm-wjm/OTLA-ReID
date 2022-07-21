import time
import numpy as np
import torch
import torch.nn as nn
from utils import sort_list_with_unique_index


def cpu_sk_ir_trainloader(args, main_net, trainloader, tIndex, n_class, print_freq=50):
    main_net.train()

    n_ir = len(tIndex)
    P = np.zeros((n_ir, n_class))

    with torch.no_grad():
        for batch_idx, (input_rgb, input_ir, label_rgb, label_ir) in enumerate(trainloader):
            t = time.time()
            input_ir = input_ir.cuda()
            _, p, _ = main_net(input_ir, input_ir, modal=2, train_set=False)
            p_softmax = nn.Softmax(1)(p).cpu().numpy()
            P[batch_idx * args.train_batch_size * args.num_pos:(batch_idx + 1) * args.train_batch_size * args.num_pos, :] = p_softmax

            if batch_idx == 0:
                ir_real_label = label_ir
            else:
                ir_real_label = torch.cat((ir_real_label, label_ir), dim=0)

            if (batch_idx + 1) % print_freq == 0:
                print("Extract predictions: [{}/{}]\t"
                      "Time consuming: {:.3f}\t"
                      .format(batch_idx + 1, len(trainloader), time.time() - t))

    # optimizer label using Sinkhorn-Knopp algorithm
    unique_tIndex_first_idx, unique_tIndex_last_idx, unique_tIndex_num, idx_order, unique_tIndex_list = sort_list_with_unique_index(tIndex)
    unique_tIndex_idx = unique_tIndex_last_idx  # last
    ir_real_label = ir_real_label[unique_tIndex_idx]
    P_ = P[unique_tIndex_idx]
    for i, idx in enumerate(idx_order):
        P_[i] = (P[unique_tIndex_list[idx]].mean(axis=0))
    PS = (P_.T) ** args.lambda_sk

    n_ir_unique = len(np.unique(tIndex))
    alpha = np.ones((n_class, 1)) / n_class  # initial value for alpha
    beta = np.ones((n_ir_unique, 1)) / n_ir_unique  # initial value for beta

    inv_K = 1. / n_class
    inv_N = 1. / n_ir_unique

    err = 1e6
    step = 0
    tt = time.time()
    while err > 1e-1:
        alpha = inv_K / (PS @ beta)  # (KxN) @ (N,1) = K x 1
        beta_new = inv_N / (alpha.T @ PS).T  # ((1,K) @ (KxN)).t() = N x 1
        if step % 10 == 0:
            err = np.nansum(np.abs(beta / beta_new - 1))
        beta = beta_new
        step += 1
    print("Sinkhorn-Knopp   Error: {:.3f}   Total step: {}   Total time: {:.3f}".format(err, step, time.time() - tt))
    PS *= np.squeeze(beta)
    PS = PS.T
    PS *= np.squeeze(alpha)
    PS = PS.T
    argmaxes = np.nanargmax(PS, 0)  # size n_ir
    ir_pseudo_label_op = torch.LongTensor(argmaxes)

    # the max prediction of softmax
    argmaxes_ = np.nanargmax(P_, 1)
    ir_pseudo_label_mp = torch.LongTensor(argmaxes_)

    return ir_pseudo_label_op, ir_pseudo_label_mp, ir_real_label, tIndex[unique_tIndex_idx]