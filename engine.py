import time
import numpy as np
import torch
from torch.autograd import Variable
from utils import AverageMeter
from eval_metrics import  eval_regdb, eval_sysu


def trainer(args, epoch, main_net, adjust_learning_rate, optimizer, trainloader, criterion, writer=None, print_freq=50):
    current_lr = adjust_learning_rate(args, optimizer, epoch)

    total_loss = AverageMeter()
    id_loss_rgb = AverageMeter()
    id_loss_ir = AverageMeter()
    tri_loss_rgb = AverageMeter()
    tri_loss_ir = AverageMeter()
    dis_loss = AverageMeter()
    pa_loss = AverageMeter()
    batch_time = AverageMeter()

    correct_tri_rgb = 0
    correct_tri_ir = 0
    pre_rgb = 0  # it is meaningful only in the case of semi supervised setting
    pre_ir = 0  # it is meaningful only in the case of semi supervised setting
    pre_rgb_ir = 0  # it is meaningful only in the case of semi supervised setting, whether labels of selected samples per batch are equal
    num_rgb = 0
    num_ir = 0

    main_net.train()  # switch to train mode
    end = time.time()

    for batch_id, (input_rgb, input_ir, label_rgb, label_ir) in enumerate(trainloader):
        # label_ir is only used to calculate the prediction accuracy of pseudo infrared labels on semi-supervised setting
        # label_ir is meaningless on unsupervised setting
        # for supervised setting, we change "label_rgb" of "loss_id_ir" and "loss_tri_ir" into "label_ir"

        label_rgb = label_rgb.cuda()
        label_ir = label_ir.cuda()
        input_rgb = input_rgb.cuda()
        input_ir = input_ir.cuda()

        feat, output_cls, output_dis = main_net(input_rgb, input_ir, modal=0, train_set=True)

        loss_id_rgb = criterion[0](output_cls[:input_rgb.size(0)], label_rgb)
        loss_tri_rgb, correct_tri_batch_rgb = criterion[1](feat[:input_rgb.size(0)], label_rgb)

        if args.setting == "semi-supervised" or args.setting == "unsupervised":
            loss_id_ir = criterion[0](output_cls[input_rgb.size(0):], label_rgb)
            loss_tri_ir, correct_tri_batch_ir = criterion[1](feat[input_rgb.size(0):], label_rgb)
        elif args.setting == "supervised":
            loss_id_ir = criterion[0](output_cls[input_rgb.size(0):], label_ir)
            loss_tri_ir, correct_tri_batch_ir = criterion[1](feat[input_rgb.size(0):], label_ir)

        dis_label = torch.cat((torch.ones(input_rgb.size(0)), torch.zeros(input_ir.size(0))), dim=0).cuda()
        loss_dis = criterion[2](output_dis.view(-1), dis_label)

        loss_pa, sim_rgbtoir, sim_irtorgb = criterion[3](output_cls[:input_rgb.size(0)], output_cls[input_rgb.size(0):])

        # loss = loss_id_rgb + loss_tri_rgb + 0.1 * loss_id_ir + 0.5 * loss_tri_ir + loss_dis + loss_pa
        loss = loss_id_rgb + loss_tri_rgb + 0.1 * loss_id_ir + 0.5 * loss_tri_ir + loss_pa

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct_tri_rgb += correct_tri_batch_rgb
        correct_tri_ir += correct_tri_batch_ir
        _, pre_label = output_cls.max(1)
        pre_batch_rgb = (pre_label[:input_rgb.size(0)].eq(label_rgb).sum().item())
        pre_batch_ir = (pre_label[input_rgb.size(0):].eq(label_ir).sum().item())
        pre_batch_rgb_ir = (label_rgb.eq(label_ir).sum().item())
        pre_rgb += pre_batch_rgb
        pre_ir += pre_batch_ir
        pre_rgb_ir += pre_batch_rgb_ir
        num_rgb += input_rgb.size(0)
        num_ir += input_ir.size(0)
        assert num_rgb == num_ir

        total_loss.update(loss.item(), input_rgb.size(0) + input_ir.size(0))
        id_loss_rgb.update(loss_id_rgb.item(), input_rgb.size(0))
        id_loss_ir.update(loss_id_ir.item(), input_ir.size(0))
        tri_loss_rgb.update(loss_tri_rgb, input_rgb.size(0))
        tri_loss_ir.update(loss_tri_ir, input_ir.size(0))
        dis_loss.update(loss_dis, input_rgb.size(0) + input_ir.size(0))
        pa_loss.update(loss_pa.item(), input_rgb.size(0) + input_ir.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_id % print_freq == 0:
            print("Epoch: [{}][{}/{}] "
                  "Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                  "Lr: {:.6f} "
                  "Coeff: {:.3f} "
                  "Total_Loss: {total_loss.val:.4f}({total_loss.avg:.4f}) "
                  "ID_Loss_RGB: {id_loss_rgb.val:.4f}({id_loss_rgb.avg:.4f}) "
                  "ID_Loss_IR: {id_loss_ir.val:.4f}({id_loss_ir.avg:.4f}) "
                  "Tri_Loss_RGB: {tri_loss_rgb.val:.4f}({tri_loss_rgb.avg:.4f}) "
                  "Tri_Loss_IR: {tri_loss_ir.val:.4f}({tri_loss_ir.avg:.4f}) "
                  "Dis_Loss: {dis_loss.val:.4f}({dis_loss.avg:.4f}) "
                  "Pa_Loss: {pa_loss.val:.4f}({pa_loss.avg:.4f}) "
                  "Tri_RGB_Acc: {:.2f}% "
                  "Tri_IR_Acc: {:.2f}% "
                  "Pre_RGB_Acc: {:.2f}% "
                  "Pre_IR_Acc: {:.2f}% "
                  "Pre_RGB_IR_Acc: {:.2f}% ".format(epoch, batch_id, len(trainloader), current_lr, main_net.adnet.coeff,
                                                100. * correct_tri_rgb / num_rgb,
                                                100. * correct_tri_ir / num_ir,
                                                100. * pre_rgb / num_rgb,
                                                100. * pre_ir / num_ir,
                                                100. * pre_rgb_ir / num_rgb,
                                                batch_time=batch_time,
                                                total_loss=total_loss,
                                                id_loss_rgb=id_loss_rgb,
                                                id_loss_ir=id_loss_ir,
                                                tri_loss_rgb=tri_loss_rgb,
                                                tri_loss_ir=tri_loss_ir,
                                                dis_loss=dis_loss,
                                                pa_loss=pa_loss))

    if writer is not None:
        writer.add_scalar("Lr", current_lr, epoch)
        writer.add_scalar("Coeff", main_net.adnet.coeff, epoch)
        writer.add_scalar("Total_Loss", total_loss.avg, epoch)
        writer.add_scalar("ID_Loss_RGB", id_loss_rgb.avg, epoch)
        writer.add_scalar("ID_Loss_IR", id_loss_ir.avg, epoch)
        writer.add_scalar("Tri_Loss_RGB", tri_loss_rgb.avg, epoch)
        writer.add_scalar("Tri_Loss_IR", tri_loss_ir.avg, epoch)
        writer.add_scalar("Dis_Loss", dis_loss.avg, epoch)
        writer.add_scalar("Pa_Loss", pa_loss.avg, epoch)
        writer.add_scalar("Tri_RGB_Acc", 100. * correct_tri_rgb / num_rgb, epoch)
        writer.add_scalar("Tri_IR_Acc", 100. * correct_tri_ir / num_ir, epoch)
        writer.add_scalar("Pre_RGB_Acc", 100. * pre_rgb / num_rgb, epoch)
        writer.add_scalar("Pre_IR_Acc", 100. * pre_ir / num_ir, epoch)


def tester(args, epoch, main_net, test_mode, gall_label, gall_loader, query_label, query_loader, feat_dim=2048, query_cam=None, gall_cam=None, writer=None):
    # switch to evaluation mode
    main_net.eval()

    print("Extracting Gallery Feature...")
    ngall = len(gall_label)
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = main_net(input, input, modal=test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print("Extracting Time:\t {:.3f}".format(time.time() - start))

    print("Extracting Query Feature...")
    nquery = len(query_label)
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = main_net(input, input, modal=test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print("Extracting Time:\t {:.3f}".format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = -np.matmul(query_feat, np.transpose(gall_feat))
    # evaluation
    if args.dataset == "sysu":
        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
    elif args.dataset == "regdb":
        cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
    print("Evaluation Time:\t {:.3f}".format(time.time() - start))

    if writer is not None:
        writer.add_scalar("Rank1", cmc[0], epoch)
        writer.add_scalar("mAP", mAP, epoch)
        writer.add_scalar("mINP", mINP, epoch)

    return cmc, mAP, mINP











