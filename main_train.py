import argparse
import easydict
import sys
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from utils import Logger, set_seed, GenIdx, IdentitySampler, SemiIdentitySampler_randomIR, SemiIdentitySampler_pseudoIR
from data_loader import SYSUData, RegDBData, TestData
from data_manager import process_query_sysu, process_gallery_sysu, process_test_regdb
from model.network import BaseResNet
from loss import TripletLoss, PredictionAlignmentLoss
from optimizer import select_optimizer, adjust_learning_rate
from engine import trainer, tester
from otla_sk import cpu_sk_ir_trainloader


def main_worker(args, args_main):
    ## set start epoch and end epoch
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch

    ## set gpu id and seed id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.benchmark = True  # accelerate the running speed of convolution network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    ## set file
    if not os.path.isdir(args.dataset + "_" + args.setting + "_" + args.file_name):
        os.makedirs(args.dataset + "_" + args.setting + "_" + args.file_name)
    file_name = args.dataset + "_" + args.setting + "_" + args.file_name

    if args.dataset == "sysu":
        data_path = args.dataset_path + "SYSU-MM01/"
        log_path = os.path.join(file_name, args.dataset + "_" + args.log_path)
        vis_log_path = os.path.join(file_name, args.dataset + "_" + args.vis_log_path)
        model_path = os.path.join(file_name, args.dataset + "_" + args.model_path)
        test_mode = [1, 2]
    elif args.dataset == "regdb":
        data_path = args.dataset_path + "RegDB/"
        log_path = os.path.join(file_name, args.dataset + "_" + args.log_path)
        vis_log_path = os.path.join(file_name, args.dataset + "_" + args.vis_log_path)
        model_path = os.path.join(file_name, args.dataset + "_" + args.model_path)
        if args.mode == "thermaltovisible":
            test_mode = [1, 2]
        elif args.mode == "visibletothermal":
            test_mode = [2, 1]

    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(vis_log_path):
        os.makedirs(vis_log_path)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    sys.stdout = Logger(os.path.join(log_path, "log_os.txt"))
    test_os_log = open(os.path.join(log_path, "log_os_test.txt"), "w")

    ## tensorboard
    writer = SummaryWriter(vis_log_path)

    ## load data
    print("==========\nargs_main:{}\n==========".format(args_main))
    print("==========\nargs:{}\n==========".format(args))
    print("==> loading data...")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.5),
    ])
    transform_train_ir = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.5),
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    end = time.time()
    if args.dataset == "sysu":
        # training set
        trainset = SYSUData(args, data_path, transform_train_rgb=transform_train_rgb, transform_train_ir=transform_train_ir)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
        # testing set
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode)
    elif args.dataset == "regdb":
        # training set
        trainset = RegDBData(args, data_path, transform_train_rgb=transform_train_rgb, transform_train_ir=transform_train_ir)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
        # testing set
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modality=args.mode.split("to")[0])
        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modality=args.mode.split("to")[1])

    gallset = TestData(gall_img, gall_label, transform_test=transform_test, img_size=(args.img_w, args.img_h))
    queryset = TestData(query_img, query_label, transform_test=transform_test, img_size=(args.img_w, args.img_h))

    # testing data loader
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

    n_class = len(np.unique(trainset.train_color_label))  # number of classes
    n_rgb = len(trainset.train_color_label)  # number of visible images
    n_ir = len(trainset.train_thermal_label)  # number of infrared images

    print("Dataset {} Statistics:".format(args.dataset))
    print("  ----------------------------")
    print("  subset   | # ids | # images")
    print("  ----------------------------")
    print("  visible  | {:5d} | {:8d}".format(n_class, len(trainset.train_color_label)))
    print("  thermal  | {:5d} | {:8d}".format(n_class, len(trainset.train_thermal_label)))
    print("  ----------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), len(query_label)))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), len(gall_label)))
    print("  ----------------------------")
    print("Data loading time:\t {:.3f}".format(time.time() - end))

    ## build model
    main_net = BaseResNet(pool_dim=args.pool_dim, class_num=n_class, per_add_iters=args.per_add_iters, arch=args.arch)
    main_net.to(device)

    ## resume checkpoints
    if args_main.resume:
        resume_path = args_main.resume_path
        if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path)
            if "epoch" in checkpoint.keys():
                start_epoch = checkpoint["epoch"]
            main_net.load_state_dict(checkpoint["main_net"])
            print("==> Loading checkpoint {} (epoch {})".format(resume_path, start_epoch))
        else:
            print("==> No checkpoint is found at {}".format(resume_path))
    print("start epoch: {}, end epoch: {}".format(start_epoch, end_epoch))

    ## define loss functions
    criterion = []
    criterion_id = nn.CrossEntropyLoss()  # id loss
    criterion.append(criterion_id.to(device))
    criterion_tri = TripletLoss(margin=args.margin)  # triplet loss
    criterion.append(criterion_tri)
    criterion_dis = nn.BCELoss()
    criterion.append(criterion_dis)
    criterion_pa = PredictionAlignmentLoss(lambda_vr=args.lambda_vr, lambda_rv=args.lambda_rv)  # prediction alignment loss
    criterion.append(criterion_pa)

    ## set optimizer
    optimizer = select_optimizer(args, main_net)

    ## start training and testing
    print("==> start training...")
    best_acc = 0
    train_thermal_pseudo_label = np.random.randint(0, n_class, len(trainset.train_thermal_label))
    for epoch in range(start_epoch, end_epoch - start_epoch):
        end = time.time()

        print("==> preparing data loader...")
        if args.setting == "unsupervised" or args.setting == "semi-supervised":
            if epoch == 0:
                sampler = SemiIdentitySampler_randomIR(trainset.train_color_label, train_thermal_pseudo_label, color_pos, args.num_pos, args.train_batch_size, args.dataset_num_size)
            else:
                sampler = SemiIdentitySampler_pseudoIR(trainset.train_color_label, train_thermal_pseudo_label, color_pos, args.num_pos, args.train_batch_size, args.dataset_num_size)
        elif args.setting == "supervised":
            sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.train_batch_size, args.dataset_num_size)

        trainset.cIndex = sampler.index1  # color index
        trainset.tIndex = sampler.index2  # thermal index
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch_size * args.num_pos, sampler=sampler, num_workers=args.workers, drop_last=True)

        ir_pseudo_label_op, ir_pseudo_label_mp, ir_real_label, unique_tIndex_idx = cpu_sk_ir_trainloader(args, main_net, trainloader, sampler.index2, n_class)
        train_thermal_pseudo_label[unique_tIndex_idx] = ir_pseudo_label_op.numpy()
        print("Total number of IR per trainloader: {}, Unique number of IR per trainloader: {}".format(len(sampler.index2), len(unique_tIndex_idx)))
        predict_per_epoch_op = (ir_pseudo_label_op.eq(ir_real_label).sum().item()) / ir_real_label.size(0)
        predict_per_epoch_mp = (ir_pseudo_label_mp.eq(ir_real_label).sum().item()) / ir_real_label.size(0)
        predict_per_epoch_all = (train_thermal_pseudo_label == trainset.train_thermal_label).sum() / len(trainset.train_thermal_label)
        print("Label prediction accuracy, Op: {:.2f}%, Mp: {:.2f}%, All: {:.2f}%".format(predict_per_epoch_op * 100, predict_per_epoch_mp * 100, predict_per_epoch_all * 100))

        # training
        trainer(args, epoch, main_net, adjust_learning_rate, optimizer, trainloader, criterion, writer=writer)
        print("training time per epoch: {:.3f}".format(time.time() - end))

        if epoch % args.eval_epoch == 0:
            if args.dataset == "sysu":
                print("Testing Epoch: {}, Testing mode: {}".format(epoch, args.mode))
                print("Testing Epoch: {}, Testing mode: {}".format(epoch, args.mode), file=test_os_log)
            elif args.dataset == "regdb":
                print("Testing Epoch: {}, Testing mode: {}, Trial: {}".format(epoch, args.mode, args.trial))
                print("Testing Epoch: {}, Testing mode: {}, Trial: {}".format(epoch, args.mode, args.trial), file=test_os_log)

            # start testing
            end = time.time()
            if args.dataset == "sysu":
                cmc, mAP, mINP = tester(args, epoch, main_net, test_mode, gall_label, gall_loader, query_label, query_loader, feat_dim=args.pool_dim, query_cam=query_cam, gall_cam=gall_cam, writer=writer)
            elif args.dataset == "regdb":
                cmc, mAP, mINP = tester(args, epoch, main_net, test_mode, gall_label, gall_loader, query_label, query_loader, feat_dim=args.pool_dim, writer=writer)
            print("Testing time per epoch: {:.3f}".format(time.time() - end))

            # save model
            if cmc[0] > best_acc:
                best_acc = cmc[0]
                best_epoch = epoch
                best_mAP = mAP
                best_mINP = mINP
                state = {
                    "main_net": main_net.state_dict(),
                    "cmc": cmc,
                    "mAP": mAP,
                    "mINP": mINP,
                    "epoch": epoch,
                    "n_class": n_class,
                }
                torch.save(state, os.path.join(model_path, "best_checkpoint.pth"))

            if epoch % args.save_epoch == 0:
                state = {
                    "main_net": main_net.state_dict(),
                    "cmc": cmc,
                    "mAP": mAP,
                    "mINP": mINP,
                    "epoch": epoch,
                    "n_class": n_class,
                }
                torch.save(state, os.path.join(model_path, "checkpoint_epoch{}.pth".format(epoch)))

            print("Performance: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            print("Performance: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP), file=test_os_log)
            print("Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}".format(best_epoch, best_acc, best_mAP, best_mINP))
            print('Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}'.format(best_epoch, best_acc, best_mAP, best_mINP), file=test_os_log)

            test_os_log.flush()

        torch.cuda.empty_cache()  # nvidia-smi memory release


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OTLA-ReID for training")
    parser.add_argument("--config", default="config/baseline.yaml", help="config file")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--resume_path", default="", help="checkpoint path")

    args_main = parser.parse_args()
    args = yaml.load(open(args_main.config), Loader=yaml.FullLoader)
    args = easydict.EasyDict(args)

    main_worker(args, args_main)
