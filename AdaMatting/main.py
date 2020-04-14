import os
import torch
import torchvision
from tensorboardX import SummaryWriter
from tqdm import tqdm
import cv2 as cv
from torchvision import transforms
import numpy as np

from dataset.dataset import AdaMattingDataset
from dataset.pre_process import composite_dataset, gen_train_valid_names
from net.adamatting import AdaMatting
from loss import task_uncertainty_loss
from utility import get_args, get_logger, lr_scheduler, save_checkpoint, AverageMeter, \
                    compute_mse, compute_sad, gen_test_names


def train(args, logger, device_ids):
    torch.manual_seed(7)
    writer = SummaryWriter()

    logger.info("Loading network")
    model = AdaMatting(in_channel=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    if args.resume != "":
        ckpt = torch.load(args.resume)
        # for key, _ in ckpt.items():
        #     print(key)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
    if args.cuda:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        device = torch.device("cuda:{}".format(device_ids[0]))
        if len(device_ids) > 1:
            logger.info("Loading with multiple GPUs")
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        # model = model.cuda(device=device_ids[0])
    else:
        device = torch.device("cpu")
    model = model.to(device)

    logger.info("Initializing data loaders")
    train_dataset = AdaMattingDataset(args.raw_data_path, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                               num_workers=16, pin_memory=True)
    valid_dataset = AdaMattingDataset(args.raw_data_path, "valid")
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, 
                                               num_workers=16, pin_memory=True)

    if args.resume != "":
        logger.info("Start training from saved ckpt")
        start_epoch = ckpt["epoch"] + 1
        cur_iter = ckpt["cur_iter"]
        peak_lr = ckpt["peak_lr"]
        best_loss = ckpt["best_loss"]
        best_alpha_loss = float('inf')
    else:
        logger.info("Start training from scratch")
        start_epoch = 0
        cur_iter = 0
        peak_lr = args.lr
        best_loss = float('inf')
        best_alpha_loss = float('inf')

    max_iter = 43100 * (1 - args.valid_portion / 100) / args.batch_size * args.epochs
    tensorboard_iter = cur_iter * (args.batch_size / 16)

    avg_lo = AverageMeter()
    avg_lt = AverageMeter()
    avg_la = AverageMeter()
    for epoch in range(start_epoch, args.epochs):
        # Training
        torch.set_grad_enabled(True)
        model.train()
        for index, (img, gt) in enumerate(train_loader):
            # cur_lr, peak_lr = lr_scheduler(optimizer=optimizer, cur_iter=cur_iter, peak_lr=peak_lr, end_lr=0.000001, 
            #                                decay_iters=args.decay_iters, decay_power=0.8, power=0.5)
            cur_lr = lr_scheduler(optimizer=optimizer, init_lr=args.lr, cur_iter=cur_iter, max_iter=max_iter, 
                                  max_decay_times=45, decay_rate=0.9)
            
            img = img.type(torch.FloatTensor).to(device) # [bs, 4, 320, 320]
            gt_alpha = (gt[:, 0, :, :].unsqueeze(1)).type(torch.FloatTensor).to(device) # [bs, 1, 320, 320]
            gt_trimap = gt[:, 1, :, :].type(torch.LongTensor).to(device) # [bs, 320, 320]

            optimizer.zero_grad()
            trimap_adaption, t_argmax, alpha_estimation, log_sigma_t_sqr, log_sigma_a_sqr = model(img)

            L_overall, L_t, L_a = task_uncertainty_loss(pred_trimap=trimap_adaption, pred_trimap_argmax=t_argmax, 
                                                        pred_alpha=alpha_estimation, gt_trimap=gt_trimap, 
                                                        gt_alpha=gt_alpha, log_sigma_t_sqr=log_sigma_t_sqr, log_sigma_a_sqr=log_sigma_a_sqr)

            sigma_t, sigma_a = torch.exp(log_sigma_t_sqr.mean() / 2), torch.exp(log_sigma_a_sqr.mean() / 2)

            optimizer.zero_grad()
            L_overall.backward()
            optimizer.step()

            avg_lo.update(L_overall.item())
            avg_lt.update(L_t.item())
            avg_la.update(L_a.item())

            if cur_iter % 10 == 0:
                logger.info("Epoch: {:03d} | Iter: {:05d}/{} | Loss: {:.4e} | L_t: {:.4e} | L_a: {:.4e}"
                            .format(epoch, index, len(train_loader), avg_lo.avg, avg_lt.avg, avg_la.avg))
                writer.add_scalar("loss/L_overall", avg_lo.avg, tensorboard_iter)
                writer.add_scalar("loss/L_t", avg_lt.avg, tensorboard_iter)
                writer.add_scalar("loss/L_a", avg_la.avg, tensorboard_iter)
                writer.add_scalar("other/sigma_t", sigma_t.item(), tensorboard_iter)
                writer.add_scalar("other/sigma_a", sigma_a.item(), tensorboard_iter)
                writer.add_scalar("other/lr", cur_lr, tensorboard_iter)

                avg_lo.reset()
                avg_lt.reset()
                avg_la.reset()
            
            cur_iter += 1
            tensorboard_iter = cur_iter * (args.batch_size / 16)
        
        # Validation
        logger.info("Validating after the {}th epoch".format(epoch))
        avg_loss = AverageMeter()
        avg_l_t = AverageMeter()
        avg_l_a = AverageMeter()
        torch.cuda.empty_cache()
        torch.set_grad_enabled(False)
        model.eval()
        with tqdm(total=len(valid_loader)) as pbar:
            for index, (img, gt) in enumerate(valid_loader):
                img = img.type(torch.FloatTensor).to(device) # [bs, 4, 320, 320]
                gt_alpha = (gt[:, 0, :, :].unsqueeze(1)).type(torch.FloatTensor).to(device) # [bs, 1, 320, 320]
                gt_trimap = gt[:, 1, :, :].type(torch.LongTensor).to(device) # [bs, 320, 320]

                trimap_adaption, t_argmax, alpha_estimation, log_sigma_t_sqr, log_sigma_a_sqr = model(img)
                L_overall_valid, L_t_valid, L_a_valid = task_uncertainty_loss(pred_trimap=trimap_adaption, pred_trimap_argmax=t_argmax, 
                                                            pred_alpha=alpha_estimation, gt_trimap=gt_trimap, 
                                                            gt_alpha=gt_alpha, log_sigma_t_sqr=log_sigma_t_sqr, log_sigma_a_sqr=log_sigma_a_sqr)

                # L_overall_valid, L_t_valid, L_a_valid = L_overall_valid.mean(), L_t_valid.mean(), L_a_valid.mean()

                avg_loss.update(L_overall_valid.item())
                avg_l_t.update(L_t_valid.item())
                avg_l_a.update(L_a_valid.item())

                if index == 0:
                    trimap_adaption_res = (t_argmax.type(torch.FloatTensor) / 2).unsqueeze(dim=1)
                    trimap_adaption_res = torchvision.utils.make_grid(trimap_adaption_res, normalize=False, scale_each=True)
                    writer.add_image('valid/pred/trimap_adaptation', trimap_adaption_res, tensorboard_iter)
                    alpha_estimation_res = torchvision.utils.make_grid(alpha_estimation, normalize=True, scale_each=True)
                    writer.add_image('valid/pred/alpha_estimation', alpha_estimation_res, tensorboard_iter)
                
                pbar.update()

        logger.info("Average loss overall: {:.4e}".format(avg_loss.avg))
        logger.info("Average loss of trimap adaptation: {:.4e}".format(avg_l_t.avg))
        logger.info("Average loss of alpha estimation: {:.4e}".format(avg_l_a.avg))
        writer.add_scalar("valid_loss/L_overall", avg_loss.avg, tensorboard_iter)
        writer.add_scalar("valid_loss/L_t", avg_l_t.avg, tensorboard_iter)
        writer.add_scalar("valid_loss/L_a", avg_l_a.avg, tensorboard_iter)

        is_best = avg_loss.avg < best_loss
        best_loss = min(avg_loss.avg, best_loss)
        is_alpha_best = avg_l_a.avg < best_alpha_loss
        best_alpha_loss = min(avg_l_a.avg, best_alpha_loss)
        if is_best or is_alpha_best or args.save_ckpt:
            if not os.path.exists("ckpts"):
                os.makedirs("ckpts")
            save_checkpoint(ckpt_path=args.ckpt_path, is_best=is_best, is_alpha_best=is_alpha_best, logger=logger, model=model, optimizer=optimizer, 
                            epoch=epoch, cur_iter=cur_iter, peak_lr=peak_lr, best_loss=best_loss, best_alpha_loss=best_alpha_loss)

    writer.close()


def test(args, logger, device_ids):
    logger.info("Loading network")
    model = AdaMatting(in_channel=4)
    ckpt = torch.load("./ckpts/ckpt_best_alpha.tar")
    model.load_state_dict(ckpt["state_dict"])
    if args.cuda:
        device = torch.device("cuda:{}".format(device_ids[0]))
        if len(device_ids) > 1:
            logger.info("Loading with multiple GPUs")
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        # model = model.cuda(device=device_ids[0])
    else:
        device = torch.device("cpu")
    model = model.to(device)
    torch.set_grad_enabled(False)
    model.eval()

    test_names = gen_test_names()

    with open(os.path.join(args.raw_data_path, "Combined_Dataset/Test_set/test_fg_names.txt")) as f:
        fg_files = f.read().splitlines()
    with open(os.path.join(args.raw_data_path, "Combined_Dataset/Test_set/test_bg_names.txt")) as f:
        bg_files = f.read().splitlines()

    out_path = os.path.join(args.raw_data_path, "pred/")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    logger.info("Start testing")
    avg_sad = AverageMeter()
    avg_mse = AverageMeter()
    for index, name in enumerate(test_names):
        torch.cuda.empty_cache()
        # file names
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        img_name = fg_files[fcount]
        bg_name = bg_files[bcount]
        merged_name = bg_name.split(".")[0] + "!" + img_name.split(".")[0] + "!" + str(fcount) + "!" + str(index) + ".png"
        trimap_name = img_name.split(".")[0] + "_" + str(index % 20) + ".png"

        # read files
        merged = os.path.join(args.raw_data_path, "test/merged/", merged_name)
        alpha = os.path.join(args.raw_data_path, "test/mask/", img_name)
        trimap = os.path.join(args.raw_data_path, "Combined_Dataset/Test_set/Adobe-licensed images/trimaps/", trimap_name)
        merged = cv.imread(merged)
        merged = cv.resize(merged, None, fx=0.75, fy=0.75)
        merged = cv.cvtColor(merged, cv.COLOR_BGR2RGB)
        trimap = cv.imread(trimap)
        trimap = cv.resize(trimap, None, fx=0.75, fy=0.75)
        alpha = cv.imread(alpha, 0)
        alpha = cv.resize(alpha, None, fx=0.75, fy=0.75)
        # cv.imwrite("merged.png", merged)
        # cv.imwrite("trimap.png", trimap)
        # cv.imwrite("alpha.png", alpha)

        # process merged image
        merged = transforms.ToPILImage()(merged)
        out_merged = merged.copy()
        merged = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(merged)
        h, w = merged.shape[1:3]
        h_crop, w_crop = h, w
        for i in range(h):
            if (h - i) % 16 == 0:
                h_crop = h - i
                break
        h_margin = int((h - h_crop) / 2)
        for i in range(w):
            if (w - i) % 16 == 0:
                w_crop = w - i
                break
        w_margin = int((w - w_crop) / 2)

        # write cropped gt alpha
        alpha = alpha[h_margin : h_margin + h_crop, w_margin : w_margin + w_crop]
        cv.imwrite(out_path + "{:04d}_gt_alpha.png".format(index), alpha)

        # generate and write cropped gt trimap
        gt_trimap = np.zeros(alpha.shape)
        gt_trimap.fill(128)
        gt_trimap[alpha <= 0] = 0
        gt_trimap[alpha >= 255] = 255
        cv.imwrite(out_path + "{:04d}_gt_trimap.png".format(index), gt_trimap)

        # concat the 4-d input and crop to feed the network properly
        x = torch.zeros((1, 4, h, w), dtype=torch.float)
        x[0, 0:3, :, :] = merged
        x[0, 3, :, :] = torch.from_numpy(trimap[:, :, 0] / 255.)
        x = x[:, :, h_margin : h_margin + h_crop, w_margin : w_margin + w_crop]

        # write cropped input images
        out_merged = transforms.ToTensor()(out_merged)
        out_merged = out_merged[:, h_margin : h_margin + h_crop, w_margin : w_margin + w_crop]
        out_merged = transforms.ToPILImage()(out_merged)
        out_merged.save(out_path + "{:04d}_input_merged.png".format(index))
        out_trimap = transforms.ToPILImage()(x[0, 3, :, :])
        out_trimap.save(out_path + "{:04d}_input_trimap.png".format(index))

        # test
        x = x.type(torch.FloatTensor).to(device)
        _, pred_trimap, pred_alpha, _, _ = model(x)
        torch.cuda.empty_cache()

        cropped_trimap = x[0, 3, :, :].unsqueeze(dim=0).unsqueeze(dim=0)
        pred_alpha[cropped_trimap <= 0] = 0.0
        pred_alpha[cropped_trimap >= 1] = 1.0

        # output predicted images
        pred_trimap = (pred_trimap.type(torch.FloatTensor) / 2).unsqueeze(dim=1)
        pred_trimap = transforms.ToPILImage()(pred_trimap[0, :, :, :])
        pred_trimap.save(out_path + "{:04d}_pred_trimap.png".format(index))
        out_pred_alpha = transforms.ToPILImage()(pred_alpha[0, :, :, :].cpu())
        out_pred_alpha.save(out_path + "{:04d}_pred_alpha.png".format(index))
        
        sad = compute_sad(pred_alpha, alpha)
        mse = compute_mse(pred_alpha, alpha, trimap)
        avg_sad.update(sad.item())
        avg_mse.update(mse.item())
        logger.info("{:04d}/{} | SAD: {:.1f} | MSE: {:.3f} | Avg SAD: {:.1f} | Avg MSE: {:.3f}".format(index, len(test_names), sad.item(), mse.item(), avg_sad.avg, avg_mse.avg))
    
    logger.info("Average SAD: {:.1f} | Average MSE: {:.3f}".format(avg_sad.avg, avg_mse.avg))


def main():
    args = get_args()
    logger = get_logger(args.write_log)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_ids_str = args.gpu.split(",")
    device_ids = []
    for i in range(len(device_ids_str)):
        device_ids.append(i)

    # if args.mode == "train":
    #     logger.info("Loading network")
    #     model = AdaMatting(in_channel=4)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    #     if args.cuda:
    #         device = torch.device("cuda:{}".format(device_ids[0]))
    #         if len(device_ids) > 1 and args.mode=="train":
    #             logger.info("Loading with multiple GPUs")
    #             model = torch.nn.DataParallel(model, device_ids=device_ids)
    #         model = model.cuda(device=device_ids[0])
    #     else:
    #         device = torch.device("cpu")
    # elif args.mode == "test":
    #     if args.cuda:
    #         device = torch.device("cuda:{}".format(device_ids[0]))
    #     else:
    #         device = torch.device("cpu")

    if args.mode == "train":
        logger.info("Program runs in train mode")
        train(args=args, logger=logger, device_ids=device_ids)
    elif args.mode == "test":
        logger.info("Program runs in test mode")
        test(args=args, logger=logger, device_ids=device_ids)
    elif args.mode == "prep":
        logger.info("Program runs in prep mode")
        # composite_dataset(args.raw_data_path, logger)
        gen_train_valid_names(args.valid_portion, logger)


if __name__ == "__main__":
    main()
