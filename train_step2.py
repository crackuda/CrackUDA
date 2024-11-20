import os
import random
import time
import numpy as np
import torch
import re
from metrics import Evaluator
from argparse import ArgumentParser
import cv2
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.dataset import CrackSource, CrackTargetNew
from model.erfnet_RA_parallel import Net as Net_RAP
from model.discriminator import Discriminator

from shutil import copyfile
from tensorboardX import SummaryWriter

current_task = 0

class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)

class BCELoss2d(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCELoss()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)

def is_shared(n):
    return 'encoder' in n and 'parallel_conv' not in n and 'bn' not in n



def is_DS_curr(n):
    if 'decoder.{}'.format(current_task) in n:
        return True
    elif 'encoder' in n:
        if 'bn' in n or 'parallel_conv' in n:
            if '.{}.weight'.format(current_task) in n or '.{}.bias'.format(current_task) in n:
                return True

def train(args, model, model_old, discriminator):
    
    NUM_CLASSES = args.num_classes[args.current_task]
    print('NUM_CLASSES: ', NUM_CLASSES)


    best_miou_old_model_source = 0
    best_miou_old_model_target = 0
    best_miou_current_model_source = 0
    best_miou_current_model_target = 0

    writer = SummaryWriter(args.save_dir)
    
    train_dataset_old = CrackSource(args = args, base_dir = args.source_dataset_path, split ='train')
    val_dataset_old= CrackSource(args = args, base_dir = args.source_dataset_path, split ='val')
    target_dataset = CrackTargetNew(args = args, base_dir = args.target_dataset_path, base_dir_source = args.source_dataset_path)

    train_loader_old = DataLoader(train_dataset_old, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader_old = DataLoader(val_dataset_old, batch_size=args.batch_size_val, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    criterion = CrossEntropyLoss2d()
    criterion_discriminator = BCELoss2d()

    print("Global current task: {}".format(current_task))


    for name, m in model_old.named_parameters():

        m.requires_grad = False

    for name, m in model.named_parameters():
        
        if 'decoder' in name:
            if 'decoder.{}'.format(current_task) not in name:
                m.requires_grad = False
            
        elif 'encoder' in name:
            if 'bn' in name or 'parallel_conv' in name:
                if '.{}.weight'.format(current_task) in name or '.{}.bias'.format(current_task) in name:
                    continue
                else:
                    m.requires_grad = False

    save_dir = args.save_dir
    log_file_path = os.path.join(save_dir, 'log.txt')
    model_txt_path = os.path.join(save_dir, 'model.txt')

    if (not os.path.exists(log_file_path)): 
        with open(log_file_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(model_txt_path, "w") as myfile:
        myfile.write(str(model))

    
    params = list(model.named_parameters())

    
    grouped_parameters = [
        {"params": [p for n, p in params if is_shared(n)], 'lr': 5e-6},
        {"params": [p for n, p in params if is_DS_curr(n)]},  
        {"params": discriminator.parameters()}
    ]

    # for name, m in discriminator.named_parameters():
    #     grouped_parameters.append(m)
    print("Model")
    for name, m in model.named_parameters():
        print(name, m.requires_grad)
    
    print("Model_old")
    for name, m in model_old.named_parameters():
        print(name, m.requires_grad)
    
    print("Discriminator")
    for name, m in discriminator.named_parameters():
        print(name, m.requires_grad)
    
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    optimizer_discriminator = Adam(grouped_parameters, 5e-4, (0.9, 0.999),eps=1e-08, weight_decay=1e-4)
    
    kl_loss = torch.nn.KLDivLoss()

    def lambda1(epoch): return pow((1-((epoch-1)/args.num_epochs)), 0.9)  # scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  # scheduler 2
    scheduler_discriminator = lr_scheduler.LambdaLR(optimizer_discriminator, lr_lambda=lambda1)  # scheduler 2


    def lambda_grl(epoch): return 2.0 / (1.0 + np.exp(-10 * epoch )) - 1

    start_epoch = 0
    segmentation_epoch = 0
    discriminator_epoch = 0
    evaluator = Evaluator(NUM_CLASSES)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_old.load_state_dict(checkpoint['state_dict_old'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    i = 0
    for epoch in range(start_epoch, args.num_epochs):
    
        NUM_CLASSES = args.num_classes[args.current_task]

        
        if i < 10:
            print("-----TRAINING SEGMENTATION- EPOCH---", epoch, "-----")
            print("-----SUB EPOCH---", segmentation_epoch, "-----")

            scheduler.step(segmentation_epoch)  # scheduler 2
            epoch_loss = []
            time_train = []
            e_kld_loss = []
            e_ce_loss = []

            usedLr = 0
            for param_group in optimizer.param_groups:
                print("LEARNING RATE: ", param_group['lr'])
                usedLr = float(param_group['lr'])
            
            model.train()
            model_old.eval()
            evaluator.reset()

            for step, sample in enumerate(train_loader_old):
                start_time = time.time()
                images, labels = sample['image'], sample['label']

                inputs = images.cuda()
                targets = labels.cuda().long()

                
                _, output_prev_task = model(inputs, current_task-1)
                _, outputs_prev_model = model_old(inputs, current_task-1)

                _, outputs = model(inputs, current_task)

                KLD_Loss = kl_loss(F.softmax(output_prev_task, dim=1), F.softmax(outputs_prev_model, dim=1))
                ce_loss = criterion(outputs, targets)

                loss = args.kld_weight * KLD_Loss + ce_loss * args.ce_weight

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
                time_train.append(time.time() - start_time)
                e_kld_loss.append(KLD_Loss.item())
                e_ce_loss.append(ce_loss.item())

                torch.cuda.empty_cache()

                pred = outputs.data.max(1)[1].cpu().numpy()
                evaluator.add_batch(targets.cpu().numpy(), pred)

                if args.steps_loss > 0 and step % args.steps_loss == 0:
                    average = sum(epoch_loss) / len(epoch_loss)
                    average_kld = sum(e_kld_loss) / len(e_kld_loss)
                    average_ce = sum(e_ce_loss) / len(e_ce_loss)
                    print(f'loss: {average:0.4} kld_loss: {average_kld:0.4} ce_loss: {average_ce:0.4} (epoch: {epoch}, step: {step})',
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

            segmentation_epoch += 1

            acc = evaluator.Pixel_Accuracy()
            acc_class = evaluator.Pixel_Accuracy_Class()
            train_mIoU = evaluator.Mean_Intersection_over_Union()
            fwavacc = evaluator.Frequency_Weighted_Intersection_over_Union()

            print("Training Segmentation: mIoU: {:.3f}, Acc: {:.3f}, Acc_class: {:.3f}, Fwavacc: {:.3f}".format(train_mIoU, acc, acc_class, fwavacc))

            average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
            average_epoch_kld_loss_train = sum(e_kld_loss) / len(e_kld_loss)
            average_epoch_ce_loss_train = sum(e_ce_loss) / len(e_ce_loss)
            print('Epoch time: ', sum(time_train))
            info = {'train_loss': average_epoch_loss_train, 'kld_loss': average_epoch_kld_loss_train, 'ce_loss': average_epoch_ce_loss_train, 'lr': usedLr}
            
            for tag, value in info.items():
                writer.add_scalar(tag, value, epoch)
                


    
        elif i >= 10 and i < 15:

            print("-----TRAINING DOMAIN ADAPTATION- EPOCH---", epoch, "-----")
            print("-----SUB EPOCH---", discriminator_epoch, "-----")
            
            model.train()
            discriminator.train()

            scheduler_discriminator.step(discriminator_epoch)  
            epoch_loss = []
            time_train = []
            for param_group in optimizer_discriminator.param_groups:
                print("LEARNING RATE: ", param_group['lr'])
            
            step = 0

            current_lambda_grl = lambda_grl(discriminator_epoch)
            print("Current lambda:", current_lambda_grl)
            for sample_source, sample_target in zip(train_loader_old, target_loader):
                
                start_time = time.time()
                images_source = sample_source['image']
                domain_source = sample_source['domain']
                images_target = sample_target['image']
                domain_target = sample_target['domain']

                inputs = torch.cat((images_source, images_target), 0).cuda()

                encoder_out, _ = model(inputs, current_task)
                domain_pred = discriminator(encoder_out, current_lambda_grl)
                domain =  torch.cat((domain_source, domain_target), 0).cuda().unsqueeze(1).float()
                loss = criterion_discriminator(domain_pred, domain)
                optimizer_discriminator.zero_grad()
                loss.backward()
                optimizer_discriminator.step()

                epoch_loss.append(loss.item())
                time_train.append(time.time() - start_time)

                if args.steps_loss > 0 and step % args.steps_loss == 0:
                    average = sum(epoch_loss) / len(epoch_loss)
                    print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})',
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))
                
                step += 1

            discriminator_epoch += 1
            average_epoch_loss_train_discriminator = sum(epoch_loss) / len(epoch_loss)
            print('Epoch time: ', sum(time_train))

            info = {'Total_discriminator_loss': average_epoch_loss_train_discriminator} 

            for tag, value in info.items():
                writer.add_scalar(tag, value, epoch)
            
            
        
        #saving checkpoint after every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'state_dict_old': model_old.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_discriminator': optimizer_discriminator.state_dict(),
                'best_miou_old_model_target': best_miou_old_model_target,
                'best_miou_old_model_source': best_miou_old_model_source,
                'best_miou_current_model_target': best_miou_current_model_target,
                'best_miou_current_model_source': best_miou_current_model_source,
            }, os.path.join(save_dir, str(epoch)+'_checkpoint.pth.tar'))
            print("Checkpoint saved")
        i += 1

        if i == 15:
            i = 0
            print("Validation afet 10 segmentation epochs and 5 DA epochs")
            average_loss_val_source_old, val_miou_source_old = eval(model, val_loader_old, criterion, 0, epoch, evaluator)
            average_loss_val_target_old, val_miou_target_old = eval(model, target_loader, criterion, 0, epoch, evaluator)
            average_loss_val_source_current, val_miou_source_current = eval(model, val_loader_old, criterion, current_task, epoch, evaluator)
            average_loss_val_target_current, val_miou_target_current = eval(model, target_loader, criterion, current_task, epoch, evaluator)
            print("Validation for old model on source: mIoU: {:.3f}".format(val_miou_source_old))
            print("Validation for old model on target: mIoU: {:.3f}".format(val_miou_target_old))
            print("Validation for new model on source: mIoU: {:.3f}".format(val_miou_source_current))
            print("Validation for new model on target: mIoU: {:.3f}".format(val_miou_target_current))


            info = {'val_loss_source_old': average_loss_val_source_old, 'val_loss_target_old': average_loss_val_target_old, 'val_loss_source_current': average_loss_val_source_current, 'val_loss_target_current': average_loss_val_target_current}

            for tag, value in info.items():
                writer.add_scalar(tag, value, epoch)
            
            if val_miou_target_old > best_miou_old_model_target and val_miou_source_old > best_miou_old_model_source:
                best_miou_old_model_target = val_miou_target_old
                best_miou_old_model_source = val_miou_source_old
                    
                print("Saving old model with best source mIoU: {} and target mIoU: {}: ".format(best_miou_old_model_source, best_miou_old_model_target))
                torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'state_dict_old': model_old.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'optimizer_discriminator': optimizer_discriminator.state_dict(),
                        'best_miou_old_model_target': best_miou_old_model_target,
                        'best_miou_old_model_source': best_miou_old_model_source,
                    }, os.path.join(save_dir, 'best_model_old.pth'))

            if val_miou_target_current > best_miou_current_model_target and val_miou_source_current > best_miou_current_model_source:
                best_miou_current_model_target = val_miou_target_current
                best_miou_current_model_source = val_miou_source_current

                print("Saving current model with best source mIoU: {} and target mIoU: {}: ".format(best_miou_current_model_source, best_miou_current_model_target))
                torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'state_dict_old': model_old.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'optimizer_discriminator': optimizer_discriminator.state_dict(),
                        'best_miou_current_model_target': best_miou_current_model_target,
                        'best_miou_current_model_source': best_miou_current_model_source,
                    }, os.path.join(save_dir, 'best_model_current.pth'))
                with open(args.save_dir + "/best.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Source Val-IoU= %.4f and Target Val-IoU" % (epoch, best_miou_current_model_source, best_miou_current_model_target))

def eval(model, dataset_loader, criterion, task, epoch, evaluator):

    model.eval()
    epoch_loss_val = []
    time_val = []
    evaluator.reset()
    with torch.no_grad():
        for step, sample in enumerate(dataset_loader):
            images, labels = sample['image'], sample['label']
            start_time = time.time()
            inputs = images.cuda()
            targets = labels.long().cuda()

            _, outputs = model(inputs, task)
            loss = criterion(outputs, targets)
            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)

            pred = outputs.data.max(1)[1].cpu().numpy()
            evaluator.add_batch(targets.cpu().numpy(), pred)

            if 50 > 0 and step % 50 == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / 6))

    average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

    acc = evaluator.Pixel_Accuracy()
    acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    fwavacc = evaluator.Frequency_Weighted_Intersection_over_Union()

    print("Validation: mIoU: {:.3f}, Acc: {:.3f}, Acc_class: {:.3f}, Fwavacc: {:.3f}".format(mIoU, acc, acc_class, fwavacc))

    print('check val loss', average_epoch_loss_val)
    return average_epoch_loss_val, mIoU

def main(args):
    
    global current_task
    current_task = args.current_task
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.save_dir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    model = Net_RAP(args.num_classes, args.nb_tasks, args.current_task)
    model_old = Net_RAP(args.num_classes, args.nb_tasks-1, args.current_task-1)
    discriminator = Discriminator()

    model = torch.nn.DataParallel(model).cuda()
    model_old = torch.nn.DataParallel(model_old).cuda()
    saved_model = torch.load(args.saved_model)
    model_old.load_state_dict(saved_model['state_dict'])
    discriminator = torch.nn.DataParallel(discriminator).cuda()

    #old saved model

    print("Loading previous weights from: ", args.saved_model)
    new_dict_load = {}
    for k, v in saved_model['state_dict'].items():
        if k in model.state_dict().keys():
            new_dict_load[k] = v

    print("Copying weights from previous model to current model")


    for k, v in saved_model['state_dict'].items():
        if 'encoder' in k:
            if 'parallel_conv' in k or 'bn' in k:
                if '.{}.weight'.format(current_task-1) in k:
                    nkey = re.sub('.{}.weight'.format(current_task-1),'.{}.weight'.format(current_task), k)
                    new_dict_load[nkey] = v
                elif '.{}.bias'.format(current_task-1) in k:
                    nkey = re.sub('.{}.bias'.format(current_task-1),
                                          '.{}.bias'.format(current_task), k)
                    new_dict_load[nkey] = v

        elif 'decoder' in k and 'output_conv' not in k:
                    # this is important so as to maintain uniformity among bdd and idd experiments.
            nkey = re.sub('decoder.{}'.format(current_task-1),
                                  'decoder.{}'.format(current_task), k)
            new_dict_load[nkey] = v

    model.load_state_dict(new_dict_load, strict=False)
    print("Loaded model from checkpoint")
    train(args, model, model_old, discriminator)

def save_results(model, dataset_loader, task, epoch, save_dir):

    model.eval()
    print("Saving results for epoch: ", epoch)
    directory = save_dir + '/results_' + str(epoch)
    os.makedirs(directory, exist_ok=True)
    with torch.no_grad():
        for step, sample in enumerate(dataset_loader):
            images, labels, image_id = sample['image'], sample['label'], sample['image_id']
            inputs = images.cuda()

            _, outputs = model(inputs, task)

            pred = outputs.data.max(1)[1].cpu().numpy()
            pred_image = np.transpose(pred, (1, 2, 0))
            pred_image = np.asarray(pred_image*255, dtype=np.uint8)
            cv2.imwrite(os.path.join(directory, image_id[0]+'.jpg'), pred_image)

def save_best_results(model, dataset_loader, task, epoch, save_dir):

    model.eval()
    print("Saving best results for epoch: ", epoch)
    directory = save_dir + '/best_results_' + str(epoch)
    os.makedirs(directory, exist_ok=True)
    with torch.no_grad():
        for step, sample in enumerate(dataset_loader):
            images, labels, image_id = sample['image'], sample['label'], sample['image_id']
            inputs = images.cuda()

            _, outputs = model(inputs, task)

            pred = outputs.data.max(1)[1].cpu().numpy()
            pred_image = np.transpose(pred, (1, 2, 0))
            pred_image = np.asarray(pred_image*255, dtype=np.uint8)
            cv2.imwrite(os.path.join(directory, image_id[0]+'.jpg'), pred_image)

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--batch_size_val', type=int, default=1, help='input val batch size')
    parser.add_argument('--source_dataset_path', type=str, default='/scratch/kushagra0301/CrackDataset1', help='source dataset')
    parser.add_argument('--target_dataset_path', type=str, default='/scratch/kushagra0301/IIITDataset', help='target dataset')
    parser.add_argument('--saved_model', type=str, default='/scratch/kushagra0301/Crack_IL_step_1/best_model.pth', help='saved model')
    parser.add_argument('--num_classes', type=int, default=[2,2], help='number of classes')
    parser.add_argument('--crop_size_height', type=int, default=512, help='crop size height')
    parser.add_argument('--crop_size_width', type=int, default=1024, help='crop size width')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--nb_tasks', type=int, default=2) 
    parser.add_argument('--current_task', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='/scratch/kushagra/crack_il', help='save directory')
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--kld_weight', type=float, default=1.0, help='kld weight')
    parser.add_argument('--ce_weight', type=float, default=1.0, help='ce weight')
    parser.add_argument('--dataset_avoided', type=str, help='dataset avoided')

    args = parser.parse_args()
    main(args)

