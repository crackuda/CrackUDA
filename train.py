import os
import time
import torch
from metrics import Evaluator
from argparse import ArgumentParser

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from data.dataset import CrackSource
from model.erfnet_RA_parallel import Net as Net_RAP

from shutil import copyfile
from tensorboardX import SummaryWriter

current_task = 0

class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

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

def train(args, model):
    
    NUM_CLASSES = args.num_classes[args.current_task]
    print('Number of classes: {}'.format(NUM_CLASSES))
    best_miou = 0

    writer = SummaryWriter(args.save_dir)
    
    train_dataset_source = CrackSource(args = args, base_dir = args.source_dataset_path, split ='train')
    val_dataset_source = CrackSource(args = args, base_dir = args.source_dataset_path, split ='val')

    train_loader_source = DataLoader(train_dataset_source, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader_source = DataLoader(val_dataset_source, batch_size=args.batch_size_val, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    criterion = CrossEntropyLoss2d()

    print("Global current task: {}".format(current_task))

    for name, m in model.named_parameters():
        
        if 'decoder' in name:
            if 'decoder.{}'.format(current_task) in name:
                m.requires_grad = True
            else:
                m.requires_grad = False
        
        elif 'encoder' in name:
            if 'bn' in name or 'parallel_conv' in name:
                if '.{}.weight'.format(current_task) in name or '.{}.bias'.format(current_task) in name:
                    m.requires_grad = True
                else:
                    m.requires_grad = False

    
    for name, m in model.named_parameters():
        print(name, m.requires_grad)

    
    save_dir = args.save_dir
    log_file_path = os.path.join(save_dir, 'log.txt')
    model_txt_path = os.path.join(save_dir, 'model.txt')

    if (not os.path.exists(log_file_path)): 
        with open(log_file_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(model_txt_path, "w") as myfile:
        myfile.write(str(model))

    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),
                     eps=1e-08, weight_decay=1e-4)


    def lambda1(epoch): return pow((1-((epoch-1)/args.num_epochs)), 0.9)  # scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    start_epoch = 1
    evaluator = Evaluator(NUM_CLASSES)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_miou = checkpoint['best_miou']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    for epoch in range(start_epoch, args.num_epochs + 1):

        print('Epoch {}/{}'.format(epoch, args.num_epochs))
        print('-' * 10)
        scheduler.step()
        epoch_loss = []
        time_train = []


        usedLR = 0
        for param_group in optimizer.param_groups:
            print("Learning rate: {}".format(param_group['lr']))
            usedLR = float(param_group['lr'])
        evaluator.reset()
        model.train()
        for step, sample in enumerate(train_loader_source):

            start_time = time.time()
            images, labels = sample['image'], sample['label']
            inputs = images.cuda()
            targets = labels.long().cuda()

            _, outputs = model(inputs, current_task)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            pred = outputs.data.max(1)[1].cpu().numpy()
            evaluator.add_batch(targets.cpu().numpy(), pred)

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)


            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        acc = evaluator.Pixel_Accuracy()
        acc_class = evaluator.Pixel_Accuracy_Class()
        train_mIoU = evaluator.Mean_Intersection_over_Union()
        fwavacc = evaluator.Frequency_Weighted_Intersection_over_Union()

        print("Training: mIoU: {:.3f}, Acc: {:.3f}, Acc_class: {:.3f}, Fwavacc: {:.3f}".format(train_mIoU, acc, acc_class, fwavacc))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        print('Epoch time: ', sum(time_train))


        print("----- VALIDATING - EPOCH", epoch, "-----")
        average_loss_val, val_miou = eval(model, val_loader_source, criterion, current_task, args.num_classes, epoch, evaluator)

        info = {'train_loss': average_epoch_loss_train, 'val_loss': average_loss_val, 'val_miou': val_miou}

        for tag, value in info.items():
            writer.add_scalar(tag, value, epoch)
        
        if val_miou > best_miou:
            best_miou = val_miou
            
            print("Saving model with best IoU: ", best_miou)
            torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_miou': best_miou,
                }, os.path.join(save_dir, 'best_model.pth'))

            with open(args.save_dir + "/best.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, val_miou))


        
        if epoch % 10 == 0:
            torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_miou': best_miou,
                }, os.path.join(save_dir, 'model_{}.pth'.format(epoch)))

        with open(log_file_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (
                epoch, average_epoch_loss_train, average_loss_val, train_mIoU, val_miou, usedLR))



def eval(model, dataset_loader, criterion, task, num_classes, epoch, evaluator):

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

    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.save_dir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))


    model = Net_RAP(args.num_classes, args.nb_tasks, args.current_task)

    model = torch.nn.DataParallel(model).cuda()
    
    train(args, model)


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--batch_size_val', type=int, default=2, help='input val batch size')
    parser.add_argument('--source_dataset_path', type=str, default='/scratch/kushagra0301/CrackDataset', help='source dataset')
    parser.add_argument('--num_classes', type=int, default=[2], help='number of classes')
    parser.add_argument('--crop_size_height', type=int, default=512, help='crop size height')
    parser.add_argument('--crop_size_width', type=int, default=1024, help='crop size width')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--nb_tasks', type=int, default=1) 
    parser.add_argument('--current_task', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='/scratch/kushagra/crack_il', help='save directory')
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--dataset_avoided', type=str, default=None)

    args = parser.parse_args()
    main(args)



