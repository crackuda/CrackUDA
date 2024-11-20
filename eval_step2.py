import os
import numpy as np
import torch
from metrics import Evaluator
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.dataset import CrackSource, CrackTargetNew
from model.erfnet_RA_parallel import Net as Net_RAP
import cv2

def eval(model, args, loader, task, dir):
    
    evaluator = Evaluator(2)
    model.eval()
    with torch.no_grad():
        for step, sample in enumerate(loader):
            images, targets, name = sample['image'], sample['label'], sample['image_id']
            # image_size = sample['image_size']
            inputs = images.cuda()
            _, outputs = model(inputs, task)
            pred = outputs.data.max(1)[1].cpu().numpy()
            evaluator.add_batch(targets.numpy(), pred)
            
            pred_image = np.transpose(pred, (1, 2, 0))
            pred_image = np.asarray(pred_image*255, dtype=np.uint8)
            # print(pred_image.shape)
            # print(name[0])
            pred_image = cv2.resize(pred_image, (448, 448))
            cv2.imwrite(os.path.join(dir, name[0]+'.png'), pred_image)
        
        acc = evaluator.Pixel_Accuracy()
        acc_class = evaluator.Pixel_Accuracy_Class()
        train_mIoU = evaluator.Mean_Intersection_over_Union()
        fwavacc = evaluator.Frequency_Weighted_Intersection_over_Union()
        print("Validation : mIoU: {}, Acc: {}, Acc_class: {}, Fwavacc: {}".format(train_mIoU, acc, acc_class, fwavacc))

def main(args):

    train_dataset_source = CrackSource(args = args, base_dir = args.source_dataset_path, split ='train')
    val_dataset_source = CrackSource(args = args, base_dir = args.source_dataset_path, split ='val')

    target_dataset = CrackTargetNew(args = args, base_dir = args.target_dataset_path, base_dir_source=args.source_dataset_path)

    train_loader_old = DataLoader( train_dataset_source, batch_size=args.batch_size_val, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader_old = DataLoader(val_dataset_source, batch_size=args.batch_size_val, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    model = Net_RAP([2,2], 2, 1)

    model = torch.nn.DataParallel(model).cuda()
    
    if os.path.isfile(args.weight):
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint = torch.load(args.weight)
        model.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.weight, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.weight))


    # target_dir = 'results/step2/target'
    # os.makedirs(target_dir, exist_ok=True)
    source_dir = 'results/step2/source_train'
    os.makedirs(source_dir, exist_ok=True)
    # eval(model, args, target_loader, 1, target_dir)
    eval(model, args, train_loader_old, 1, source_dir)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--batch_size_val', type=int, default=1, help='input val batch size')
    parser.add_argument('--source_dataset_path', type=str, default='/scratch/kushagra0301/CrackDataset', help='source dataset')
    parser.add_argument('--target_dataset_path', type=str, default='/scratch/kushagra0301/CustomCrackDetectionModified', help='target dataset')
    parser.add_argument('--weight', type=str, default='/scratch/kushagra0301/Crack_IL_step_1/best_model.pth', help='saved model')
    parser.add_argument('--crop_size_height', type=int, default=512, help='crop size height')
    parser.add_argument('--crop_size_width', type=int, default=1024, help='crop size width')
    parser.add_argument('--result_dir', type=str, default='/scratch/kushagra/crack_il_results', help='save directory')
    parser.add_argument('--dataset_avoided', type=str)

    args = parser.parse_args()

    main(args)