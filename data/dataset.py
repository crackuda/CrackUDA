from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import sys
import data.custom_transforms as tr

class CrackSource(Dataset):
    """
    Crack dataset
    """
    NUM_CLASSES = 2

    def __init__(self, args, base_dir, split='train'):

        super().__init__()

        self.split = split

        self._base_dir = base_dir
        # print(self._base_dir)
        self._image_dir = os.path.join(self._base_dir, self.split, 'images')
        self._cat_dir = os.path.join(self._base_dir, self.split, 'labels')


        self.args = args

        self.im_ids = []
        self.images = []
        self.categories = []
        self.dataset_avoided = args.dataset_avoided

        print("Total Images: ", len(os.listdir(self._image_dir)))
        print('Avoided sub-datasets: ', self.dataset_avoided)

        # self.dataset_names = {'Mason':0, 'Ceramic':0, 'CFD':0, 'CRACK500':0, 'cracktree200':0, 'DeepCrack':0, 'GAPS384':0, 'noncrack':0, 'Rissbilder':0, 'Volker':0}

        if self.split == 'train' or self.split == 'val':
            for image in os.listdir(self._image_dir):
                if self.dataset_avoided not in image:
                    self.images.append(os.path.join(self._image_dir, image))
                    self.categories.append(os.path.join(self._cat_dir, image.replace('jpg', 'png')))
                    self.im_ids.append(image.split('.')[0])
        
        elif self.split == 'test':
            for image in os.listdir(self._image_dir):
                self.images.append(os.path.join(self._image_dir, image))
                self.im_ids.append(image.split('.')[0])
                self.categories.append(os.path.join(self._image_dir, image))

        if self.split == 'train' or self.split == 'val':
            assert (len(self.images) == len(self.categories))

        print("After removing sub-datasets: ", len(self.images))

        # Display stats
            # print('Number of images in {}: {:d}'.format(split, len(self.images)))
            # print('Number of labels in {}: {:d}'.format(split, len(self.categories)))
        
        # elif self.split == 'test':
            # print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        
        _img, _target,= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            sample = self.transform_tr(sample)
            sample['image_id'] = self.im_ids[index]
            sample['image_size'] = [_img.size[0], _img.size[1]]
            sample['domain'] = 0 # 0 for source and 1 for target
            return sample
        elif self.split == 'val' or self.split == 'test':
            sample = self.transform_val(sample)
            sample['image_id'] = self.im_ids[index]
            sample['image_size'] = [_img.size[0], _img.size[1]]
            sample['domain'] = 0 # 0 for source and 1 for target
            return sample
        

    def print_dataset_names(self):
        print(self.dataset_names)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _img = _img.resize((self.args.crop_size_width, self.args.crop_size_height), Image.BILINEAR)
        _target = Image.open(self.categories[index]).convert('L')
        _target = np.asarray(_target)
        new_target = np.zeros(_target.shape)
        new_target[_target == 255] = 1
        _target = Image.fromarray(new_target.astype(np.uint8))
        _target = _target.resize((self.args.crop_size_width, self.args.crop_size_height), Image.NEAREST)
        # print(_target.size)
        # print(np.unique(np.array(_target)))

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(),
            tr.Ignore_label(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(),
            tr.Ignore_label(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return '(split=' + str(self.split) + ')'

class CrackSourceTesting(Dataset):
    """
    Crack dataset
    """
    NUM_CLASSES = 2

    def __init__(self, args, base_dir, split='val'):

        super().__init__()

        self.split = split

        self._base_dir = base_dir
        # print(self._base_dir)
        self._image_dir = os.path.join(self._base_dir, self.split, 'images')
        self._cat_dir = os.path.join(self._base_dir, self.split, 'labels')
        self.dataset_avoided = args.dataset_avoided


        self.args = args

        self.im_ids = []
        self.images = []
        self.categories = []

        print("Total Images: ", len(os.listdir(self._image_dir)))
        print('Sub-dataset: ', self.dataset_avoided)

        # self.dataset_names = {'Mason':0, 'Ceramic':0, 'CFD':0, 'CRACK500':0, 'cracktree200':0, 'DeepCrack':0, 'GAPS384':0, 'noncrack':0, 'Rissbilder':0, 'Volker':0}

        if self.split == 'train' or self.split == 'val':
            for image in os.listdir(self._image_dir):
                if self.dataset_avoided in image:
                    self.images.append(os.path.join(self._image_dir, image))
                    self.categories.append(os.path.join(self._cat_dir, image))
                    self.im_ids.append(image.split('.')[0])
    
        elif self.split == 'test':
            for image in os.listdir(self._image_dir):
                self.images.append(os.path.join(self._image_dir, image))
                self.im_ids.append(image.split('.')[0])
                self.categories.append(os.path.join(self._image_dir, image))

        if self.split == 'train' or self.split == 'val':
            assert (len(self.images) == len(self.categories))

        print("After keeping sub-datasets: ", len(self.images))

        # Display stats
            # print('Number of images in {}: {:d}'.format(split, len(self.images)))
            # print('Number of labels in {}: {:d}'.format(split, len(self.categories)))
        
        # elif self.split == 'test':
            # print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        
        _img, _target,= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            sample = self.transform_tr(sample)
            sample['image_id'] = self.im_ids[index]
            sample['image_size'] = [_img.size[0], _img.size[1]]
            sample['domain'] = 0 # 0 for source and 1 for target
            return sample
        elif self.split == 'val' or self.split == 'test':
            sample = self.transform_val(sample)
            sample['image_id'] = self.im_ids[index]
            sample['image_size'] = [_img.size[0], _img.size[1]]
            sample['domain'] = 0 # 0 for source and 1 for target
            return sample
        

    def print_dataset_names(self):
        print(self.dataset_names)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _img = _img.resize((self.args.crop_size_width, self.args.crop_size_height), Image.BILINEAR)
        _target = Image.open(self.categories[index]).convert('L')
        _target = _target.resize((self.args.crop_size_width, self.args.crop_size_height), Image.NEAREST)
        # print(_target.size)
        # print(np.unique(np.array(_target)))

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(),
            tr.Ignore_label(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(),
            tr.Ignore_label(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return '(split=' + str(self.split) + ')'


class CrackTarget(Dataset):
    """
    Crack dataset
    """

    def __init__(self,args,base_dir):

        super().__init__()


        self._base_dir = base_dir
        # print(self._base_dir)
        self._image_dir = os.path.join(self._base_dir, 'images')


        self.args = args

        self.im_ids = []
        self.images = []

        for image in os.listdir(self._image_dir):
            self.images.append(os.path.join(self._image_dir, image))
            self.im_ids.append(image.split('.')[0])
    
        # Display stats
            # print('Number of images in target dataset: {:d}'.format(len(self.images)))
        

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        
        _img, _label,= self._make_img_gt_point_pair(index)

        #label is none for target domain
        sample = {'image': _img, 'label': _label}

        sample = self.transform(sample)
        # print(sample['image'].shape)
        sample['image_id'] = self.im_ids[index]
        sample['image_size'] = [_img.size[0], _img.size[1]]
        sample['domain'] = 1 # 0 for source and 1 for target
        return sample
    


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _img = _img.resize((self.args.crop_size_width, self.args.crop_size_height), Image.BILINEAR)
        _target = _img.copy()
        _target = _target.resize((self.args.crop_size_width, self.args.crop_size_height), Image.NEAREST)

        return _img, _target

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(),
            tr.Ignore_label(),
            tr.ToTensor()])

        return composed_transforms(sample)
    
    def __len__(self):
        return len(self.images)

class CrackTargetNew(Dataset):
    """
    Crack dataset
    """

    def __init__(self,args,base_dir, base_dir_source):

        super().__init__()


        self._base_dir = base_dir
        self._base_dir_source = base_dir_source
        self._image_dir = os.path.join(self._base_dir, 'images')
        self._label_dir = os.path.join(self._base_dir, 'labels')
        self._image_dir_source_train = os.path.join(self._base_dir_source, 'train','images')
        self._label_dir_source_train = os.path.join(self._base_dir_source, 'train','labels')
        self._image_dir_source_val = os.path.join(self._base_dir_source, 'val','images')
        self._label_dir_source_val = os.path.join(self._base_dir_source, 'val','labels')

        self.dataset_avoided = args.dataset_avoided



        self.args = args

        self.im_ids = []
        self.images = []
        self.labels = []

        for image in os.listdir(self._image_dir):
            self.images.append(os.path.join(self._image_dir, image))
            self.labels.append(os.path.join(self._label_dir, image))
            self.im_ids.append(image.split('.')[0])
        
        print("Total Images in Target Dataset: ", len(self.images))

        for image in os.listdir(self._image_dir_source_train):
            if self.dataset_avoided in image:
                self.images.append(os.path.join(self._image_dir_source_train, image))
                self.labels.append(os.path.join(self._label_dir_source_train, image.replace('.jpg','.png')))
                self.im_ids.append(image.split('.')[0])

        for image in os.listdir(self._image_dir_source_val):
            if self.dataset_avoided in image:
                self.images.append(os.path.join(self._image_dir_source_val, image))
                self.labels.append(os.path.join(self._label_dir_source_val, image.replace('.jpg','.png')))
                self.im_ids.append(image.split('.')[0])
        
        print("After adding {} images from source dataset, total images in target dataset: {}".format(self.dataset_avoided, len(self.images)))
        # Display stats
            # print('Number of images in target dataset: {:d}'.format(len(self.images)))
        

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        
        _img, _label,= self._make_img_gt_point_pair(index)

        #label is none for target domain
        sample = {'image': _img, 'label': _label}

        sample = self.transform(sample)
        # print(sample['image'].shape)
        sample['image_id'] = self.im_ids[index]
        sample['image_size'] = [_img.size[0], _img.size[1]]
        sample['domain'] = 1 # 0 for source and 1 for target
        return sample
    


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _img = _img.resize((self.args.crop_size_width, self.args.crop_size_height), Image.BILINEAR)
        _target = Image.open(self.labels[index]).convert('L')
        _target = np.asarray(_target)
        new_target = np.zeros(_target.shape)
        new_target[_target==255] = 1
        _target = Image.fromarray(new_target)
        _target = _target.resize((self.args.crop_size_width, self.args.crop_size_height), Image.NEAREST)

        return _img, _target

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(),
            tr.Ignore_label(),
            tr.ToTensor()])

        return composed_transforms(sample)
    
    def __len__(self):
        return len(self.images)



class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)



if __name__ == '__main__':
    
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = CrackSource(args, base_dir= '/scratch/kushagra0301/CrackDataset',split='val')

    dataloader = DataLoader(voc_train, batch_size=1, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        pass

    print(dataloader.dataset.print_dataset_names())
