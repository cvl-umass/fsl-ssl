# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
identity = lambda x:x
import math

def get_patches(img, transform_jigsaw, transform_patch_jigsaw, permutations):
    if np.random.rand() < 0.30:
        img = img.convert('LA').convert('RGB')## this should be L instead....... need to change that!!

    img = transform_jigsaw(img)

    s = float(img.size[0]) / 3
    a = s / 2
    tiles = [None] * 9
    for n in range(9):
        i = int(n / 3)
        j = n % 3
        c = [a * i * 2 + a, a * j * 2 + a]
        c = np.array([math.ceil(c[1] - a), math.ceil(c[0] - a), int(c[1] + a ), int(c[0] + a )]).astype(int)
        tile = img.crop(c.tolist())
        tile = transform_patch_jigsaw(tile)
        # Normalize the patches indipendently to avoid low level features shortcut
        m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
        s[s == 0] = 1
        norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
        tile = norm(tile)
        tiles[n] = tile
        
    order = np.random.randint(len(permutations))
    data = [tiles[permutations[order][t]] for t in range(9)]
    data = torch.stack(data, 0)

    return data, int(order)

def retrive_permutations(classes):
    all_perm = np.load('permutations_%d.npy' % (classes))
    if all_perm.min() == 1:
        all_perm = all_perm - 1

    return all_perm

class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity, \
                jigsaw=False, transform_jigsaw=None, transform_patch_jigsaw=None, rotation=False, isAircraft=False, grey=False):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

        self.jigsaw = jigsaw
        self.transform_jigsaw = transform_jigsaw
        self.transform_patch_jigsaw = transform_patch_jigsaw
        self.permutations = retrive_permutations(35)

        self.rotation = rotation
        self.isAircraft = isAircraft
        self.grey = grey

    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        if self.grey:
            img = Image.open(image_path).convert('L').convert('RGB')
        else:
            img = Image.open(image_path).convert('RGB')
        
        if self.isAircraft:
            ## crop the banner
            img = img.crop((0,0,img.size[0],img.size[1]-20))
        
        if self.jigsaw:
            patches, order = get_patches(img, self.transform_jigsaw, self.transform_patch_jigsaw, self.permutations)
        if self.rotation:
            rotated_imgs = [
                    self.transform(img),
                    self.transform(img.rotate(90,expand=True)),
                    self.transform(img.rotate(180,expand=True)),
                    self.transform(img.rotate(270,expand=True))
                ]
            rotation_labels = torch.LongTensor([0, 1, 2, 3])
        
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        if self.jigsaw:
            return patches, order
        elif self.rotation:
            return torch.stack(rotated_imgs, dim=0), rotation_labels
        else:
            return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, batch_size, transform, jigsaw=False, \
                transform_jigsaw=None, transform_patch_jigsaw=None, rotation=False, isAircraft=False, grey=False):
        self.jigsaw = jigsaw
        self.transform_jigsaw = transform_jigsaw
        self.transform_patch_jigsaw = transform_patch_jigsaw
        self.rotation = rotation
        self.isAircraft = isAircraft
        self.grey = grey

        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform, jigsaw=self.jigsaw, \
                                    transform_jigsaw=self.transform_jigsaw, transform_patch_jigsaw=self.transform_patch_jigsaw, \
                                    rotation=self.rotation, isAircraft=self.isAircraft, grey=self.grey)
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, \
                jigsaw=False, transform_jigsaw=None, transform_patch_jigsaw=None, rotation=False, isAircraft=False, grey=False):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

        self.rotation = rotation
        self.isAircraft = isAircraft
        self.grey = grey

        self.jigsaw = jigsaw
        if jigsaw:
            self.permutations = retrive_permutations(35)
            self.transform_jigsaw = transform_jigsaw
            self.transform_patch_jigsaw = transform_patch_jigsaw

    def __getitem__(self,i):
        image_path = os.path.join(self.sub_meta[i])
        if self.grey:
            img = Image.open(image_path).convert('L').convert('RGB')
        else:
            img = Image.open(image_path).convert('RGB')

        if self.isAircraft:
            ## crop the banner
            img = img.crop((0,0,img.size[0],img.size[1]-20))

        if self.jigsaw:
            patches, order = get_patches(img, self.transform_jigsaw, self.transform_patch_jigsaw, self.permutations)
        if self.rotation:
            rotated_imgs = [
                    self.transform(img),
                    self.transform(img.rotate(90,expand=True)),
                    self.transform(img.rotate(180,expand=True)),
                    self.transform(img.rotate(270,expand=True))
                ]
            rotation_labels = torch.LongTensor([0, 1, 2, 3])
        img = self.transform(img)
        target = self.target_transform(self.cl)
        
        if self.jigsaw:
            return img, target, patches, order
        elif self.rotation:
            return img, target, torch.stack(rotated_imgs, dim=0), rotation_labels
        else:
            return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
