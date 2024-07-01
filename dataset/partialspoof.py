import os
import torch
import random
import numpy as np
import soundfile as sf
import lightning as L
import logging
import re
import glob

from utils import *
from dataset.base_dataset import BaseDataset
from torch.utils.data import DataLoader

class PartialSpoofDataModule(L.LightningDataModule):
    def __init__(self,args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.get_dataset('train')
            self.vlidate_dataset = self.get_dataset('dev')
            
        if stage == 'test' or stage is None:
            self.test_dataset = self.get_dataset('eval')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.args.batch_size,
                            shuffle=True, num_workers=self.args.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.vlidate_dataset,batch_size=1,shuffle=False, num_workers=self.args.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=1,shuffle=False, num_workers=self.args.num_workers)

    def get_dataset(self, type):
        dataset = PartialSpoofDataset(
            samplerate = self.args.samplerate,
            resolution = self.args.resolution,
            root = getattr(self.args, f'{type}_root'),
            input_type = type,
            input_maxlength = self.args.input_maxlength,
            input_minlength = self.args.input_minlength,
            input_query = '*.wav',
            input_load_fn = None,
            label_root = self.args.label_root,
            label_load_fn = None,
            label_maxlength = self.args.label_maxlength,
            pad_mode = self.args.pad_mode if type=='train' else None,
            add_label = True,
        )
        return dataset

class PartialSpoofDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def default_input_load_fn(self, path):
        audio, sr = sf.read(path)
        return audio

    def default_label_load_fn(self):
        label_file = os.path.join(self.label_root, f'{self.input_type}_seglab_{self.resolution}.npy')
        labels = np.load(label_file, allow_pickle=True).item()
        labels = {k:v.astype(int) for k, v in labels.items()}
        return labels

    def add_other_label(self, items):
        utt_id, input, ori_label, ori_label_length = items
        boundary_labels_root = f'{self.label_root}/boundary_{self.resolution}_labels/{self.input_type}/{utt_id}_boundary.npy'
        boundary_label = np.load(boundary_labels_root).astype(np.float32)
        boundary_label = torch.FloatTensor(boundary_label)
        boundray_length = len(boundary_label)
        new_items = (utt_id, input, ori_label, boundary_label, ori_label_length, boundray_length)
        return new_items

    def pad(self, items):
        if self.pad_mode == 'label':
            utt_id, input, ori_label, boundary_label, ori_label_length, boundary_length = items
            scale = int(self.samplerate * self.resolution)
            input = torch.nn.functional.pad(input,(0,ori_label_length*scale-len(input)), mode='constant', value=0)
            # custom_scale = int(self.resolution // 0.02)
            custom_scale = 1
            boundary_label = torch.nn.functional.pad(boundary_label, (0,ori_label_length*custom_scale-len(boundary_label)), mode='constant', value=0)
            if ori_label_length < self.label_maxlength:
                ori_label = torch.nn.functional.pad(ori_label,(0,self.label_maxlength-ori_label_length),mode='constant',value=0)
                boundary_label = torch.nn.functional.pad(boundary_label,(0,self.label_maxlength*custom_scale-len(boundary_label)),mode='constant',value=0)
                input = torch.nn.functional.pad(input,(0,self.label_maxlength*scale-len(input)),mode='constant',value=0) 
            if ori_label_length > self.label_maxlength:
                startp = random.randint(0, ori_label_length-self.label_maxlength)
                ori_label = ori_label[startp:startp+self.label_maxlength]
                boundary_label = boundary_label[startp*custom_scale:(startp+self.label_maxlength)*custom_scale]
                input = input[startp*scale : (startp+self.label_maxlength)*scale]
            new_items = (utt_id, input, ori_label, boundary_label, ori_label_length, boundary_length)
        return new_items

if __name__ == '__main__':
    print('define of partialspoof dataset')