import os
import numpy as np
import torch

from torch.utils.data import Dataset
from utils import find_files

class BaseDataset(Dataset):
    def __init__(
        self,
        samplerate = None,
        resolution = None,
        root = None,
        input_type = None,
        input_maxlength = None,
        input_minlength = None,
        input_query = None,
        input_load_fn = None,
        label_root = None,
        label_load_fn = None,
        label_maxlength = None,
        pad_mode = 'label',
        add_label = False,
        
        ) -> None:
        super().__init__()
        self.root = root
        self.input_load_fn = input_load_fn
        self.input_query = input_query
        self.input_maxlength = input_maxlength
        self.input_minlength = input_minlength
        self.input_type = input_type
        self.label_root = label_root
        self.samplerate = samplerate
        self.resolution = resolution
        self.pad_mode = pad_mode
        self.add_label = add_label
        self.label_maxlength = label_maxlength

        self.label_load_fn = self.default_label_load_fn if label_load_fn is None else label_load_fn
        self.input_load_fn = self.default_input_load_fn if input_load_fn is None else input_load_fn

        if isinstance(root,str):
            sample_list = sorted(find_files(root, query=self.input_query))
        elif isinstance(root,list):
            sample_list = []
            for r in root:
                sample_list += sorted(find_files(r, query=self.input_query))  
        else:
            raise AttributeError(f'{input_type} root is not a list or str, {root}')
        
        # filter by minlength
        if input_minlength is not None:
            sample_length = [self.input_load_fn(f).shape[-1] for f in sample_list]
            idxs = [idx for idx in range(len(sample_list)) if sample_length[idx] > input_minlength]  
            if len(sample_list) != len(idxs):
                print(
                    "some files are filtered by audio length threshold "
                    f"({len(sample_list)} -> {len(idxs)})."
                )
            sample_list = [sample_list[idx] for idx in idxs]  #直接丢弃短样本

        # assert the number of files
        assert len(sample_list) != 0, f"Not found any sample files in ${root}."

        # filting audio sample or segment 
        self.labels = self.label_load_fn()
        self.sample_list = self.sample_filter(sample_list) if self.input_type=='test' else sample_list
        self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.sample_list]
        

    def sample_filter(self, sample_list):
        return sample_list

    def default_input_load_fn(self):
        return np.load

    def add_other_label(self, items):
        raise NotImplementedError

    def default_label_load_fn(self):
        raise NotImplementedError
    
    def pad(self, items):
        raise NotImplementedError

    def __getitem__(self, index):
        utt_id = self.utt_ids[index]
        input = self.input_load_fn(self.sample_list[index])
        input = torch.FloatTensor(input)

        label = self.labels[utt_id]
        label = torch.FloatTensor(label)
        
        ori_label_length = len(label) 
        items = (utt_id, input, label, ori_label_length, )

        # add other supervise information        
        if self.add_label:
            items = self.add_other_label(items)

        # pad audio and label to certain length
        if self.pad_mode is not None:
            items = self.pad(items)

        return items  
    
    def __len__(self):
        return len(self.sample_list)

    def get_length_list(self):
        length_list = []
        for f in self.sample_list:
            audio = self.input_load_fn(f)
            length_list.append(audio.shape[-1])
        return length_list

    

    