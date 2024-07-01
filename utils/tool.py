import os
import fnmatch
import random
import torch
import numpy as np 
import subprocess
import pickle
import shutil
import importlib

class Attribution_Config:
    def __init__(self, **entries) -> None:
        self.__dict__.update(entries)

def import_class_from_path(target_class,target_path):
    cls_name = target_class.split('.')[-1]
    spec = importlib.util.spec_from_file_location('model', target_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package_path = module.__file__
    cls = getattr(module,cls_name)
    return cls, package_path

def import_class(target_class):
    cls_name = target_class.split('.')[-1]
    module = importlib.import_module('.'.join(target_class.split('.')[:-1]))
    cls = getattr(module,cls_name)
    package_path = module.__file__
    return cls, package_path

def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files

def setup_seed(random_seed, cudnn_deterministic=True):
    """ set_random_seed(random_seed, cudnn_deterministic=True)

    Set the random_seed for numpy, python, and cudnn

    input
    -----
      random_seed: integer random seed
      cudnn_deterministic: for torch.backends.cudnn.deterministic

    Note: this default configuration may result in RuntimeError
    see https://pytorch.org/docs/stable/notes/randomness.html
    """

    # # initialization
    # torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False

def get_src_mask(label,length=None):
    """
    length : (bs, 1 )
    label : (bs, frames_num)
    """
    mask = torch.ones_like(label)
    if len(mask.size()) > 2:
        mask = mask[:,:,0]
    if length is None:
        return mask
    for i,l in enumerate(length):
        mask[i][int(l):] = 0.
    return mask

def cut_according_length(preds, labels, lengths):
    pred_list = []
    label_list = []
    for pred, label, length in zip(preds, labels, lengths):
        pred_list.append(pred[:length,...].tolist())
        label_list.append(label[:length].tolist())
    return pred_list, label_list

def freeze_model(model):
    for param in model.parameters():
        param.requireds_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requireds_grad = True



################################### I/O operation fcuntion

def link_file(from_file, to_file):
    subprocess.check_call(
        f'ln -s "`realpath --relative-to="{os.path.dirname(to_file)}" "{from_file}"`" "{to_file}"', shell=True)

def move_file(from_file, to_file):
    subprocess.check_call(f'mv "{from_file}" "{to_file}"', shell=True)


def copy_file(from_file, to_file):
    subprocess.check_call(f'cp -r "{from_file}" "{to_file}"', shell=True)


def remove_file(*fns):
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)

def save_running_script(src, target):
    shutil.copyfile(src, target)
    

def pickle_save(object, path):
    with open(path, 'wb') as file:
        pickle.dump(object, file)

def pickle_load(path):
    with open(path,'rb') as file:
        object = pickle.load(file)
    return object


############################ other #####################
def operation_with_dim(input, dim, op='sum'):
    all_dim = input.size(-1)
    keep_dim = [i for i in range(all_dim) if i not in dim]
    sum_list = []
    for d in dim:
        sum_list.append(input[...,d])
    sum_tensor = torch.stack(sum_list,dim=-1)
    if op =='sum':
        sum_tensor = torch.sum(sum_tensor, dim=-1,keepdim=True)
    elif op=='mean':
        sum_tensor = torch.mean(sum_tensor, dim=-1,keepdim=True)

    keep_list = []
    for d in keep_dim:
        keep_list.append(input[...,d])
    keep_tensor = torch.stack(keep_list,dim=-1)
    
    output = torch.cat([sum_tensor,keep_tensor],dim=-1)
    return output
