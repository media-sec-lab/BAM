import torch
import numpy as np
import random
import collections.abc as container_abcs
import re

from torch.utils.data import Sampler

np_str_obj_array_pattern = re.compile(r'[SaUO]')
customize_collate_err_msg = (
    "customize_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

class SamplerBlockShuffleByLen(Sampler):
    """ Sampler with block shuffle based on sequence length
    e.g., data length [1, 2, 3, 4, 5, 6] -> [3,1,2, 6,5,4] if block size =3
    """
    def __init__(self, buf_dataseq_length, batch_size):
        """ SamplerBlockShuffleByLength(buf_dataseq_length, batch_size)
        args
        ----
          buf_dataseq_length: list or np.array of int, 
                              length of each data in a dataset
          batch_size: int, batch_size
        """
        if batch_size == 1:
            mes = "Sampler block shuffle by length requires batch-size>1"
            raise ValueError(mes)

        # hyper-parameter, just let block_size = batch_size * 3
        self.m_block_size = batch_size * 4
        # idx sorted based on sequence length
        self.m_idx = np.argsort(buf_dataseq_length)
        return
    
    def __iter__(self):
        """ Return a iterator to be iterated. 
        """
        tmp_list = list(self.m_idx.copy())

        # shuffle within each block
        # e.g., [1,2,3,4,5,6], block_size=3 -> [3,1,2,5,4,6]
        f_shuffle_in_block_inplace(tmp_list, self.m_block_size)

        # shuffle blocks
        # e.g., [3,1,2,5,4,6], block_size=3 -> [5,4,6,3,1,2]
        f_shuffle_blocks_inplace(tmp_list, self.m_block_size)

        # return a iterator, list is iterable but not a iterator
        # https://www.programiz.com/python-programming/iterator
        return iter(tmp_list)


    def __len__(self):
        """ Sampler requires __len__
        https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
        """
        return len(self.m_idx)

def f_shuffle_in_block_inplace(input_list, block_size):
    """
    f_shuffle_in_block_inplace(input_list, block_size)
    
    Shuffle the input list (in place) by dividing the list input blocks and 
    shuffling within each block
    
    Example:
    >>> data = [1,2,3,4,5,6]
    >>> random_tools.f_shuffle_in_block_inplace(data, 3)
    >>> data
    [3, 1, 2, 5, 4, 6]

    input
    -----
      input_list: input list
      block_size: int
    
    output
    ------
      None: shuffling is done in place
    """
    if block_size <= 1:
        # no need to shuffle if block size if 1
        return
    else:
        list_length = len(input_list)
        # range( -(- x // y) ) -> int(ceil(x / y))
        for iter_idx in range( -(-list_length // block_size) ):
            # shuffle within each block
            f_shuffle_slice_inplace(
                input_list, iter_idx * block_size, (iter_idx+1) * block_size)
        return

def f_shuffle_blocks_inplace(input_list, block_size):
    """ 
    f_shuffle_blocks_inplace(input_list, block_size)
    
    Shuffle the input list (in place) by dividing the list input blocks and 
    shuffling blocks
    
    Example:
     >> data = np.arange(1, 7)
     >> f_shuffle_blocks_inplace(data, 3)
     >> print(data)
     [4 5 6 1 2 3]

    input
    -----
      input_list: input list
      block_size: int
    
    output
    ------
      None: shuffling is done in place
    """
    # new list
    tmp_list = input_list.copy()

    block_number = len(input_list) // block_size
    
    shuffle_block_idx = [x for x in range(block_number)]
    random.shuffle(shuffle_block_idx)  

    new_idx = None
    for iter_idx in range(block_size * block_number):
        block_idx = iter_idx // block_size
        in_block_idx = iter_idx % block_size
        new_idx = shuffle_block_idx[block_idx] * block_size + in_block_idx
        input_list[iter_idx] = tmp_list[new_idx]   # 计算shuffer后与原始顺序索引的对应关系直接赋值
    return

def f_shuffle_slice_inplace(input_list, slice_start=None, slice_stop=None):
    """ shuffle_slice(input_list, slice_start, slice_stop)
    
    Shuffling input list (in place) in the range specified by slice_start
    and slice_stop.

    Based on Knuth shuffling 
    https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle

    input
    -----
      input_list: list
      slice_start: int, start idx of the range to be shuffled
      slice_end: int, end idx of the range to be shuffled
      
      Both slice_start and slice_end should be in the style of python index
      e.g., shuffle_slice(input_list, 0, N) will shuffle the slice input[0:N]
    
      When slice_start / slice_stop is None,
      slice_start = 0 / slice_stop = len(input_list)

    output
    ------
      none: shuffling is done in place
    """
    if slice_start is None or slice_start < 0:
        slice_start = 0 
    if slice_stop is None or slice_stop > len(input_list):
        slice_stop = len(input_list)
        
    idx = slice_start
    while (idx < slice_stop - 1):
	    # 每个元素都与后面的元素随机交换
        idx_swap = random.randrange(idx, slice_stop)
        # naive swap
        tmp = input_list[idx_swap]
        input_list[idx_swap] = input_list[idx]
        input_list[idx] = tmp
        idx += 1
    return



def customize_collate(batch):
    """ customize_collate(batch)
    
    Collate a list of data into batch. Modified from default_collate.
    
    """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        # this is the main part to handle varied length data in a batch
        # batch = [data_tensor_1, data_tensor_2, data_tensor_3 ... ]
        # 
        batch_new = pad_sequence(batch)
        
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy

            # allocate the memory based on maximum numel
            out = torch.stack(batch_new, 0)
            out.share_memory_()
        return out
	
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(customize_collate_err_msg.format(elem.dtype))
            # this will go to loop in the last case
            return customize_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
        
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: customize_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(customize_collate(samples) \
                           for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in batch should be of equal size')
        
        # zip([[A, B, C], [a, b, c]])  -> [[A, a], [B, b], [C, c]]
        transposed = zip(*batch)
        return [customize_collate(samples) for samples in transposed]

    raise TypeError(customize_collate_err_msg.format(elem_type))



def pad_sequence(batch, padding_value=0.0):
    """ pad_sequence(batch)
    
    Pad a sequence of data sequences to be same length.
    Assume batch = [data_1, data2, ...], where data_1 has shape (len, dim, ...)
    
    This function is based on 
    pytorch.org/docs/stable/_modules/torch/nn/utils/rnn.html#pad_sequence
    """
    trailing_dims = tuple(batch[0].size()[:-1])
    max_len = max([s.size(-1) for s in batch])

    if all(x.size(-1) == max_len for x in batch):
        # if all data sequences in batch have the same length, no need to pad
        return batch
    else:
        # we need to pad
        out_dims = trailing_dims + (max_len,)
        
        output_batch = []
        for i, tensor in enumerate(batch):
            # check the rest of dimensions
            if tensor.size()[:-1] != trailing_dims:
                print("Data in batch has different dimensions:")
                for data in batch:
                    print(str(data.size()))
                raise RuntimeError('Fail to create batch data')
            # save padded results
            out_tensor = tensor.new_full(out_dims, padding_value)
            out_tensor[...,:tensor.size(-1)] = tensor
            output_batch.append(out_tensor)
        return output_batch