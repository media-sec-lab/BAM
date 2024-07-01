import torch
import numpy as np
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_weigth(folder):
    file_list = os.listdir(folder)
    data_list = []
    for f in file_list:
        t = torch.load(os.path.join(folder,f),map_location='cpu')
        t = torch.nn.functional.softmax(t,dim=-1)
        data_list.append(t.unsqueeze(0).detach().numpy())
    
    data = np.concatenate(data_list, axis=0)
    x = np.array(range(len(data_list)))
    colors = np.random.rand(25, 3)

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.plot(x, data[:,i], color=colors[i], label=f'layer {i}')

    plt.legend()
    plt.savefig('out/weight_trend.png')

def speech_and_nospeech_accuracy(pred_path, flag_root, type, resolution):
    pred = pickle_load(pred_path)
    label_file = f'/pubdata/zhongjiafeng/PartialSpoofV1.2/database/segment_labels/{type}_seglab_{resolution}.npy'
    labels = np.load(label_file, allow_pickle=True).item() 
    correct_count = [0,0]
    error_count = [0,0]
    for name, utt_pred in tqdm(pred.items()):
        utt_pred_class = utt_pred.argmax(dim=-1)
        utt_flag = np.load(f'{flag_root}/{name}_vad_label.npy')
        utt_label = labels[name].astype(np.int32)
        for i in range(len(utt_label)):
            if utt_label[i] != utt_pred_class[i]:
                error_count[utt_flag[i]] += 1
            else:
                correct_count[utt_flag[i]] += 1
    
    total_count = list(map(lambda x,y: x+y, correct_count, error_count))
    
    flag_name = ['nonspeech','speech']
    for i,s in enumerate(flag_name):
        acc = correct_count[i] / total_count[i]
        print(f'{s} acc : {acc*100}')
    
    total_acc = sum(correct_count) / sum(total_count)
    print(f'total acc : {total_acc*100}')

def full_and_mix_accuracy(pred_path, flag_root, type, resolution):
    pred = pickle_load(pred_path)
    label_file = f'/pubdata/zhongjiafeng/PartialSpoofV1.2/database/segment_labels/{type}_seglab_{resolution}.npy'
    labels = np.load(label_file, allow_pickle=True).item() 
    correct_count = [0,0,0]
    error_count = [0,0,0]
    for name, utt_pred in tqdm(pred.items()):
        utt_pred_class = utt_pred.argmax(dim=-1)
        utt_flag = np.load(f'{flag_root}/{name}_custom_label.npy')
        utt_label = labels[name].astype(np.int32)
        for i in range(len(utt_label)):
            if utt_label[i] != utt_pred_class[i]:
                error_count[utt_flag[i]] += 1
            else:
                correct_count[utt_flag[i]] += 1
    
    total_count = list(map(lambda x,y: x+y, correct_count, error_count))
    
    flag_name = ['spoof','bonafide','boundary']
    for i,s in enumerate(flag_name):
        acc = correct_count[i] / total_count[i]
        print(f'{s} acc : {acc*100}')
    
    total_acc = sum(correct_count) / sum(total_count)
    print(f'total acc : {total_acc*100}')
    print(f'spoof number : {total_count[0]} bonafide number: {total_count[1]} boundary number: {total_count[2]}')

if __name__ == '__main__':
    pred_path = 'exp/sapd/test/lightning_logs/version_106/test_pred.pkl'
    # speech_and_nospeech_accuracy(
    #     pred_path=pred_path, 
    #     flag_root='/pubdata/zhongjiafeng/PartialSpoofV1.2/database/eval/speech_0.16_label', 
    #     type='eval', 
    #     resolution=0.16)
    full_and_mix_accuracy(
        pred_path=pred_path, 
        flag_root='/pubdata/zhongjiafeng/PartialSpoofV1.2/database/eval/custom_0.16_label', 
        type='eval', 
        resolution=0.16)