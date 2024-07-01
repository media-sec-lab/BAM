import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import soundfile as sf
import numpy as np 
import random
import torch
import importlib
import glob

from tqdm import tqdm
from collections import OrderedDict
from sklearn.manifold import TSNE
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from utils import *

def display_vad_result(path,vad_result,save_dir):
    """
        vad_result: [(speech_1_start,speech_1_end),(speech_2_start,speech_2_end),...]
    """
    data, fs = sf.read(path)
    # plot time waveform
    plt.figure(figsize=(10,4))
    time_axis = np.linspace(0, len(data)/fs, len(data))
    plt.plot(time_axis,data, color='b')

    for regoin in vad_result:
        start, end = [i/fs for i in regoin]
        rect = Rectangle((start, min(data)), (end - start),max(data)-min(data),linewidth=2, facecolor='red',alpha=0.1)   
        plt.gca().add_patch(rect)
        plt.text(start, max(data),'clip',fontsize=10, color="red", weight="bold")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.tight_layout()

    utt = (path.strip().split('/'))[-1].split('.')[0]
    save_path = os.path.join(save_dir,f'{utt}_vad.png')
    plt.savefig(save_path)
    plt.close()

def display_parital_spoof_wavform(path,resolution_labels,absolute_labels_dir,save_dir,resolution, ret=False, pred=None):
    """
    @Description :
        plot audio wavform with segment-level label. 
    @Inputs      :
        protocol_abspath: str, protocol file path.
        root: str, corresponding data path of protocol file.
        seglab_abspath: str, segment label .npy file path.
        resolution: int, time resolution in number of sample points.
        visualize_num: int, how many samples want to display.
        store:   str, to save in which folder.
        utt_index: int , the colum index in protocol .txt file.
    """

    data, sr = sf.read(path)
    name = os.path.basename(path).split('.')[0]
    frame_len = sr * resolution
    data = np.pad(data, (0,int(frame_len-len(data)%frame_len)), 'constant', constant_values=0)
    # plot time waveform
    fig, ax2 = plt.subplots(1,1,figsize=(40,6))
    time_axis = np.linspace(0, len(data)/sr, len(data))

    # # resolution plot
    # start = 0
    # scale = len(data) / len(resolution_labels)
    # for idx, label in enumerate(resolution_labels):
    #     if idx == 0:
    #         last = label
    #     if label != last:
    #         flag = 'bonafide' if last == 1 else 'spoof'
    #         color = 'green' if flag=='bonafide' else 'red'
    #         next_start = int(idx * scale)
    #         ax1.plot(time_axis[start:next_start], data[start:next_start], color=color, label=flag, linewidth=1)
    #         start = next_start
    #         last = label

    # for i in range(len(resolution_labels)+1):
    #     ax1.axvline(x=i*resolution, linestyle='--', color='b', alpha=0.5)

    # if pred is not None:
    #     pred_class = torch.argmax(pred,dim=-1).numpy()
    #     for i in range(len(resolution_labels)):
    #         if pred_class[i] == resolution_labels[i]:
    #             continue
    #         else:
    #             s, e = i*resolution, (i+1)*resolution
    #             rect = Rectangle((s, min(data)), (e - s),max(data)-min(data),linewidth=2, facecolor='red',alpha=0.1)   
    #             ax1.add_patch(rect)

    # flag = 'bonafide' if resolution_labels[-1] == 1 else 'spoof'
    # color = 'green' if flag=='bonafide' else 'red'
    # ax1.plot(time_axis[start:], data[start:], color=color, label=flag, linewidth=1)
    # ax1.set_title(f'{resolution} resolution time waveform')
    # handles, labels = ax1.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax1.legend(by_label.values(), by_label.keys())

    
    # absolute time
    base_flag = resolution_labels[0]
    color_list = ['red','green']
    flag_list = ['spoof','bonafide']
    label_path = f'{absolute_labels_dir}/{name}.vad'
    with open(label_path,'r') as f:
        lines = f.readlines()

    for line in lines:
        st, et, idx = line.strip().split(' ')
        st, et = int(float(st)*sr), int(float(et)*sr)
        flag_idx = int(idx) if int(idx)!=2 else base_flag
        flag = flag_list[flag_idx]
        color = color_list[flag_idx]
        ax2.plot(time_axis[st:et], data[st:et], color=color, label=flag,  linewidth=1)
    
    for i in range(len(resolution_labels)+1):
        ax2.axvline(x=i*resolution, linestyle='--', color='b', alpha=0.5)

    # ax2.set_title("absolute time waveform")

    # handles, labels = ax2.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax2.legend(by_label.values(), by_label.keys())
    plt.axis('off')

    # if ret:
    #     return fig, (ax1, ax2)

    utt_save_name = f'{name}.png'
    utt_save_abspath = os.path.join(save_dir, utt_save_name)
    fig.savefig(utt_save_abspath, dpi=500, bbox_inches='tight')
    plt.close()

class BaseTSNE():
    def __init__(self, name, ckpt, save_dir, type) -> None:
        self.save_dir = save_dir
        self.type = type
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        module_name = '.'.join(name.split('.')[:-1])
        cls_name = name.split('.')[-1]
        model_class = getattr(importlib.import_module(module_name),cls_name)

        self.lightning_model = model_class.load_from_checkpoint(ckpt, map_location='cpu', strict=True)
        self.lightning_model.to(self.device)
        self.lightning_model.eval()

    def utterence_display(self, path, label):
        utt_name = os.path.basename(path).split('.')[0]
        data = torch.load(path,map_location='cpu').to(self.device)
        scale = int(0.16 // 0.02)
        data = torch.nn.functional.pad(data,(0,0,0,len(label)*scale-data.size(1)))
        with torch.no_grad():
            output ,embedding = self.lightning_model.model(data)
        self.display(embedding.squeeze().detach().cpu().numpy(), label, utt_name)

    def batch_display(self,utt_list,num=200):
        embedding_list = []
        label_list = []
        for utt in tqdm(utt_list[:num]):
            name = os.path.basename(utt).split('.')[0]
            data, sr = sf.read(utt)
            data = torch.FloatTensor(data).to(self.device)
            label = np.load(f'/pubdata/zhongjiafeng/PartialSpoofV1.2/database/{self.type}/custom_0.16_label/{name}_custom_label.npy')
            scale = int(0.16 * sr)
            data = torch.nn.functional.pad(data,(0,len(label)*scale-data.size(-1)))
            data = data.unsqueeze(0)   # (1, time)
            with torch.no_grad():
                output, _, s_out, embedding = self.lightning_model.model(data)
            embedding_list.append(embedding.squeeze().detach().cpu())
            label_list.extend(label)
        embedding_all = torch.cat(embedding_list, dim=0)
        self.display2D(embedding_all, label_list, 'Batch')

    def display2D(self, embedding, label, utt_name):
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.numpy()
        
        tsne_result = TSNE(n_components=2).fit_transform(embedding)
        x_min, x_max = np.min(tsne_result,0), np.max(tsne_result,0)
        tsne_result = (tsne_result - x_min)/(x_max-x_min)
        colors = ['red','green','blue']    # set as the number of speaker id.
        flag = ['spoof','bonafide','combine']
        markers = ['o','^','s','p','*','+','x','D']
        plt.figure(figsize=(10,4))
        for i in tqdm(range(tsne_result.shape[0]),desc='plotting...'):
            plt.scatter(tsne_result[i,0], tsne_result[i,1],marker=markers[label[i].item()],s=5,
                            c=colors[label[i].item()], label=flag[label[i].item()])
        plt.xticks([])
        plt.yticks([])
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(f'{self.save_dir}/{utt_name}_t_sne.png',dpi=500)
        plt.close()
    
    def display3D(self, embedding, label, utt_name):
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.numpy()
        
        tsne_result = TSNE(n_components=3).fit_transform(embedding)
        x_min, x_max = np.min(tsne_result,0), np.max(tsne_result,0)
        tsne_result = (tsne_result - x_min)/(x_max-x_min)
        colors = ['red','green','blue']    # set as the number of speaker id.
        flag = ['spoof','bonafide','conbine']
        markers = ['o','^','s','p','*','+','x','D']
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(tsne_result.shape[0]):
            ax.scatter(tsne_result[i,0], tsne_result[i,1],tsne_result[i,2],marker=markers[label[i].item()],s=5,
                            c=colors[label[i].item()], label=flag[label[i].item()])

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(f'{self.save_dir}/{utt_name}_t_sne.png',dpi=500)
        plt.close()

    @staticmethod
    def get_colortable(n):
        colors = list(mcolors.CSS4_COLORS)
        sample_colors = random.sample(colors,k=n)
        return sample_colors 

def partialspoof_display(name,type='train',resolution=0.16):
    path = f'/pubdata/zhongjiafeng/PartialSpoofV1.2/database/{type}/con_wav/{name}.wav'
    save_dir = 'out'
    label_file = f'/pubdata/zhongjiafeng/PartialSpoofV1.2/database/segment_labels/{type}_seglab_{resolution}.npy'
    labels = np.load(label_file, allow_pickle=True).item() 
    utt_label = labels[name].astype(np.int32)
    absolute_labels_dir = f'/pubdata/zhongjiafeng/PartialSpoofV1.2/database/vad/{type}'
    display_parital_spoof_wavform(path,utt_label,absolute_labels_dir,save_dir,resolution=resolution)

def partialspoof_display_with_pred(name,pred,type='train',resolution=0.16):
    path = f'/pubdata/zhongjiafeng/PartialSpoofV1.2/database/{type}/con_wav/{name}.wav'
    save_dir = 'out/sample'
    label_file = f'/pubdata/zhongjiafeng/PartialSpoofV1.2/database/segment_labels/{type}_seglab_{resolution}.npy'
    labels = np.load(label_file, allow_pickle=True).item() 
    utt_label = labels[name].astype(np.int32)
    absolute_labels_dir = f'/pubdata/zhongjiafeng/PartialSpoofV1.2/database/vad/{type}'
    display_parital_spoof_wavform(path,utt_label,absolute_labels_dir,save_dir,resolution=resolution, pred=pred)

def partialspoof_boundary_display(name,preds_root,type='train',resolution=0.16):
    path = f'/pubdata/zhongjiafeng/PartialSpoofV1.2/database/{type}/con_wav/{name}.wav'
    save_dir = 'out'
    label_file = f'/pubdata/zhongjiafeng/PartialSpoofV1.2/database/segment_labels/{type}_seglab_{resolution}.npy'
    labels = np.load(label_file, allow_pickle=True).item() 
    utt_label = labels[name].astype(np.int32)
    utt_preds = pickle_load(preds_root)
    preds = utt_preds[name].tolist()
    absolute_labels_dir = f'/pubdata/zhongjiafeng/PartialSpoofV1.2/database/vad/{type}'

    fig, (ax1, ax2) = display_parital_spoof_wavform(path,utt_label,absolute_labels_dir,save_dir,resolution=resolution, ret=True)
    display_boundary_prediction(name, fig, ax1, preds, resolution=resolution, samplerate=16000, save_dir=save_dir)

def take_look_on_wrongest_sample(utt_log_path,type,pred_path=None,num=300, reverse=False):
    with open(utt_log_path,'r') as f:
        lines = f.readlines()
    if reverse:
        lines.reverse()
    display_list = lines[:num]
    if pred_path is not None:
        pred_dict = pickle_load(pred_path)

    for i,line in enumerate(tqdm(display_list)):
        meta = line.strip().split(' ')
        utt_name = meta[0].split(":")[-1]
        partialspoof_display_with_pred(name=utt_name,type=type,pred=pred_dict[utt_name])

def display_badsample_energy(pred_path,type,resolution):
    pred_dict = pickle_load(pred_path)
    wav_root = f'/pubdata/zhongjiafeng/PartialSpoofV1.2/database/{type}/con_wav'
    label_file = f'/pubdata/zhongjiafeng/PartialSpoofV1.2/database/segment_labels/{type}_seglab_{resolution}.npy'
    labels = np.load(label_file, allow_pickle=True).item() 
    wav_list = glob.glob(f'{wav_root}/**/*.wav',recursive=True)
    bad_energy = []
    good_energy = []
    frame_count = 0
    for utt in tqdm(wav_list):
        name = os.path.basename(utt).split('.')[0]
        utt_pred = torch.argmax(pred_dict[name],dim=-1).numpy()
        utt_label = labels[name].astype(np.int32)
        data, sr = sf.read(utt)
        frame_count += len(utt_label)
        for i in range(len(utt_label)):
            s, e = int(resolution * sr*i), int(resolution * sr*(i+1))
            frame_energy = np.sum(data[s:e] ** 2)
            if frame_energy > 1.0:
                continue
            if utt_pred[i] != utt_label[i]:
                bad_energy.append(frame_energy)
            else:
                good_energy.append(frame_energy)

    print(f'total frame number is : {frame_count}')
    print(f'total bad frame number is : {len(bad_energy)}')
    plt.figure(figsize=(20,5))
    plt.hist([bad_energy, good_energy],bins=3000, edgecolor='black', color=['red','green'],stacked=True)
    plt.xlim(0, 0.05)
    plt.xticks(np.arange(0, 0.05, 0.005))
    plt.savefig('out/badsample_energy.png', dpi=500)
                
def display_boundary_prediction(name, fig, ax1, preds, resolution, samplerate, save_dir):
    # plot time waveform
    scale = int(resolution * samplerate)
    time_axis = np.linspace(0, len(preds)*resolution, int(len(preds)*scale))

    for idx, pred in enumerate(preds):
        start = int(scale * idx)
        end = int(start + scale)
        ax1.plot(time_axis[start:end], np.full_like(time_axis[start:end],fill_value=pred), color='blue', label='pred_boundary', linewidth=1)

    ax1.plot(time_axis, np.full_like(time_axis,fill_value=0.5), color='yellow', label='threshold_boundary', linewidth=1)
    plt.ylim(-1.0, 1.0)

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    
    utt_save_name = f'{name}_boundary_pred.png'
    utt_save_abspath = os.path.join(save_dir, utt_save_name)
    fig.savefig(utt_save_abspath, dpi=500, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    partialspoof_display(type='train',name='CON_T_0010830',resolution=0.08)
    # preds_root = 'exp/boundary/lightning_logs/version_3/test_utt_preds.pkl'
    # partialspoof_boundary_display('CON_E_0047977',preds_root,type='eval')
    # take_look_on_wrongest_sample('exp/sapd/lightning_logs/version_4/test_utt.log',type='eval',
    #     pred_path='exp/sapd/lightning_logs/version_4/test_pred.pkl')

    # displayer = BaseTSNE(
    #     name='train.sapd_train.Lighting_Model_Wrapper', 
    #     ckpt='exp/sapd/lightning_logs/version_3/checkpoints/sample-mnist-epoch=14-validate_loss=0.331619.ckpt', 
    #     save_dir='out',
    #     type='eval'
    #     )
    # utt_path_list = glob.glob('data/raw/eval/*.wav')
    # displayer.batch_display(utt_path_list,num=500)
    
    # display_badsample_energy(pred_path='exp/sapd/lightning_logs/version_4/test_pred.pkl',type='eval',resolution=0.16)
    print('define of display function.')

