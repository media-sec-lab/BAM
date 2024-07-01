import glob
import os
from tqdm import tqdm


def resample(in_dir, out_dir, samplerate, extend='wav'):
    os.makedirs(out_dir,exist_ok=True)
    file_list = glob.glob(f'{in_dir}/**/*.{extend}',recursive=True)

    for path in tqdm(file_list):
        name = os.path.basename(path).split('.')[0]
        speaker_dir = os.path.join(out_dir,os.path.dirname(path).split('/')[-1])
        os.makedirs(speaker_dir,exist_ok=True)
        sp = os.path.join(speaker_dir,f'{name}.wav')
        command = f'ffmpeg -loglevel quiet -i {path} -ar {samplerate} {sp}'
        os.system(command)

if __name__ == '__main__':
    resample(
        in_dir='/pubdata/zhongjiafeng/AISHELL3/train',
        out_dir='/pubdata/zhongjiafeng/data/raw/aishell3',
        samplerate=16000
    )