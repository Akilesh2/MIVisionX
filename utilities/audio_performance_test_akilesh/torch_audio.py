
import numpy as np
import scipy.io.wavfile
import torch
import torchaudio
from torch.optim import Optimizer
import time
import os
import timeit
from tqdm import tqdm
#change the folder_path 
folder_path1 = '/media/akilesh/audio/sample_audio/'

file_list = os.listdir(folder_path1)
tot_time = 0
for i in tqdm(range(100)):
    for f in file_list:
        filename = folder_path1+f
        waveform,sample_rate = torchaudio.load(filename)
        start = timeit. default_timer()
        # spectro = torchaudio.transforms.Spectrogram(n_fft=512,win_length=512,center =True,power=2)
        todecible = torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=80)
    
        # spectro(waveform)
        todecible(waveform)
        tot_time = tot_time + (timeit. default_timer()-start)
        
print("Time in microseconds ",(tot_time/100) * 1000000)