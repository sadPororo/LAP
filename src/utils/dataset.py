
""" Dataset utilities/functions in common """
import glob
import copy
import math
import random
import torch
import torchaudio

import numpy as np
import soundfile as sf

from scipy import signal
from os.path import join as opj
from torch.utils.data import Dataset



def load_audio(fpath:str):
    """ load waveform from (.wav) file
    Returns: 
        waveform: np.ndarray (T,)
    """
    waveform, sr = sf.read(fpath)
    assert sr == 16000, f'sr mismatch: {fpath}, sr: {sr} != 16000'
    return waveform.astype(np.float32)


def crop_audio(waveform:np.ndarray, max_length:int):
    """ randomly crop over-length audio 
    Returns:
        waveform: np.ndarray (T,) where T=max_length
    """
    start_index = random.choice(list(range(len(waveform)-max_length+1)))
    return waveform[start_index:start_index+max_length]


def pad_audio(waveform:np.ndarray, max_length:int, mode='wrap'):
    """ pad the shortage of audio sample length
    Returns:
        waveform: np.ndarray (T,) where T=max_length
    """
    return np.pad(waveform, (0, max_length-len(waveform)), mode=mode)


def fit_audio(waveform:np.ndarray, max_length:int, pad_mode='wrap'):
    """ fit audio to given max_length """
    ##____crop overlength audio
    if len(waveform) > max_length: 
        waveform = crop_audio(waveform, max_length)
        
    else: ##____pad audio
        waveform = pad_audio(waveform, max_length, pad_mode)
        
    return waveform


def load_voxceleb_corpus(data_path:str, dataset:str, dev_test:str='dev'):
    """ load voxceleb file paths and speakers """ 
    corpus = []
    with open(f'{data_path}/speakers/{dataset}-{dev_test}.txt', 'r') as f:
        speakers = [line.strip() for line in f.readlines()]
    for spk_id in speakers:
        corpus.extend(glob.glob(f"{data_path}/data/{dataset}/{spk_id}/*/*.wav"))
    return corpus, speakers


def pad_collate_fn(batch):
    """
    batch (list): [(waveform, length, label), ...]
    """
    waveform, length, label = zip(*batch)
    
    if len(batch) > 1:
        max_length = max(length)
        waveform = [fit_audio(w, max_length) for w in waveform]
        
    waveform = torch.from_numpy(np.stack(waveform, axis=0)) # (B, max_length)
    length   = torch.LongTensor(length)
    label    = torch.LongTensor(label)
    
    return waveform, length, label



class Augmentation(object):
    """ waveform augmentation """
    def __init__(self, config:dict):
        self.filelist = { 'rirs': [], 'noise': [], 'music': [], 'speech': [] }
        
        ##____ RIRs
        with open(f'{config.model.rir_path}/RIRS_NOISES/real_rirs_isotropic_noises/rir_list', 'r') as f:
            self.filelist['rirs'].extend([opj(config.model.rir_path, line.strip().split(' ')[-1]) for line in f.readlines()]) # real RIRs
        self.filelist['rirs'].extend(glob.glob(f'{config.model.rir_path}/RIRS_NOISES/simulated_rirs/*/*/*.wav')) # simulated RIRs
        
        ##____ MUSAN
        self.filelist['noise'].extend(glob.glob(f'{config.model.musan_path}/musan_split/noise/*/*/*.wav')) # musan-noise
        self.filelist['music'].extend(glob.glob(f'{config.model.musan_path}/musan_split/music/*/*/*.wav'))        
        self.filelist['speech'].extend(glob.glob(f'{config.model.musan_path}/musan_split/speech/*/*/*.wav'))
        
        ##____ additive noise augmentation configs
        self.noise_cnt = {'noise':[1, 1],   'music':[1, 1],    'speech':[4, 7]}
        self.noise_snr = {'noise':[0, 15],  'music':[5, 15],   'speech':[13, 20]}
        self.noise_dur = {'full' :[1., 1.], 'half' :[0.4, 0.6],'third':[0.2, 0.4]}
        
    def reverberation(self, waveform):
        ##____ load random RIR sample
        rir = load_audio(random.choice(self.filelist['rirs']))
        if len(rir.shape) > 1: # random channel selection
            rir = rir[:, random.randint(0, rir.shape[1]-1)] 
        
        rir = np.expand_dims(rir, 0) # (1, T)
        rir = rir / np.sqrt(np.sum(rir**2))

        return signal.convolve(np.expand_dims(waveform, 0), rir, mode='full').squeeze(0)[:len(waveform)]
    
    def add_noise(self, waveform, noise_type:str, duration_range:str=None):
        clean_db = 10 * np.log10(np.mean(waveform ** 2)+1e-4)
                
        noises = []
        for _ in range(random.randint(self.noise_cnt[noise_type][0], self.noise_cnt[noise_type][1])):
            ##____ load random noise audio file
            noise_audio = load_audio(random.choice(self.filelist[noise_type]))
            if len(noise_audio.shape) > 1: # random channel selection on multi-channel mic array
                noise_audio = noise_audio[:, random.randint(0, noise_audio.shape[1]-1)] 
                
            if duration_range == None: # select duration
                duration_range = random.choice(list(self.noise_dur.keys()))
            duration  = int(random.uniform(self.noise_dur[duration_range][0], self.noise_dur[duration_range][1]) * len(waveform))
            noise_audio = fit_audio(noise_audio, duration)
            
            ##____ set noise db
            snr      = random.uniform(self.noise_snr[noise_type][0], self.noise_snr[noise_type][1])
            noise_db = 10 * np.log10(np.mean(noise_audio ** 2)+1e-4)
            
            ##____ stack noise
            add_noise = np.zeros_like(waveform)
            start_idx = random.choice(list(range(len(waveform)-duration+1)))
            add_noise[start_idx:start_idx+duration] = noise_audio
            
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * add_noise)
        
        ##____ apply noises
        return np.stack(noises, axis=0).sum(axis=0) + waveform
    
    def speed_perturbation(self, waveform, sample_rate:int, factor:float):
        """ 
        code from: https://github.com/espnet/espnet/blob/master/espnet2/layers/augmentation.py#L294
        """
        orig_freq = sample_rate
        source_sample_rate = int(factor * orig_freq)
        target_sample_rate = int(orig_freq)

        gcd = math.gcd(source_sample_rate, target_sample_rate)
        source_sample_rate = source_sample_rate // gcd
        target_sample_rate = target_sample_rate // gcd

        ret = torchaudio.functional.resample(
            waveform, source_sample_rate, target_sample_rate
        )
        return ret



class TrainDataset(Dataset):
    """ Speaker classification Dataset """
    def __init__(self, config:dict, utterance_corpus:list, speaker_list:list):
        self.sr = config.model.sample_rate
        self.max_sec = config.model.train_max_duration
        self.max_len = int(self.sr * self.max_sec)
        self.speed_perturb = config.model.speed_perturb
        self.training_set = config.model.training_set
        
        self.train_corpus   = utterance_corpus
        self.train_speakers = speaker_list
        self.train_corpus.sort()
        
        ##____ apply speed perturbation (~offline augmentation)
        self.speed_perturb = config.model.speed_perturb
        unique_factors = set(self.speed_perturb)
        unique_factors.remove(1.0)
        unique_factors = [1.0] + list(unique_factors) # make offset 0 for the original file
        self.speed_factor_to_label_offset = {factor: offset for offset, factor in enumerate(unique_factors)}
                
        ##____ noise/rir augmentation
        self.augment = Augmentation(config)
        self.p_augment = config.model.p_augment
                
        ##____ if 'n_iter_per_epoch' specified
        self.total_batch_size = config.model.batch_size * config.model.grad_acc # total_batch_size
        self.n_iter_per_epoch = config.model.n_iter_per_epoch
        self.train_sampler = []
        self.generate_iteration(config.model.seed)
                
    def n_speakers(self):
        return len(self.train_speakers)
    
    def n_utterances(self):
        return len(self.train_corpus)
    
    def n_speed_perturbation(self):
        return len(self.speed_perturb)
    
    def generate_iteration(self, rng_state:int):
        if self.n_iter_per_epoch > 0:
            target_length = self.total_batch_size * self.n_iter_per_epoch
            
            self.train_iterator = []
            while len(self.train_iterator) < target_length:
                if len(self.train_sampler) == 0:
                    self.train_sampler = copy.deepcopy(self.train_corpus)
                    np.random.RandomState(rng_state).shuffle(self.train_sampler)
                    
                i = min(len(self.train_sampler), target_length-len(self.train_iterator))
                self.train_iterator.extend(self.train_sampler[:i])
                self.train_sampler = self.train_sampler[i:]
            
        else:
            self.train_iterator = self.train_corpus
        
    def __len__(self):
        return len(self.train_iterator)

    def __getitem__(self, index):
        fpath = self.train_iterator[index]
        
        ##____ load audio and index speaker label
        waveform = load_audio(fpath)
        label    = self.train_speakers.index(fpath.split('/')[-3])
        
        ##____ apply speed perturbation
        speed_factor = random.choice(self.speed_perturb)
        if speed_factor != 1.0:
            waveform = self.augment.speed_perturbation(torch.from_numpy(waveform), sample_rate=self.sr, factor=speed_factor).numpy()
        label = self.n_speakers() * self.speed_factor_to_label_offset[speed_factor] + label
        
        ##____ fit audio to the 'train_max_duration'
        waveform = fit_audio(waveform, self.max_len)
        
        ##____ apply noise/rir augmentation
        if random.random() < self.p_augment:
            augtype = random.randint(0, 4)
            if augtype == 0: # Reverb
                waveform = self.augment.reverberation(waveform)
            elif augtype == 1: # Noise
                waveform = self.augment.add_noise(waveform, 'noise')
            elif augtype == 2: # Music
                waveform = self.augment.add_noise(waveform, 'music')
            elif augtype == 3: # Babble
                waveform = self.augment.add_noise(waveform, 'speech')
            elif augtype == 4: # Television
                waveform = self.augment.add_noise(waveform, 'speech', 'full')
                waveform = self.augment.add_noise(waveform, 'music', 'full')
        
        return waveform, len(waveform), label
            


class EmbeddingExtractionDataset(Dataset):
    """ Imposter cohort dataset """
    def __init__(self, utterance_corpus:list, speaker_list:list, sample_rate:int=16000, max_duration:int=None):
        
        self.evaluation_corpus = utterance_corpus
        self.evaluation_speakers = speaker_list
        
        self.sr = sample_rate
        self.max_sec = max_duration
        self.max_len = int(self.max_sec * self.sr) if max_duration is not None else None

    def idx2spk(self, index:int):
        return self.evaluation_speakers[index]
    
    def spk2idx(self, speaker_id:str):
        return self.evaluation_speakers.index(speaker_id)
    
    def n_speakers(self):
        return len(self.evaluation_speakers)
    
    def n_utterances(self):
        return len(self.evaluation_corpus)
    
    def max_duration(self):
        if self.max_sec is None:
                return '--'
        else:   return self.max_sec
            
    def __len__(self):
        return len(self.evaluation_corpus)
    
    def __getitem__(self, index):
        """ 
        Returns:
            waveform: torch.Tensor
            ipath: list[str]
            speaker_id: list[str]
            label: torch.LongTensor
        """
        fpath = self.evaluation_corpus[index]
        ipath = '/'.join(fpath.split('/')[-3:])
        
        ##____ load audio
        waveform = load_audio(fpath)
        length   = len(waveform)
        
        if self.max_len is not None:
            if len(waveform) > self.max_len:
                waveform = crop_audio(waveform, self.max_len)
            else:
                waveform = pad_audio(waveform, self.max_len)
        
        ##____ speaker index
        label = self.evaluation_speakers.index(fpath.split('/')[-3])
        
        return waveform, length, ipath, label

            
        