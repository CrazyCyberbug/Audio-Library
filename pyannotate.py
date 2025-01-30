import re
import os
import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from pyannote.audio import Pipeline


def load_audio(audio_path):
    """Loads  the audio into 1D array of  shape [dim,]. of sampling_rate 16000."""
    
    audio, sr = torchaudio.load(audio_path)
    if sr!=16000:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16_000)
    
    audio = audio.numpy()
    if len(audio.shape) == 2 and audio.shape[0] == 2:
        audio = audio.mean(axis=0)
    
    return audio

class Disrize:
    def __init__(self, audio_file_path):
        self.audio_file_path = audio_file_path
        self.auth_token = ""
        self.model_id = "pyannote/speaker-diarization"
        self.model_cache_dir = "pyannotate"
        self.diaristion_text_file_path = "diarization.txt"
        self.pipeline = None
        self.temp_files_dir = "./temp_files"
    
    def prepare_audio(self):
        spacermilli = 2000
        spacer = AudioSegment.silent(duration=spacermilli)
        audio = AudioSegment.from_wav(self.audio_file_path)
        audio = spacer.append(audio, crossfade=0)
        audio.export('input_prep.wav', format='wav')
    
    def load_pipeline(self):
        if self.pipeline == None:
            self.pipeline = Pipeline.from_pretrained(self.model_id,
                                                     use_auth_token = self.auth_token,
                                                     cache_dir =  self.model_cache_dir)
        return self.pipeline
    
    def diarize_audio(self, pipeline):
        pipeline = self.load_pipeline()
        DEMO_FILE = {'uri': 'blabla',
                     'audio': 'input_prep.wav'}
        
        dz = pipeline(DEMO_FILE)
        with open(self.diaristion_text_file_path, "w") as text_file:
            text_file.write(str(dz))
    
    def parse_time_string(self, timeStr: str):
        spl = timeStr.split(":")
        s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
        return s
    
    def group_segments(self):
        dzs = open(self.diaristion_text_file_path).read().splitlines()
        groups = []
        g = []
        lastend = 0

        for d in dzs:
            if g and (g[0].split()[-1] != d.split()[-1]):
                groups.append(g)
                g = []

            g.append(d)
            
            end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
            end = self.parse_time_string(end)
            if (lastend > end):
                groups.append(g)
                g = []
            else:
                lastend = end
        if g:
            groups.append(g)
            
        return groups
    
    def split_and_export_segments(self, groups):
        
        os.makedirs(self.temp_files_dir, exist_ok = True)
        
        audio = AudioSegment.from_wav("input_prep.wav")
        gidx = -1
        for g in groups:
            start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
            end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
            start = self.parse_time_string(start) 
            end = self.parse_time_string(end) 
            gidx += 1
            audio[start:end].export(f"{self.temp_files_dir}/{str(gidx)}.wav", format='wav')
    
    def get_audio_segments(self, groups, sr = 16000):
        audio = load_audio(self.audio_file_path)
        segments = []    
        for g in groups:
            start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
            end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
            start = self.parse_time_string(start) 
            end = self.parse_time_string(end) 
            speaker = g[0].split()[-1]
            start_in_secs = start/1000
            end_in_secs = end/1000
            segments.append({"start" : start_in_secs,
                            "end": end_in_secs,
                            "audio": audio[int(start_in_secs *sr):  int(end_in_secs * sr)],
                            "speaker":speaker})
        return segments
    
    def run(self):        
        self.prepare_audio()
        pipeline = self.load_pipeline()
        self.diarize_audio(pipeline= pipeline)
        groups = self.group_segments()
        self.split_and_export_segments(groups)
        audio_segments = self.get_audio_segments(groups = groups)
        return audio_segments
