import torch
from transformers import (WhisperProcessor,
                          WhisperForConditionalGeneration,
                          pipeline)

class HuggingfaceWhisper:
    def __init__(self, model_id, lang, task):
        self.model_id = model_id
        self.model = None
        self.processor = None
        
        self.lang = lang
        self.task = task
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.suppress_tokens = None
    
    def load_model(self):
        if self.model is None:
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
    
    def get_pipeline(self):
        transcribe = pipeline(task="automatic-speech-recognition", model=self.model_id, chunk_length_s=30, device=self.device)
        return transcribe
    
    def get_suppress_tokens(self):
        if self.suppress_tokens is None:
            whisper_smaall = pipeline(task="automatic-speech-recognition", model="openai/whisper-small", chunk_length_s=30, device=self.device)
            suppress_tokens = whisper_smaall.model.config.suppress_tokens
            self.suppress_tokens = suppress_tokens
        return self.suppress_tokens
    
    def transcribe(self, audio = None, audio_file_path = None):
        transcribe = self.get_pipeline()
        suppress_tokens = self.get_suppress_tokens()        
        transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language=self.lang, task= self.task, no_timestamps = False)
        transcribe.model.config.suppress_tokens = suppress_tokens
        transcribe.generation_config.suppress_tokens = suppress_tokens
        transcribe.generation_config.no_timestamps_token_id = transcribe.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        transcripts = transcribe(audio_file_path if audio is None else audio, return_timestamps = True)
        return transcripts
    
