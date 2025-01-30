import os
import torch
import numpy as np
import PyPDF2
import tempfile
import markdown

from time import time
from typing import Union, List
from datetime import datetime
from markdown import markdown
from weasyprint import HTML
from enum import Enum
import torchaudio

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import (AutoTokenizer,
                          AutoModel,
                          AutoModelForSeq2SeqLM,
                          VitsModel,
                          AutoProcessor,
                          SeamlessM4Tv2Model,
                          SeamlessM4TModel)

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat



class HuggingFaceModelID(Enum):
    SEAMLESS_LARGE = "facebook/seamless-m4t-v2-large"
    SEAMLESS_MEDIUM = "facebook/seamless-m4t-medium"
    SEAMLESS_MYA_FINETUNED = "/home/SWATHI/New_Prasikshan_MV/checkpoint-8000"
    NLLB = "facebook/nllb-200-distilled-1.3B"
    VITS_KAZ_TTS = "facebook/mms-tts-kaz"
    VITS_MYA_TTS = "facebook/mms-tts-mya"
    OPUS_FRA = "Helsinki-NLP/opus-mt-fr-en"
    
class SeamlessModel():
    def __init__(self, model_id):
        self.model_id = model_id
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = SeamlessM4Tv2Model.from_pretrained(self.model_id).to(self.device)
        model = model.to(self.device)
        self.processor = processor
        self.model = model    
        
    def delete_model(self):
        if self.model is not None:
            self.model.to('cpu')
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            del self.processor
            self.processor = None
            
    def translate(self, tgt_lang: str, src_lang: str = None, audio: Union[np.ndarray, List[np.ndarray]] = None,
                text: Union[List[str], str] = None, generate_speech: bool = False,
                generate_speech_and_text: bool = False, return_text_as_list:bool = False):
        
        """
        function translates text or audio to text in 
        args:
        tgt_lang: target language
        audio (Optional): array of audio sample to be translated.
        Text(Optional): text to be translated.
        
        """
                
        if audio is None  and text is None:
             raise ValueError("You have to specify either text or audios. Both cannot be none.")
        elif text is not None and audio is not None:
            raise ValueError(
                "Text and audios are mututally exclusive when passed to `SeamlessM4T`. Specify one or another."
                )
                    
        else:
            
            if self.model is None:
                self.load_model()
            
            if  audio is not None and type(audio) == list:
                audio = np.vstack(audio)
                        
            input_tokens = self.processor(text = text, audios = audio, src_lang = src_lang, return_tensors="pt", sampling_rate = 16000).to(self.device)
            
            # if generate_speech_and_text is set to True  model returns both speech and text.
            if generate_speech_and_text:
                output_tokens = self.model.generate( **input_tokens,
                                                tgt_lang= tgt_lang,
                                                return_intermediate_token_ids = True)
                
                # handling batched inputs
                    
                texts = [self.processor.decode(sequence.cpu().tolist(),skip_special_tokens=True)
                            for sequence in output_tokens.sequences]
                
                waveforms = [waveform.cpu().numpy().flatten()
                                for waveform in output_tokens.waveform]
                                    
                translated_audio = np.concatenate(waveforms) if len(waveforms) > 1 else waveforms[0]
                translated_texts = " ".join(texts)
                                        
                return {"text": translated_texts, "audio": translated_audio}
            
            # if generate_speech is set to True  the model returns only speech.
            elif generate_speech:
                output_tokens = self.model.generate( **input_tokens,
                                                tgt_lang = tgt_lang,
                                                generate_speech = True)
                
                waveforms = [waveform.cpu().numpy().flatten()
                                for waveform in output_tokens[0]]
                
                translated_audio = np.concatenate(waveforms) if len(waveforms) > 1 else waveforms[0]
                
                return {"audio": translated_audio}
            
            # if both are set to false the model returns only text.
            else:
                output_tokens = self.model.generate( **input_tokens,
                                                tgt_lang = tgt_lang,
                                                generate_speech = False)
                
                # handling batched inputs                    
                texts = [self.processor.decode(sequence.cpu().tolist(),skip_special_tokens=True)
                            for sequence in output_tokens.sequences]                                        

                translated_texts = texts if return_text_as_list else " ".join(texts) 
                                        
                return {"text": translated_texts}

class VitsModel():
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    def load_model(self,):
        self.model = VitsModel.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model.to(self.device)
        
    def delete_model(self):
        if self.model is not None:
            self.model.to('cpu')
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            del self.tokenizer
            self.tokenizer = None
    
    def generate_speech(self, text: Union[str, list[str]]):
        if self.model is None:
            self.load_model()
            
        
        if type(text) == str:
            text = [text,]
        
        generated_waveforms = [] 
        for t in text:    
            inputs = self.tokenizer(t, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model(**inputs).waveform
            waveform  = output.squeeze().cpu().numpy()
            generated_waveforms.append(waveform)
            
        waveform = np.concatenate(generated_waveforms)
        return waveform
                  
class PdfDocumentHandler():
    def __init__(self):
        self.filename = None    
        
    def extract_text(self, file_path):
        try:
            # Case 1: File-like object with a 'read' method
            if hasattr(file_path, 'read'):
                pdf_reader = PyPDF2.PdfReader(file_path)
            else:  # Case 2: File path as a string
                with open(file_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)

                    # Extract text while the file is open
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def split_text(self, text, chunk_size = 520, overlap = 0):
        text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=overlap,
                        )
        chunks = [i.page_content for i in text_splitter.create_documents([text,])]
        return chunks
    
    def get_chunks(self, file_path):
        text =  self.extract_text(file_path)
        chunks = self.split_text(text)
        return chunks       
       
class TranslationHandler():
    def __init__(self):
        self.pdf_handlder = PdfDocumentHandler()
        self.seamless_model = None
        self.vits_kaz = None
        self.vits_mya = None
    
    def text_to_text(self, text: Union[str, list[str]], tgt_lang:str, src_lang:str = None):
                
        if type(text) == str and len(text) > 512:
            text = self.pdf_handlder.split_text(text = text)
         
          
        seamless_model = self.load_model(HuggingFaceModelID.SEAMLESS_LARGE)
        translated_text = seamless_model.translate(text = text,
                                                   tgt_lang = tgt_lang,
                                                   src_lang = src_lang)["text"]
        return {"text": translated_text}
    
    def text_to_speech(self, text: Union[str, list[str]], tgt_lang: str):
        if len(text) > 512 and type(text) == str:
            pdf_handler = self.pdf_handlder
            text = pdf_handler.split_text(text = text)
            
        seamless_model = self.load_model(HuggingFaceModelID.SEAMLESS_LARGE)           
        if tgt_lang not in['mya', 'kaz']:
            audio = seamless_model.translate(text = text, tgt_lang=tgt_lang,
                                                generate_speech = True)["audio"]
            
            return {"audio": audio}
        
        # use tts model for mya and kaz.
        else:
            
            if tgt_lang == "kaz":
                tts_model = self.load_model(HuggingFaceModelID.VITS_KAZ_TTS)
                seamless_model = self.load_model(HuggingFaceModelID.SEAMLESS_LARGE)
                
                text = seamless_model.translate(text = text,
                                                tgt_lang= tgt_lang,
                                                return_text_as_list= True)
                
                audio = tts_model.generate_speech(text)
                return {"audio": audio}
                

            elif tgt_lang == "mya":
                tts_model = self.load_model(HuggingFaceModelID.VITS_MYA_TTS)                
                text = seamless_model.translate(text = text,
                                                tgt_lang= tgt_lang,
                                                return_text_as_list= True)
                
                audio = tts_model.generate_speech(text)
                return {"audio": audio}
                
    def translate_pdf(self, file_path,tgt_lang, generate_speech = False):
        text = self.pdf_handlder.get_chunks(file_path = file_path)
        
        if generate_speech:
            speech = self.text_to_speech(text = text,
                                         tgt_lang = tgt_lang)["audio"]
            return {"audio": speech}
        
        else:
            text = self.text_to_text(text = text,
                                     tgt_lang = tgt_lang)["text"]
            return {"text": text}
    
    def load_model(self, model_id):
        if model_id == HuggingFaceModelID.SEAMLESS_LARGE:
            if self.seamless_model is None:
                seamless_model =  SeamlessModel(model_id =  model_id.value)
                self.seamless_model = seamless_model
            return self.seamless_model
            
        if model_id == HuggingFaceModelID.VITS_KAZ_TTS:
            if self.vits_kaz is None:
                vits_kaz = VitsModel(model_id = model_id.value)
                self.vits_kaz = vits_kaz
            return self.vits_kaz
        
        if model_id == HuggingFaceModelID.VITS_MYA_TTS:
            if self.vits_mya is None:
                vits_mya = VitsModel(model_id = model_id.value)
                self.vits_mya = vits_mya
            return self.vits_mya
        
        else:
            raise Exception("Recieved invalid model id.")
          
    def delete_model(self, model_id =None , clear_all:bool = False):
        if model_id == HuggingFaceModelID.SEAMLESS_LARGE or clear_all:
            if self.seamless_model is not None:
                self.seamless_model.delete_model()
                self.seamless_model = None
            
        if model_id == HuggingFaceModelID.VITS_KAZ_TTS or clear_all:
            if self.vits_kaz is not None:
                self.vits_kaz.delete_model()
                self.vits_kaz = None
        
        if model_id == HuggingFaceModelID.VITS_MYA_TTS or clear_all:
            if self.vits_mya is not None:
                self.vits_mya.delete_model()
                self.vits_mya = None 
        
if __name__ == "__main__":
    
    # T
    print(HuggingFaceModelID.SEAMLESS_LARGE.value)
    seamless = SeamlessModel(HuggingFaceModelID.SEAMLESS_LARGE.value)
    
    print("_"*100)
    print("Testing SeamlessModel class with text inputs")
    print( "\n"*3)
    # text
    result = seamless.translate(text = "Hello, how are you?",
                       tgt_lang = "hin")
    
    print("text", result, "\n\n")
    
    # list of text to text
    result = seamless.translate(text = ["Hello, how are you?" for _ in range(5)],
                    tgt_lang = "hin")
        
    print("list of text", result, "\n\n")
    
    # list of text to text to speech
    result = seamless.translate(text = ["Hello, how are you?" for _ in range(5)],
                    tgt_lang = "hin", generate_speech = True)
        
    print("list of text to speech ", result, "\n\n")
    
    # list of text to text to speech and text
    result = seamless.translate(text = ["Hello, how are you?" for _ in range(5)],
                    tgt_lang = "hin", generate_speech_and_text= True)
        
    print("list of text to speech and text ", result, "\n\n")
    print("_"*100)
    print("Testing SeamlessModel class with audio inputs")
    print( "\n"*3)
    
    # testing with audio    

    audio, sr = torchaudio.load("tam_0030.wav")
    if sr!= 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
    

    chunk = audio.numpy().flatten()
    chunk = chunk[:16000]
    
    
    audio = chunk
    audio_list = [chunk for _ in range(5)]
    
      # text
    result = seamless.translate(audio = audio,
                       tgt_lang = "hin")
    
    print("audio", result, "\n\n")
    
    # list of text to text
    result = seamless.translate(audio = audio_list,
                    tgt_lang = "hin")
        
    print("list of audio", result, "\n\n")
    
    # list of text to text to speech
    result = seamless.translate(audio = audio_list,
                    tgt_lang = "hin", generate_speech = True)
        
    print("list of audio to speech ", result, "\n\n")
    
    # list of text to text to speech and text
    result = seamless.translate(audio = audio_list,
                    tgt_lang = "hin", generate_speech_and_text= True)
        
    print("list of text to speech and text ", result, "\n\n")
    
    seamless.delete_model()
    
    print("_"*100)
    print("Testing TranslationHanlder class.")
    print( "\n"*3)
    
    
    translation_handler = TranslationHandler()
    result = translation_handler.text_to_text("Hello, how are you?", "hin")
    print("text to text", result, "\n\n")
    
    result = translation_handler.text_to_speech("Hello, how are you?", "hin")
    print("text to speech", result, "\n\n")
    
    result = translation_handler.translate_pdf("utils/sample.pdf", "fra")
    print("pdf to text", result, "\n\n")
    
    result = translation_handler.translate_pdf("utils/sample.pdf", "fra", generate_speech = True)
    print("pdf to speech", result, "\n\n")
    
    translation_handler.delete_model(clear_all = True)
    
    

    
    

    
    
    