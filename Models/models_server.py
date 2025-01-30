import uvicorn
from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from typing import Union, List
import tempfile
import numpy as np
from Models.Seamless import TranslationHandler

app = FastAPI()

# Initialize the TranslationHandler
translation_handler = TranslationHandler()


class TextToTextRequest(BaseModel):
    text: Union[str, List[str]]
    tgt_lang: str
    src_lang: str

class TextToSpeechRequest(BaseModel):
    text: Union[str, List[str]]
    tgt_lang: str
    src_lang: str

@app.post("/text-to-text")
async def text_to_text(request: TextToTextRequest):
    """
    Endpoint for text-to-text translation.
    """
    try:
        result = translation_handler.text_to_text(text=request.text, tgt_lang=request.tgt_lang, src_lang= request.src_lang)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    """
    Endpoint for text-to-speech translation.
    """
    try:
        result = translation_handler.text_to_speech(text=request.text, tgt_lang=request.tgt_lang)
        if "audio" in result:
            # Convert audio numpy array to list for JSON compatibility
            result["audio"] = result["audio"].tolist()
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/pdf-to-text")
async def pdf_to_text(file: UploadFile, tgt_lang: str = Form(...)):
    """
    Endpoint for PDF-to-text translation.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file.close()
            result = translation_handler.translate_pdf(file_path=temp_file.name, tgt_lang=tgt_lang, generate_speech=False)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/pdf-to-speech")
async def pdf_to_speech(file: UploadFile, tgt_lang: str = Form(...)):
    """
    Endpoint for PDF-to-speech translation.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file.close()
            result = translation_handler.translate_pdf(file_path=temp_file.name, tgt_lang=tgt_lang, generate_speech=True)
        if "audio" in result:
            # Convert audio numpy array to list for JSON compatibility
            result["audio"] = result["audio"].tolist()
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("models_server:app", host="127.0.0.1", port=8000, reload=True)  # Run the server on localhost:8000
