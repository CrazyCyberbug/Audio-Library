import os
import tempfile
import numpy as np
import soundfile as sf
from fpdf import FPDF
import requests
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import datetime
from weasyprint import HTML
from datetime import datetime

class RequestManager:
    def __init__(self, base_url):
        """
        Initializes the RequestManager with the base URL of the server.

        :param base_url: The base URL of the API (e.g., http://localhost:8000)
        """
        self.base_url = base_url

    def request_text_to_text(self, text, tgt_lang, src_lang):
        """
        Sends a request to the text-to-text translation endpoint.

        :param text: The text to translate.
        :param tgt_lang: The target language.
        :return: Response JSON or error message.
        """
        url = f"{self.base_url}/text-to-text"
        payload = {
            "text": text,
            "tgt_lang": tgt_lang,
            "src_lang": src_lang
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def request_text_to_speech(self, text, tgt_lang):
        """
        Sends a request to the text-to-speech endpoint.

        :param text: The text to convert to speech.
        :param tgt_lang: The target language.
        :return: Response content (audio file) or error message.
        """
        url = f"{self.base_url}/text-to-speech"
        payload = {
            "text": text,
            "tgt_lang": tgt_lang
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            # print("\n"*6)
            # print(eval(response.content)['data']['audio'])
            # print("\n"*6)
            audio_list = eval(response.content)['data']['audio']
            audio_array = np.array(audio_list)
            
            sf.write(file = "output_audio.wav",
                        data = audio_array,
                        samplerate = 16000)
            
            return {"message": "Audio saved as output_audio.wav"}
        except requests.RequestException as e:
            return {"error": str(e)}

    def request_pdf_to_text(self, file_path, tgt_lang):
        """
        Sends a request to the PDF-to-text endpoint.

        :param file_path: The path to the PDF file.
        :param tgt_lang: The target language.
        :return: Response JSON or error message.
        """
        url = f"{self.base_url}/pdf-to-text"
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                data = {"tgt_lang": tgt_lang}
                response = requests.post(url, files=files, data=data)
                response.raise_for_status()
                print(f"{response.content} {type(response.content)}")
                translated_text = eval(response.content)["data"]["text"]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{os.path.splitext(file_path)[0]}_translated_{tgt_lang}_{timestamp}.pdf"
                print(f" the output file name would be {output_filename}")
                output_filepath = save_text_as_pdf(file_name = output_filename, text = translated_text)       
                
                return response.json(), output_filepath
        except (requests.RequestException, FileNotFoundError) as e:
            return {"error": str(e)}

    def request_pdf_to_speech(self, file_path, tgt_lang):
        """
        Sends a request to the PDF-to-speech endpoint.

        :param file_path: The path to the PDF file.
        :param tgt_lang: The target language.
        :return: Response content (audio file) or error message.
        """
        url = f"{self.base_url}/pdf-to-speech"
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                data = {"tgt_lang": tgt_lang}
                response = requests.post(url, files=files, data=data)
                response.raise_for_status()
                # print(response.content)
                audio_list = eval(response.content)['data']['audio']
                audio_array = np.array(audio_list)

                sf.write(file = "output_audio_from_pdf.wav",
                         data = audio_array,
                         samplerate = 16000)
                
                return {"message": "Audio saved as output_audio_from_pdf.wav"}
        except (requests.RequestException, FileNotFoundError) as e:
            return {"error": str(e)}


def save_text_as_pdf(text, file_name, output_dir="output_doc", font_family="Noto Sans"):
    """
    Converts the input text to a PDF file and saves it.

    Args:
        text (str): The text content to be saved in the PDF.
        output_dir (str): Directory to save the generated PDF file.
        font_family (str): The font family to be used for multilingual support (default: "Noto Sans").

    Returns:
        str: The path to the saved PDF file.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = file_name
    pdf_output_path = os.path.join(output_dir, output_filename)
    print(f"attempting to save the results in {pdf_output_path}")

    # Create the HTML content with UTF-8 encoding and proper font family
    full_html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text to PDF</title>
        <style>
            body {{
                font-family: '{font_family}', sans-serif;
                line-height: 1.6;
                margin: 20px;
            }}
        </style>
    </head>
    <body>
        <p>{text}</p>
    </body>
    </html>
    """
    
    # Create a temporary HTML file to pass to WeasyPrint
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_html_file:
            temp_html_file.write(full_html_content)
            temp_html_path = temp_html_file.name
        
        # Convert the HTML file to PDF using WeasyPrint
        html = HTML(temp_html_path, base_url=os.path.dirname(temp_html_path))
        html.write_pdf(pdf_output_path)

        # Clean up the temporary HTML file
        os.remove(temp_html_path)

        print(f"PDF file saved to: {pdf_output_path}")
        return os.path.abspath(pdf_output_path)
    except Exception as e:
        print(f"Error while generating PDF: {e}")
        return None




# Example Usage
if __name__ == "__main__":
    base_url = "http://127.0.0.1:8000"  # Replace with your server URL
    manager = RequestManager(base_url)

    # Example calls
    print(manager.request_text_to_text("Hello, world!", "hin"))
    print(manager.request_text_to_speech("Hello, world!", "hin"))
    print(manager.request_pdf_to_text("sample.pdf", "hin"))
    
    
    
    
    response, pdf_path = manager.request_pdf_to_text("sample.pdf", "arb")
    print(type(response))
    
    print("-"*30)
    print(response.keys())
    print("-"*30)
    translated_text =  response["data"]["text"]
    print(f"translated text: {translated_text}")
    # print(f"Requesting pdf tranlations for  {os.path.abspath("sample.pdf")} and the output stored at {pdf_path}")
    # response, pdf_path = manager.request_pdf_to_text("sample.pdf", "arb")
    print(f"the file is saved at {pdf_path}")

    print(os.path.exists(pdf_path))
    # print(manager.request_pdf_to_speech("sample.pdf", "hin"))


