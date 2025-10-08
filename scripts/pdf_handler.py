# src/pdf_handler.py

import fitz 
import pytesseract
from PIL import Image
import io

try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception as e:
    print(f"Tesseract executable not found, please check the path in src/pdf_handler.py: {e}")

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_bytes = uploaded_file.getvalue()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        full_text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # First, try to extract standard embedded text
            page_text = page.get_text("text")
            
            # If the page has very little text, assume it's an image and use OCR
            if len(page_text.strip()) < 100:
                # Render the page to a high-resolution image
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes))
                
                # Use Tesseract to perform OCR on the image
                ocr_text = pytesseract.image_to_string(image, lang='eng')
                
                # Use the OCR text if it's more substantial than the embedded text
                if len(ocr_text.strip()) > len(page_text.strip()):
                    page_text = ocr_text
            
            full_text += page_text.replace('\n', ' ') + ' '
            
        return full_text.strip()

    except Exception as e:
        error_message = f"PDF processing error: {e}. Ensure Tesseract-OCR is installed and the path in src/pdf_handler.py is correct."
        return error_message