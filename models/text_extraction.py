import pytesseract
from PIL import Image

class TextExtractionModel:
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from a given image using Tesseract OCR.
        Args:
            - image (PIL.Image): The input image to extract text from.

        Returns:
            - text (str): The extracted text from the image.
        """
        # Convert the image to grayscale if it's not already in grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Run text extraction using pytesseract
        text = pytesseract.image_to_string(image)

        return text.strip()  # Return the extracted text, stripped of extraneous whitespace
