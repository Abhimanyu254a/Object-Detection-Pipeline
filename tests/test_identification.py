from PIL import Image
from model.identification import IdentificationModel
from model.text_extraction import TextExtractionModel

class TextIdentificationModel:
    def __init__(self, confidence_threshold=0.5):
        """
        Initializes both the object identification and text extraction models.
        """
        self.identification_model = IdentificationModel(confidence_threshold=confidence_threshold)
        self.text_extraction_model = TextExtractionModel()

    def identify_objects_and_text(self, image: Image.Image):
        """
        Identifies objects in the image, extracts text from objects (if applicable).
        Args:
            - image (PIL.Image): The input image in which to identify objects and extract text.

        Returns:
            - result (list): A list of dictionaries containing labels, boxes, scores, and extracted text (if present).
        """
        # Identify objects using the IdentificationModel
        identified_objects = self.identification_model.identify(image)
        result = []

        # Iterate through the identified objects
        for obj in identified_objects:
            # Extract the bounding box and crop the object from the image
            x1, y1, x2, y2 = [int(coord) for coord in obj['box']]
            cropped_object = image.crop((x1, y1, x2, y2))

            # Try to extract text from the cropped object
            extracted_text = self.text_extraction_model.extract_text(cropped_object)

            # Store the object information along with extracted text
            result.append({
                'label': obj['label'],
                'box': obj['box'],
                'score': obj['score'],
                'extracted_text': extracted_text if extracted_text else "No text found"
            })

        return result
