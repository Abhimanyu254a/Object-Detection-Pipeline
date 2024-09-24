import unittest
from PIL import Image
import cv2
import os

# Import the necessary modules from the model
from model.segmentation import SegmentationModel
from model.object_extraction import ObjectExtractor

class TestObjectExtraction(unittest.TestCase):

    def setUp(self):
        # Set up paths to test images
        self.image_path = 'tests/sample_image.jpg'  # Replace with a valid test image path
        self.segmented_image_path = 'tests/test_segmented_image.png'  # Segmented image

        # Load the test image
        self.image = Image.open(self.image_path)
        self.segmented_image = cv2.imread(self.segmented_image_path, cv2.IMREAD_GRAYSCALE)

        # Initialize models
        self.segmentation_model = SegmentationModel(confidence_threshold=0.5)
        self.object_extractor = ObjectExtractor()

        # Perform segmentation on the image
        self.segmented_regions = self.segmentation_model.segment(self.image)
        self.extracted_objects_output_dir = 'tests/extracted_objects/'

    def test_object_extraction(self):
        """
        Test the object extraction from a segmented image.
        """
        # Extract objects using the ObjectExtractor
        extracted_objects = self.object_extractor.extract_objects(
            self.image, self.segmented_regions, output_dir=self.extracted_objects_output_dir
        )
        
        # Check that the extracted objects are saved in the output directory
        self.assertTrue(os.path.exists(self.extracted_objects_output_dir), "Output directory should be created")
        self.assertGreater(len(os.listdir(self.extracted_objects_output_dir)), 0, "Extracted objects directory should not be empty")

    def tearDown(self):
        """
        Clean up the extracted objects directory after the tests to avoid clutter.
        """
        if os.path.exists(self.extracted_objects_output_dir):
            for filename in os.listdir(self.extracted_objects_output_dir):
                file_path = os.path.join(self.extracted_objects_output_dir, filename)
                os.remove(file_path)
            os.rmdir(self.extracted_objects_output_dir)

if __name__ == "__main__":
    unittest.main()
