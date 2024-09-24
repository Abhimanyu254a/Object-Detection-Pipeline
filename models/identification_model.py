import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
from PIL import Image

# COCO class labels for Faster R-CNN
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class IdentificationModel:
    def __init__(self, confidence_threshold=0.5):
        """
        Initializes the Faster R-CNN model for object identification.
        """
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()  # Set the model to evaluation mode
        self.confidence_threshold = confidence_threshold  # Set a confidence threshold

        # Define image transformations (resize, convert to tensor, and normalize)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def identify(self, image: Image.Image):
        """
        Identifies objects in the image using the Faster R-CNN model.
        Args:
            - image (PIL.Image): The input image to identify objects in.

        Returns:
            - predictions (dict): A dictionary containing bounding boxes, labels, and scores for each identified object.
        """
        # Preprocess the image
        image_tensor = self.transform(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Extract relevant information (boxes, labels, and scores)
        boxes = predictions[0]['boxes'].cpu().numpy()  # Bounding boxes
        labels = predictions[0]['labels'].cpu().numpy()  # Object labels
        scores = predictions[0]['scores'].cpu().numpy()  # Confidence scores

        # Filter out low-confidence predictions
        identified_objects = []
        for i in range(len(labels)):
            if scores[i] > self.confidence_threshold:
                identified_objects.append({
                    'box': boxes[i].tolist(),
                    'label': COCO_CLASSES[labels[i]],
                    'score': scores[i]
                })

        return identified_objects


if __name__ == "__main__":
    # Replace 'insert_the_image' with the actual image path or image variable
    image_path = 'insert_the_image'
    image = Image.open(image_path)

    # Initialize the identification model
    identification_model = IdentificationModel(confidence_threshold=0.6)
    
    # Identify objects in the image
    identified_objects = identification_model.identify(image)

    # Print identified objects
    for obj in identified_objects:
        print(f"Label: {obj['label']}, Box: {obj['box']}, Score: {obj['score']:.2f}")
