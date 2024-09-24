import torch
import torchvision
from PIL import Image
import torchvision.transforms
import matplotlib.pyplot as plt
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import cv2
import os
import numpy as np
from PIL import Image

class SegmentationModel:
    
    def __init__(self, confidence_threshold=0.5, image_size=(512, 512)):
        self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def segment(self, image: Image.Image):
        image_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        masks = outputs[0]['masks'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        boxes = outputs[0]['boxes'].cpu().numpy()

        segmented_regions = []
        for i in range(len(labels)):
            if scores[i] > self.confidence_threshold:
                mask = masks[i][0] > 0.5
                segmented_regions.append({
                    'label': COCO_CLASSES[labels[i]],
                    'mask': mask.astype(np.uint8),
                    'box': boxes[i].tolist(),
                    'score': scores[i]
                })

        return segmented_regions

    def post_process_masks(self, segmented_regions, original_image):
        original_size = original_image.size
        post_processed_regions = []

        for region in segmented_regions:
            mask_resized = np.array(Image.fromarray(region['mask']).resize(original_size))
            post_processed_regions.append({
                'label': region['label'],
                'mask': mask_resized,
                'box': region['box'],
                'score': region['score']
            })

        return post_processed_regions