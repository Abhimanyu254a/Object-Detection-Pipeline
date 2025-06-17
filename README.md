# Image Segmentation, Object Detection, and Object Extraction Pipeline

## Overview
This project provides an end-to-end pipeline that integrates **image segmentation**, **object detection**, **object extraction**, and **text extraction**. The pipeline uses deep learning models such as **Mask R-CNN** and **Faster R-CNN** to detect and segment objects from images, extract relevant information, and identify any text present within the objects using **Tesseract OCR**.

### Features:
- **Image Segmentation** using **Mask R-CNN**.
- **Object Detection** using **Faster R-CNN**.
- **Object Extraction** from detected bounding boxes.
- **Text Extraction** from detected objects using **Pytesseract**.
- **Text Summarization** using **Hugging Face's Transformers** for summarizing extracted text or object attributes.

---

## Tech Stack
The project is built using the following technologies:
- **PyTorch**: For loading and running pre-trained models for object detection and segmentation.
- **Torchvision**: For access to pre-trained **Mask R-CNN** and **Faster R-CNN** models.
- **OpenCV**: For image manipulation and object extraction.
- **Pillow (PIL)**: For image loading, preprocessing, and manipulation.
- **Pytesseract**: For text extraction using **Tesseract OCR**.
- **Hugging Face Transformers**: For text summarization using **DistilBART**.
- **NumPy**: For numerical operations and array manipulations.

---

## Installation
There are some things code and also some things I don't done but I will do my best to complete it
### Clone the Repository
First, clone the repository to your local machine:
```bash
git clone //github.com/Object-Detection-Pipeline
cd Object-Detection-Pipeline
pip install -r requirement.txt
 
