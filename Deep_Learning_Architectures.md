# MindMap
- Image Based
  - Object Detection
    - YOLO
  - Classification
    - VGG16
    - Inception V3
    - ResNet
  - Segmentation
    - U-Net
    - Segmentation Model
  - 3D Representation
    - NeRF

# 1. Image Based
## 1.1. Object Detection
- Object detection refers to the ability of a system to identify and locate objects within an image. It involves drawing bounding boxes around detected objects and assigning class labels to them.
### 1.1.1 YOLO (You only look once):

## 1.2 Classification
### 1.2.1 VGG16
- Architecture: Consists of 16 layers (13 convolutional layers and 3 fully connected layers) with small 3x3 filters and max pooling layers.
- Strengths: Its deep architecture captures detailed features from images, leading to high classification accuracy. VGG16 is known for its simplicity and effectiveness, often serving as a baseline for other models.
- Use Cases: Image recognition tasks, such as identifying objects in photos or distinguishing between different categories.
### 1.2.2 Inception V3
## 1.3 Segmentation Model
- **Purpose:** Segmentation models aim to classify each pixel in an image, which is crucial in various applications such as medical imaging, autonomous driving, and image editing.
### 1.3.1 U-Net
- **Purpose:**
  - U-Net is primarily designed for image segmentation tasks, particularly in medical imaging (like segmenting organs or tumors in MRI scans).
- **Architecture:**
  - **Contracting Path (Encoder):** This part of the network captures context. It consists of several convolutional layers followed by max-pooling layers. Each step reduces the spatial dimensions of the image while increasing the number of feature channels.
  - **Bottleneck:** The deepest part of the U-Net where the spatial dimensions are the smallest, but the feature representation is rich.
  - **Expansive Path (Decoder):** This part upsamples the feature maps to the original image size. It uses transposed convolutions (or deconvolutions) to increase the spatial dimensions.
  - **Skip Connections:** The architecture includes skip connections that concatenate feature maps from the encoder with corresponding feature maps in the decoder. This helps retain high-resolution information, improving the model’s ability to localize and delineate objects in the output mask.
- **Key Feature:** The combination of downsampling and upsampling with skip connections enables the U-Net to produce precise segmentation masks, making it very effective for tasks where pixel-level accuracy is crucial.
![OIP](https://github.com/user-attachments/assets/33c7aeab-406d-409a-8c67-6ab2d9ff80af)
