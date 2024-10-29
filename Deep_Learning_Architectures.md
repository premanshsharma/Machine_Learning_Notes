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
- **Basic Idea:** VGG16 focuses on deepening the network architecture while maintaining simplicity in the design. It primarily uses small 3x3 convolution filters stacked on top of each other to learn complex features from images.
- **Architecture:** Consists of 16 layers (13 convolutional layers and 3 fully connected layers) with small 3x3 filters and max pooling layers.
- **Strengths:** Its deep architecture captures detailed image features, leading to high classification accuracy. VGG16 is known for its simplicity and effectiveness, often as a baseline for other models.
- **Use Cases:** Image recognition tasks include identifying objects in photos or distinguishing between different categories.
### 1.2.2 Inception V3
- **Basic Idea:** Inception V3 introduces inception modules that allow the network to learn features at different scales simultaneously. This architecture helps to capture a richer representation of the input images.
- **Architecture:**
  - **Inception Modules:** These modules consist of parallel convolutions with different kernel sizes (1x1, 3x3, 5x5), allowing the network to learn multi-scale features. This versatility helps the model adapt to various object sizes and shapes.
  - **Factorization:** The architecture employs factorization techniques (e.g., breaking down 5x5 convolutions into two 3x3 convolutions) to reduce the computational cost while retaining performance.
  - **Auxiliary Classifiers:** Inception V3 includes auxiliary classifiers at intermediate layers, which help combat the vanishing gradient problem during training by providing additional gradient signals.
  - **Global Average Pooling:** Instead of fully connected layers, Inception V3 uses global average pooling to reduce overfitting and computational burden while maintaining performance.
 
### 1.2.3 ResNet




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
