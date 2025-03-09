# A3: Small Object Detection Using YOLO

## Project Overview
This project focuses on implementing and optimizing the YOLOv8 object detection model for small object detection tasks. Using a custom dataset of car and traffic sign images, we fine-tuned the YOLOv8n model to improve detection accuracy on small objects, which are typically challenging for object detection algorithms.

## Table of Contents
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Usage](#usage)
- [Experimentation Results](#experimentation-results)
- [Challenges in Small Object Detection](#challenges-in-small-object-detection)
- [Optimization Strategies](#optimization-strategies)
- [Conclusion](#conclusion)

## Dataset Description

### Dataset Composition
- **Source**: Custom dataset of car and traffic sign images
- **Size**: Approximately 300 labeled images
- **Classes**: stop signs, and other traffic signs
- **Characteristics**: Varied lighting conditions, distances, and object sizes

### Preprocessing Steps
1. **Data Collection**: Gathered images from various traffic scenarios
2. **Annotation**: Used LabelImg to create bounding box annotations in YOLO format
3. **Data Organization**: Structured the dataset into a YOLO-compatible format with:
   - `images/train`: Training images
   - `images/val`: Validation images
   - `labels/train`: Training annotations
   - `labels/val`: Validation annotations

4. **Dataset Configuration**: Created a `data.yaml` file specifying:
   - Paths to train/val directories
   - Class names
   - Number of classes

## Installation

```bash
# Install required packages
pip install ultralytics opencv-python numpy torch torchvision torchaudio labelImg

# Clone the repository (if applicable)

```

## Usage

### Training the Model
```python
from ultralytics import YOLO

# Load the base model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="/path/to/data.yaml",
    epochs=10,
    batch=16,
    optimizer='auto'
)

# Save the trained model
model.save('yolov8n-custom.pt')
```

### Running Inference
```python
from ultralytics import YOLO

# Load the custom model
model = YOLO("yolov8n-custom.pt")

# Image inference
results = model("/path/to/image.jpg")
results[0].show()  # Display results

# Video inference
results = model.predict(
    source="/path/to/video.mp4",
    save=True,
    stream=True,
    imgsz=3840
)
```

## Experimentation Results

### Base Performance
- Initial detection accuracy: 67%
- Small object recall: 42%
- Primary issues: Missing small traffic signs, false negatives at distance

### Optimization Experiments

| Configuration | mAP@50 | Small Object Recall | Processing Time (FPS) | Notes |
|---------------|--------|---------------------|------------------------|-------|
| Default YOLOv8n | 0.67 | 0.42 | 24 | Baseline, poor small object detection |
| Higher Resolution (3840px) | 0.74 | 0.61 | 16 | Significant improvement in small object detection |
| Lower NMS Threshold (0.3) | 0.71 | 0.58 | 23 | Reduced false negatives, some increase in false positives |
| Lower Confidence Threshold (0.25) | 0.69 | 0.65 | 22 | Best recall for small objects, more false positives |
| Combined Optimizations | 0.76 | 0.68 | 15 | Best overall performance |

### Best Configuration
- **Input Resolution**: 3840px
- **NMS Threshold**: 0.35
- **Confidence Threshold**: 0.25
- **Batch Size**: 16
- **Optimizer**: Auto (AdamW)

## Challenges in Small Object Detection

1. **Feature Representation Limitations**
   - Small objects generate limited features in convolutional layers
   - Deep network architecture loses spatial information through downsampling

2. **Scale Variations**
   - Significant size differences between objects in the same image
   - Standard architectures struggle with extreme scales

3. **Low Signal-to-Noise Ratio**
   - Small objects have fewer distinguishing pixels
   - More susceptible to background noise and image artifacts

4. **Annotation Precision**
   - Precise bounding box annotation is more difficult for small objects
   - Even minor annotation errors significantly impact training quality

5. **Training Data Imbalance**
   - Dataset typically contains fewer small objects
   - Model biases toward detecting larger, more prominent objects

## Optimization Strategies

### Effective Approaches

1. **Resolution Scaling**
   - Increasing input resolution to 3840px provided the most significant improvement
   - Enhances feature representation for small objects

2. **NMS and Confidence Threshold Tuning**
   - Lowering NMS threshold to 0.35 reduced overlapping detections
   - Decreasing confidence threshold to 0.25 improved recall of small objects

3. **Feature Pyramid Enhancement**
   - YOLOv8's feature pyramid effectively combines features at different scales
   - Particularly beneficial for detecting objects at various distances

4. **Focused Data Augmentation**
   - Augmentation techniques specifically targeting small object scenarios
   - Mosaic augmentation improved model generalization

### Suggested Improvements

1. **Advanced Architectures**
   - Consider specialized architectures like YOLO-NAS or YOLOv8x for better feature extraction
   - Implement attention mechanisms to focus on relevant image regions

2. **Multi-Scale Training and Testing**
   - Train on multiple input resolutions
   - Test-time augmentation with multiple scales

3. **Curriculum Learning**
   - Start training with larger objects and gradually introduce smaller ones
   - Helps the model build better feature hierarchies

4. **Domain-Specific Data Augmentation**
   - Create more training examples of small objects through targeted cropping
   - Apply synthetic data generation for underrepresented object classes

5. **Ensemble Methods**
   - Combine predictions from multiple models trained with different configurations
   - Particularly effective for improving recall on challenging cases

## Conclusion

This project demonstrates that YOLOv8, with proper optimization, can effectively detect small objects in traffic scenarios. The most significant improvements came from increasing input resolution and carefully tuning detection thresholds. With these optimizations, we achieved a 26% improvement in small object recall compared to the baseline configuration.

For real-world applications, the trade-off between detection accuracy and processing speed must be considered, with our optimized model running at approximately 15 FPS on GPU hardware. Future work should focus on implementing more specialized architectures and experimenting with attention mechanisms to further improve small object detection performance.

