---
sidebar_position: 3
---

# Object Detection + Manipulation Pipeline

## Chapter Objectives

- Understand computer vision techniques for object detection in robotics
- Implement real-time object detection pipelines for humanoid robots
- Create manipulation planning systems for object interaction
- Integrate perception and action for autonomous manipulation
- Optimize object detection for real-time robotic applications

## Introduction to Object Detection for Robotics

Object detection in robotics is a critical capability that enables robots to perceive, identify, and interact with objects in their environment. For humanoid robots, object detection systems must:

- **Detect objects in real-time** for responsive interaction
- **Handle diverse object types** from household items to tools
- **Work in varied lighting conditions** and environments
- **Provide accurate 3D localization** for manipulation
- **Integrate seamlessly** with manipulation planning systems

### Why Specialized Object Detection for Robotics?

1. **3D Understanding**: Need 3D position and orientation for manipulation
2. **Real-time Performance**: Robot control requires low-latency detection
3. **Robustness**: Must handle motion blur, occlusions, and changing conditions
4. **Safety**: Critical to avoid collisions and ensure safe interaction
5. **Context Awareness**: Understanding object affordances and relationships

### Key Challenges in Robotic Object Detection

- **Scale Variation**: Objects at different distances appear at different scales
- **Viewpoint Changes**: Objects look different from various angles
- **Occlusions**: Partially hidden objects require completion understanding
- **Lighting Conditions**: Performance must be consistent across lighting
- **Real-time Constraints**: Detection must be fast enough for robot control

## Object Detection Architectures

### YOLO for Robotics Applications

```python
# python/yolo_robotics.py
import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass

@dataclass
class DetectionResult:
    """Result of object detection"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center_3d: Optional[Tuple[float, float, float]]  # 3D position in world coordinates
    rotation_3d: Optional[Tuple[float, float, float]]  # 3D rotation

class YOLORobotDetector:
    """YOLO-based object detector optimized for robotics"""

    def __init__(self, model_path: str = "yolov5s.pt", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

        # Load YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.to(self.device)
        self.model.eval()

        # Set model parameters
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4
        self.image_size = 640  # Standard YOLO input size

        # Class names (COCO dataset by default)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Camera intrinsic parameters (should be calibrated for your specific camera)
        self.camera_matrix = np.array([
            [554.25, 0.0, 320.0],
            [0.0, 554.25, 240.0],
            [0.0, 0.0, 1.0]
        ])

        print(f"YOLO Robot Detector initialized on {self.device}")

    def detect_objects(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect objects in an image"""
        start_time = time.time()

        # Preprocess image
        img_tensor = self._preprocess_image(image)

        # Run inference
        with torch.no_grad():
            results = self.model(img_tensor)

        # Process results
        detections = self._process_detections(results, image.shape[:2])

        # Calculate 3D positions if depth information is available
        if hasattr(self, 'depth_image') and self.depth_image is not None:
            detections = self._calculate_3d_positions(detections)

        # Performance metrics
        inference_time = time.time() - start_time
        print(f"Detection completed in {inference_time:.3f}s, found {len(detections)} objects")

        return detections

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for YOLO inference"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Resize image to model input size while maintaining aspect ratio
        h, w = image_rgb.shape[:2]
        scale = min(self.image_size / h, self.image_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image_rgb, (new_w, new_h))

        # Pad to make it square
        delta_w = self.image_size - new_w
        delta_h = self.image_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114]
        )

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        return img_tensor

    def _process_detections(self, results, image_shape) -> List[DetectionResult]:
        """Process YOLO detection results"""
        detections = []

        # Get predictions
        pred = results.pred[0]  # First image in batch

        if len(pred) > 0:
            for *xyxy, conf, cls in pred.tolist():
                # Convert to integer coordinates
                x1, y1, x2, y2 = map(int, xyxy)

                # Calculate width and height
                width = x2 - x1
                height = y2 - y1

                # Apply confidence threshold
                if conf >= self.confidence_threshold:
                    detection = DetectionResult(
                        class_id=int(cls),
                        class_name=self.class_names[int(cls)] if int(cls) < len(self.class_names) else f"unknown_{int(cls)}",
                        confidence=conf,
                        bbox=(x1, y1, width, height),
                        center_3d=None,  # Will be calculated later if depth is available
                        rotation_3d=None
                    )
                    detections.append(detection)

        return detections

    def _calculate_3d_positions(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Calculate 3D positions from 2D detections and depth image"""
        updated_detections = []

        for detection in detections:
            # Get center of bounding box in 2D
            x, y, w, h = detection.bbox
            center_x = x + w // 2
            center_y = y + h // 2

            # Get depth at center point
            if (center_y < self.depth_image.shape[0] and
                center_x < self.depth_image.shape[1]):
                depth = self.depth_image[center_y, center_x]

                # Convert 2D point to 3D using camera intrinsics
                z = depth  # Depth value
                x_3d = (center_x - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                y_3d = (center_y - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]

                detection.center_3d = (x_3d, y_3d, z)

            updated_detections.append(detection)

        return updated_detections

    def set_depth_image(self, depth_image: np.ndarray):
        """Set depth image for 3D position calculation"""
        self.depth_image = depth_image

    def visualize_detections(self, image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Visualize detections on image"""
        vis_image = image.copy()

        for detection in detections:
            x, y, w, h = detection.bbox

            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw center point
            center_x, center_y = x + w // 2, y + h // 2
            cv2.circle(vis_image, (center_x, center_y), 5, (0, 0, 255), -1)

        return vis_image

class RealTimeObjectDetector:
    """Real-time object detection for robotics applications"""

    def __init__(self, detector: YOLORobotDetector):
        self.detector = detector
        self.is_running = False
        self.detection_callback = None
        self.frame_buffer = []
        self.max_buffer_size = 5

    def start_detection(self, callback: callable = None):
        """Start real-time detection"""
        self.detection_callback = callback
        self.is_running = True
        print("Real-time object detection started")

    def stop_detection(self):
        """Stop real-time detection"""
        self.is_running = False
        print("Real-time object detection stopped")

    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """Process a single frame"""
        if not self.is_running:
            return []

        # Add frame to buffer
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)  # Remove oldest frame

        # Run detection
        detections = self.detector.detect_objects(frame)

        # Call callback if provided
        if self.detection_callback:
            self.detection_callback(detections, frame)

        return detections

def demonstrate_yolo_detection():
    """Demonstrate YOLO object detection"""
    print("Demonstrating YOLO Object Detection for Robotics")

    try:
        # Initialize detector (this would load a pre-trained model)
        # For this example, we'll just show the structure
        detector = YOLORobotDetector(model_path="yolov5s.pt")  # Replace with actual model path

        # Example: Process a sample image
        # In practice, you would load an actual image
        sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect_objects(sample_image)

        print(f"Found {len(detections)} objects:")
        for detection in detections[:5]:  # Show first 5 detections
            print(f"  - {detection.class_name} (confidence: {detection.confidence:.2f})")

    except Exception as e:
        print(f"Error initializing YOLO detector: {e}")
        print("Make sure you have YOLOv5 installed and model file available")

if __name__ == "__main__":
    demonstrate_yolo_detection()
```

## Deep Learning Approaches for Object Detection

### Transformer-Based Detection

```python
# python/transformer_detection.py
import torch
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import numpy as np
from typing import List, Dict, Any
import requests
from PIL import Image
import time

class DETRRobotDetector:
    """DETR (DEtection TRansformer) for robotic object detection"""

    def __init__(self, model_name: str = "facebook/detr-resnet-50", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

        # Load DETR model and processor
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # COCO dataset categories
        self.categories = [
            "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
            "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
            "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        ]

        self.confidence_threshold = 0.9

        print(f"DETR Robot Detector initialized on {self.device}")

    def detect_objects(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect objects using DETR"""
        start_time = time.time()

        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = Image.fromarray(image)
        else:
            image_pil = image

        # Process image
        inputs = self.processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs to detections
        target_sizes = torch.tensor([image_pil.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]

        # Process results
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detection = DetectionResult(
                class_id=label.item(),
                class_name=self.categories[label] if label < len(self.categories) else f"unknown_{label}",
                confidence=score.item(),
                bbox=(int(box[0].item()), int(box[1].item()),
                      int(box[2].item() - box[0].item()), int(box[3].item() - box[1].item())),
                center_3d=None,
                rotation_3d=None
            )
            detections.append(detection)

        inference_time = time.time() - start_time
        print(f"DETR detection completed in {inference_time:.3f}s, found {len(detections)} objects")

        return detections

class CustomObjectDetector(nn.Module):
    """Custom object detection model for robotics-specific objects"""

    def __init__(self, num_classes: int = 20, pretrained: bool = True):
        super().__init__()

        # Use a backbone network (ResNet50 as example)
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)

        # Remove the final classification layer
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

        # Feature pyramid network for multi-scale detection
        self.backbone = backbone
        self.fpn = self._build_fpn()

        # Detection heads
        self.classification_head = self._build_classification_head(num_classes)
        self.regression_head = self._build_regression_head()

        # Anchor generation
        self.anchors = self._generate_anchors()

    def _build_fpn(self):
        """Build Feature Pyramid Network"""
        # Simplified FPN implementation
        return nn.ModuleList([
            nn.Conv2d(2048, 256, kernel_size=1),  # Top-down pathway
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.Conv2d(512, 256, kernel_size=1),
        ])

    def _build_classification_head(self, num_classes: int):
        """Build classification head"""
        return nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
        )

    def _build_regression_head(self):
        """Build bounding box regression head"""
        return nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 4, kernel_size=3, padding=1)  # 4 coordinates: x, y, w, h
        )

    def _generate_anchors(self):
        """Generate anchor boxes"""
        # This is a simplified anchor generation
        # In practice, you'd have multiple scales and aspect ratios
        anchors = []
        scales = [32, 64, 128, 256, 512]
        ratios = [0.5, 1.0, 2.0]

        for scale in scales:
            for ratio in ratios:
                w = scale * (ratio ** 0.5)
                h = scale / (ratio ** 0.5)
                anchors.append((w, h))

        return anchors

    def forward(self, x):
        """Forward pass"""
        # Extract features from backbone
        features = self.backbone(x)

        # Apply FPN
        fpn_features = []
        for i, layer in enumerate(self.fpn):
            if i == 0:
                fpn_features.append(layer(features))
            else:
                # Simplified top-down pathway
                upsampled = torch.nn.functional.interpolate(
                    fpn_features[-1], size=features.shape[-2:], mode='nearest'
                )
                fpn_features.append(layer(features) + upsampled)

        # Apply detection heads to each FPN level
        classifications = []
        regressions = []

        for feat in fpn_features:
            classifications.append(self.classification_head(feat))
            regressions.append(self.regression_head(feat))

        return classifications, regressions

class MultiModalDetector:
    """Multi-modal object detection combining vision and other sensors"""

    def __init__(self):
        self.vision_detector = DETRRobotDetector()
        self.fusion_weights = {
            'vision': 0.7,
            'depth': 0.2,
            'touch': 0.1
        }
        self.confidence_threshold = 0.5

    def detect_with_fusion(self, rgb_image: np.ndarray, depth_image: np.ndarray = None) -> List[DetectionResult]:
        """Detect objects using multi-modal fusion"""
        # Get vision-based detections
        vision_detections = self.vision_detector.detect_objects(rgb_image)

        # If depth image is available, refine detections
        if depth_image is not None:
            refined_detections = self._refine_with_depth(vision_detections, rgb_image, depth_image)
        else:
            refined_detections = vision_detections

        # Filter based on confidence
        final_detections = [
            det for det in refined_detections
            if det.confidence >= self.confidence_threshold
        ]

        return final_detections

    def _refine_with_depth(self, detections: List[DetectionResult], rgb_image: np.ndarray, depth_image: np.ndarray) -> List[DetectionResult]:
        """Refine detections using depth information"""
        refined_detections = []

        for detection in detections:
            # Get bounding box center
            x, y, w, h = detection.bbox
            center_x, center_y = x + w // 2, y + h // 2

            # Get depth at center point
            if center_y < depth_image.shape[0] and center_x < depth_image.shape[1]:
                depth = depth_image[center_y, center_x]

                # Refine confidence based on depth consistency
                # Objects with reasonable depth values get higher confidence
                if 0.1 < depth < 5.0:  # Reasonable depth range for robotics
                    refined_confidence = min(1.0, detection.confidence * 1.2)  # Boost confidence
                else:
                    refined_confidence = max(0.1, detection.confidence * 0.8)  # Reduce confidence

                # Update detection with refined confidence and 3D position
                detection.confidence = refined_confidence
                detection.center_3d = self._get_3d_position(x, y, depth)

            refined_detections.append(detection)

        return refined_detections

    def _get_3d_position(self, x: int, y: int, depth: float) -> Tuple[float, float, float]:
        """Convert 2D pixel coordinates + depth to 3D world coordinates"""
        # This is a simplified version - in practice you'd use calibrated camera parameters
        # Assuming camera intrinsics: fx=554.25, fy=554.25, cx=320, cy=240
        fx, fy, cx, cy = 554.25, 554.25, 320.0, 240.0

        z = depth
        x_3d = (x - cx) * z / fx
        y_3d = (y - cy) * z / fy

        return (x_3d, y_3d, z)

def compare_detection_methods():
    """Compare different detection methods"""
    print("Comparing Object Detection Methods for Robotics")

    # Initialize detectors
    detr_detector = DETRRobotDetector()

    print("\nDETR Detector initialized")
    print("Testing with sample image...")

    # Create a sample image for testing
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test DETR
    start_time = time.time()
    detr_detections = detr_detector.detect_objects(sample_image)
    detr_time = time.time() - start_time

    print(f"DETR: {len(detr_detections)} objects detected in {detr_time:.3f}s")

def demonstrate_transformer_detection():
    """Demonstrate transformer-based detection"""
    print("Demonstrating Transformer-Based Object Detection")

    try:
        # Initialize DETR detector
        detector = DETRRobotDetector()

        # Example usage would be:
        # image = cv2.imread("path_to_image.jpg")
        # detections = detector.detect_objects(image)

        print("DETR detector ready for object detection")
        print("Model: facebook/detr-resnet-50")
        print("Classes: COCO dataset (91 classes)")

    except Exception as e:
        print(f"Error initializing DETR detector: {e}")
        print("Make sure transformers and torch are properly installed")

if __name__ == "__main__":
    demonstrate_transformer_detection()
    compare_detection_methods()
```

## 3D Object Detection and Pose Estimation

### Point Cloud Processing

```python
# python/point_cloud_detection.py
import open3d as o3d
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
import time
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R

class PointCloudObjectDetector:
    """Object detection using 3D point cloud data"""

    def __init__(self):
        self.voxel_size = 0.01  # 1cm voxel size
        self.cluster_eps = 0.02  # 2cm clustering distance
        self.min_points = 10     # Minimum points for a cluster
        self.table_height = 0.0  # Assumed table height for ground removal

    def detect_objects_from_pointcloud(self, pointcloud: np.ndarray) -> List[Dict]:
        """Detect objects from 3D point cloud"""
        start_time = time.time()

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)

        # Downsample point cloud for efficiency
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Remove ground plane (table surface)
        objects_pcd = self._remove_ground_plane(downsampled_pcd)

        # Segment objects using clustering
        object_clusters = self._segment_objects(objects_pcd)

        # Analyze each cluster to extract object information
        objects = []
        for i, cluster_indices in enumerate(object_clusters):
            if len(cluster_indices) >= self.min_points:
                cluster_points = np.asarray(objects_pcd.select_by_index(cluster_indices).points)

                # Calculate object properties
                centroid = np.mean(cluster_points, axis=0)
                bbox = self._calculate_bounding_box(cluster_points)
                dimensions = self._calculate_dimensions(cluster_points)

                object_info = {
                    'id': i,
                    'centroid': centroid,
                    'bbox': bbox,
                    'dimensions': dimensions,
                    'point_count': len(cluster_points),
                    'convex_hull': self._calculate_convex_hull(cluster_points)
                }

                objects.append(object_info)

        detection_time = time.time() - start_time
        print(f"Point cloud detection completed in {detection_time:.3f}s, found {len(objects)} objects")

        return objects

    def _remove_ground_plane(self, pcd):
        """Remove ground plane using RANSAC"""
        # Segment the largest plane (assumed to be ground/table)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )

        # Remove the ground plane points
        objects_pcd = pcd.select_by_index(inliers, invert=True)

        return objects_pcd

    def _segment_objects(self, pcd):
        """Segment individual objects using DBSCAN clustering"""
        points = np.asarray(pcd.points)

        # Perform clustering
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=2).fit(points)
        labels = clustering.labels_

        # Group points by cluster
        unique_labels = set(labels)
        clusters = []

        for label in unique_labels:
            if label == -1:  # Noise points
                continue

            cluster_indices = np.where(labels == label)[0]
            clusters.append(cluster_indices)

        return clusters

    def _calculate_bounding_box(self, points: np.ndarray) -> Dict:
        """Calculate 3D bounding box for a point cluster"""
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)

        center = (min_vals + max_vals) / 2
        size = max_vals - min_vals

        return {
            'center': center,
            'size': size,
            'min': min_vals,
            'max': max_vals
        }

    def _calculate_dimensions(self, points: np.ndarray) -> np.ndarray:
        """Calculate object dimensions using PCA"""
        # Center the points
        centered_points = points - np.mean(points, axis=0)

        # Calculate covariance matrix
        cov_matrix = np.cov(centered_points.T)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalues (largest first)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Dimensions are proportional to sqrt of eigenvalues
        dimensions = np.sqrt(eigenvalues) * 2  # Multiply by 2 for full extent

        return dimensions

    def _calculate_convex_hull(self, points: np.ndarray) -> np.ndarray:
        """Calculate convex hull of point cloud"""
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            return hull.vertices
        except:
            # Return all points if convex hull fails
            return np.arange(len(points))

class RGBDObjectDetector:
    """Object detection using RGB-D data (combining color and depth)"""

    def __init__(self):
        self.depth_threshold = 2.0  # Maximum depth to consider
        self.min_object_size = 100  # Minimum number of pixels for an object
        self.color_threshold = 30   # Color difference threshold for segmentation

    def detect_objects_rgbd(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> List[Dict]:
        """Detect objects using RGB-D data"""
        start_time = time.time()

        # Convert to grayscale for processing
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Create mask for valid depth regions
        valid_depth_mask = (depth_image > 0.1) & (depth_image < self.depth_threshold)

        # Apply depth-based segmentation
        segmented_objects = self._segment_by_depth(depth_image, valid_depth_mask)

        # Refine segments using color information
        refined_objects = []
        for obj_mask in segmented_objects:
            # Combine depth and color segmentation
            combined_mask = obj_mask & valid_depth_mask
            if np.sum(combined_mask) >= self.min_object_size:
                # Extract object properties
                obj_info = self._extract_object_properties(
                    rgb_image, depth_image, combined_mask
                )
                refined_objects.append(obj_info)

        detection_time = time.time() - start_time
        print(f"RGB-D detection completed in {detection_time:.3f}s, found {len(refined_objects)} objects")

        return refined_objects

    def _segment_by_depth(self, depth_image: np.ndarray, valid_mask: np.ndarray) -> List[np.ndarray]:
        """Segment objects based on depth discontinuities"""
        # Calculate depth gradients
        grad_x = np.gradient(depth_image, axis=1)
        grad_y = np.gradient(depth_image, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Create initial segmentation based on depth gradients
        # Use watershed or region growing algorithm
        from scipy import ndimage

        # Mark regions with low gradient as potential objects
        smooth_regions = grad_magnitude < 0.1  # Threshold for smooth regions
        smooth_regions = smooth_regions & valid_mask

        # Label connected components
        labeled_regions, num_regions = ndimage.label(smooth_regions)

        # Create masks for each region
        object_masks = []
        for i in range(1, num_regions + 1):
            region_mask = (labeled_regions == i)
            if np.sum(region_mask) >= self.min_object_size:
                object_masks.append(region_mask)

        return object_masks

    def _extract_object_properties(self, rgb_image: np.ndarray, depth_image: np.ndarray, mask: np.ndarray) -> Dict:
        """Extract properties of an object from RGB-D data"""
        # Get object pixels
        obj_rgb = rgb_image[mask]
        obj_depth = depth_image[mask]

        # Calculate 2D bounding box
        coords = np.argwhere(mask)
        min_row, min_col = coords.min(axis=0)
        max_row, max_col = coords.max(axis=0)

        # Calculate 3D centroid
        y_coords, x_coords = coords.T
        z_values = depth_image[y_coords, x_coords]

        # Convert 2D coordinates to 3D using camera intrinsics
        # Assuming camera intrinsics: fx=554.25, fy=554.25, cx=320, cy=240
        fx, fy, cx, cy = 554.25, 554.25, 320.0, 240.0

        x_3d = (x_coords - cx) * z_values / fx
        y_3d = (y_coords - cy) * z_values / fy

        centroid_3d = np.array([
            np.mean(x_3d),
            np.mean(y_3d),
            np.mean(z_values)
        ])

        # Calculate object dimensions
        dimensions_3d = np.array([
            np.max(x_3d) - np.min(x_3d),
            np.max(y_3d) - np.min(y_3d),
            np.max(z_values) - np.min(z_values)
        ])

        # Calculate average color
        avg_color = np.mean(obj_rgb, axis=0)

        return {
            'bbox_2d': (min_col, min_row, max_col - min_col, max_row - min_row),
            'centroid_3d': centroid_3d,
            'dimensions_3d': dimensions_3d,
            'avg_color': avg_color,
            'pixel_count': len(obj_rgb),
            'confidence': self._calculate_detection_confidence(obj_depth, obj_rgb)
        }

    def _calculate_detection_confidence(self, depth_values: np.ndarray, color_values: np.ndarray) -> float:
        """Calculate confidence score for object detection"""
        # Confidence based on depth consistency and color uniformity
        depth_std = np.std(depth_values)
        color_std = np.std(color_values, axis=0).mean()

        # Lower std = higher confidence
        depth_conf = max(0.1, 1.0 - depth_std)  # Normalize to [0.1, 1.0]
        color_conf = max(0.1, 1.0 - color_std / 255.0)  # Normalize color std

        # Combine confidences
        confidence = (depth_conf + color_conf) / 2.0
        return min(1.0, confidence)

class ObjectPoseEstimator:
    """Estimate 6D pose of objects for manipulation"""

    def __init__(self):
        self.template_models = {}  # Predefined object models
        self.icp_threshold = 0.01  # ICP convergence threshold

    def estimate_pose(self, detected_object: Dict, camera_matrix: np.ndarray) -> Dict:
        """Estimate 6D pose of detected object"""
        # This is a simplified pose estimation
        # In practice, you'd use template matching, PnP, or ICP

        # For now, return a simple pose based on the detected properties
        pose_info = {
            'position': detected_object['centroid_3d'],
            'orientation': self._estimate_orientation(detected_object),  # Simple orientation estimation
            'confidence': detected_object.get('confidence', 0.8),
            'object_type': self._classify_object_type(detected_object)
        }

        return pose_info

    def _estimate_orientation(self, object_info: Dict) -> np.ndarray:
        """Estimate object orientation from dimensions"""
        # Simple heuristic: align longest dimension with Z-axis (upright)
        dimensions = object_info['dimensions_3d']
        max_dim_idx = np.argmax(dimensions)

        # Create a rotation matrix
        rotation = np.eye(3)
        # This is a simplified orientation - in practice you'd use more sophisticated methods
        return rotation.flatten()  # Return as 1D array

    def _classify_object_type(self, object_info: Dict) -> str:
        """Classify object type based on dimensions and other properties"""
        dimensions = object_info['dimensions_3d']
        aspect_ratios = dimensions / np.min(dimensions)

        if aspect_ratios[2] > 3:  # Tall object
            return "tall_object"
        elif aspect_ratios[0] > 2 and aspect_ratios[1] > 2:  # Flat object
            return "flat_object"
        elif np.all(aspect_ratios < 2):  # Cuboid-like
            return "cuboid_object"
        else:
            return "irregular_object"

def demonstrate_point_cloud_detection():
    """Demonstrate point cloud object detection"""
    print("Demonstrating Point Cloud Object Detection")

    # Initialize detector
    detector = PointCloudObjectDetector()

    # Create a sample point cloud (simulated)
    # In practice, this would come from a depth sensor or RGB-D camera
    np.random.seed(42)
    # Simulate a table with some objects on it
    table_points = np.random.uniform(low=[-0.5, -0.5, 0], high=[0.5, 0.5, 0.01], size=(1000, 3))
    object1_points = np.random.uniform(low=[-0.2, -0.2, 0.01], high=[0.0, 0.0, 0.05], size=(200, 3))
    object2_points = np.random.uniform(low=[0.1, 0.1, 0.01], high=[0.3, 0.3, 0.03], size=(150, 3))

    full_pointcloud = np.vstack([table_points, object1_points, object2_points])

    # Detect objects
    objects = detector.detect_objects_from_pointcloud(full_pointcloud)

    print(f"Detected {len(objects)} objects:")
    for i, obj in enumerate(objects):
        print(f"  Object {i+1}: centroid at {obj['centroid']}, dimensions: {obj['dimensions']}")

def demonstrate_rgbd_detection():
    """Demonstrate RGB-D object detection"""
    print("\nDemonstrating RGB-D Object Detection")

    # Initialize detector
    detector = RGBDObjectDetector()

    # Create sample RGB and depth images
    rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth_image = np.random.uniform(0.5, 2.0, (480, 640)).astype(np.float32)

    # Add some "objects" to the depth image
    cv2.circle(depth_image, (100, 100), 30, 0.8, -1)  # Object 1
    cv2.circle(depth_image, (300, 200), 25, 1.2, -1)  # Object 2
    cv2.circle(depth_image, (500, 300), 40, 1.5, -1)  # Object 3

    # Detect objects
    objects = detector.detect_objects_rgbd(rgb_image, depth_image)

    print(f"Detected {len(objects)} objects with RGB-D:")
    for i, obj in enumerate(objects):
        print(f"  Object {i+1}: 3D centroid at {obj['centroid_3d']}, confidence: {obj['confidence']:.2f}")

if __name__ == "__main__":
    demonstrate_point_cloud_detection()
    demonstrate_rgbd_detection()
```

## Manipulation Planning

### Grasp Planning

```python
# python/grasp_planning.py
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass
from enum import Enum

class GraspType(Enum):
    """Types of grasps for different objects"""
    PINCH = "pinch"
    PALM = "palm"
    LATERAL = "lateral"
    SUCTION = "suction"
    SPECIALIZED = "specialized"

@dataclass
class GraspPose:
    """Represents a potential grasp pose"""
    position: np.ndarray  # 3D position [x, y, z]
    orientation: np.ndarray  # 4D quaternion [x, y, z, w]
    grasp_type: GraspType
    quality_score: float
    approach_direction: np.ndarray  # 3D approach vector
    width: float  # Required gripper width

class GraspPlanner:
    """Plan grasps for detected objects"""

    def __init__(self):
        self.min_grasp_quality = 0.3
        self.max_grasp_attempts = 10
        self.gripper_width_range = (0.02, 0.1)  # 2cm to 10cm

    def plan_grasps(self, object_info: Dict, object_pose: Dict) -> List[GraspPose]:
        """Plan grasps for a detected object"""
        start_time = time.time()

        grasps = []

        # Determine object type and plan appropriate grasps
        obj_type = object_pose.get('object_type', 'irregular_object')
        dimensions = object_info.get('dimensions_3d', np.array([0.1, 0.1, 0.1]))

        if obj_type == "tall_object":
            # Plan grasps for tall objects (cylindrical, bottles, etc.)
            grasps.extend(self._plan_cylindrical_grasps(object_pose, dimensions))
        elif obj_type == "flat_object":
            # Plan grasps for flat objects (books, plates, etc.)
            grasps.extend(self._plan_planar_grasps(object_pose, dimensions))
        elif obj_type == "cuboid_object":
            # Plan grasps for box-like objects
            grasps.extend(self._plan_box_grasps(object_pose, dimensions))
        else:
            # Plan generic grasps for irregular objects
            grasps.extend(self._plan_generic_grasps(object_pose, dimensions))

        # Filter grasps by quality
        filtered_grasps = [g for g in grasps if g.quality_score >= self.min_grasp_quality]

        # Sort by quality score (highest first)
        filtered_grasps.sort(key=lambda g: g.quality_score, reverse=True)

        planning_time = time.time() - start_time
        print(f"Grasp planning completed in {planning_time:.3f}s, generated {len(filtered_grasps)} viable grasps")

        return filtered_grasps

    def _plan_cylindrical_grasps(self, object_pose: Dict, dimensions: np.ndarray) -> List[GraspPose]:
        """Plan grasps for cylindrical objects"""
        grasps = []

        # Get object center and orientation
        center = object_pose['position']
        orientation = object_pose.get('orientation', np.array([0, 0, 0, 1]))

        # Calculate grasp positions around the cylinder
        radius = max(dimensions[0], dimensions[1]) / 2
        height = dimensions[2]

        # Plan circumferential grasps
        num_grasps = 8
        for i in range(num_grasps):
            angle = 2 * np.pi * i / num_grasps

            # Position grasp point at the radius
            x_offset = radius * np.cos(angle)
            y_offset = radius * np.sin(angle)

            grasp_pos = np.array([
                center[0] + x_offset,
                center[1] + y_offset,
                center[2] + height / 2  # Middle height
            ])

            # Calculate approach direction (radially inward)
            approach_dir = -np.array([x_offset, y_offset, 0])
            approach_dir = approach_dir / np.linalg.norm(approach_dir)

            # Calculate orientation for pinch grasp
            # Grasp along the cylinder axis
            grasp_orientation = self._calculate_grasp_orientation(
                approach_dir, np.array([0, 0, 1])  # Along Z axis
            )

            grasp_width = min(0.08, radius * 1.5)  # Adaptive gripper width

            grasp = GraspPose(
                position=grasp_pos,
                orientation=grasp_orientation,
                grasp_type=GraspType.PINCH,
                quality_score=self._calculate_grasp_quality(grasp_pos, dimensions, GraspType.PINCH),
                approach_direction=approach_dir,
                width=grasp_width
            )
            grasps.append(grasp)

        return grasps

    def _plan_planar_grasps(self, object_pose: Dict, dimensions: np.ndarray) -> List[GraspPose]:
        """Plan grasps for planar/flat objects"""
        grasps = []

        center = object_pose['position']

        # Plan edge grasps
        thickness = min(dimensions)  # Smallest dimension is thickness
        width = max(dimensions)      # Largest is width
        length = np.sort(dimensions)[1]  # Middle is length

        # Grasp along the four edges
        edge_offsets = [
            np.array([width/2, 0, thickness/2]),   # Right edge
            np.array([-width/2, 0, thickness/2]),  # Left edge
            np.array([0, length/2, thickness/2]),  # Top edge
            np.array([0, -length/2, thickness/2])  # Bottom edge
        ]

        for offset in edge_offsets:
            grasp_pos = center + offset

            # Approach from above (perpendicular to the plane)
            approach_dir = np.array([0, 0, -1])  # From above

            # Calculate orientation for lateral grasp
            grasp_orientation = self._calculate_grasp_orientation(
                approach_dir, np.array([1, 0, 0])  # Gripper fingers horizontal
            )

            grasp_width = min(0.06, length * 0.7)  # Adaptive width

            grasp = GraspPose(
                position=grasp_pos,
                orientation=grasp_orientation,
                grasp_type=GraspType.LATERAL,
                quality_score=self._calculate_grasp_quality(grasp_pos, dimensions, GraspType.LATERAL),
                approach_direction=approach_dir,
                width=grasp_width
            )
            grasps.append(grasp)

        return grasps

    def _plan_box_grasps(self, object_pose: Dict, dimensions: np.ndarray) -> List[GraspPose]:
        """Plan grasps for box-like objects"""
        grasps = []

        center = object_pose['position']

        # Plan corner grasps
        corner_offsets = [
            np.array([dimensions[0]/2, dimensions[1]/2, 0]),   # Top-right corner
            np.array([dimensions[0]/2, -dimensions[1]/2, 0]),  # Bottom-right corner
            np.array([-dimensions[0]/2, dimensions[1]/2, 0]),  # Top-left corner
            np.array([-dimensions[0]/2, -dimensions[1]/2, 0])  # Bottom-left corner
        ]

        for offset in corner_offsets:
            grasp_pos = center + offset

            # Approach from above
            approach_dir = np.array([0, 0, -1])

            # Calculate orientation for corner grasp
            grasp_orientation = self._calculate_grasp_orientation(
                approach_dir, np.array([0, 1, 0])  # Gripper fingers along Y
            )

            grasp_width = min(0.08, max(dimensions[0], dimensions[1]) * 0.8)

            grasp = GraspPose(
                position=grasp_pos,
                orientation=grasp_orientation,
                grasp_type=GraspType.PALM,
                quality_score=self._calculate_grasp_quality(grasp_pos, dimensions, GraspType.PALM),
                approach_direction=approach_dir,
                width=grasp_width
            )
            grasps.append(grasp)

        # Plan face grasps
        face_centers = [
            np.array([0, 0, dimensions[2]/2]),      # Top face
            np.array([dimensions[0]/2, 0, 0]),      # Right face
            np.array([-dimensions[0]/2, 0, 0]),     # Left face
            np.array([0, dimensions[1]/2, 0]),      # Front face
            np.array([0, -dimensions[1]/2, 0])      # Back face
        ]

        for offset in face_centers:
            grasp_pos = center + offset

            # Approach direction normal to the face
            if abs(offset[2]) > 0:  # Top face
                approach_dir = np.array([0, 0, -1])
            elif abs(offset[0]) > 0:  # Side faces
                approach_dir = np.array([-np.sign(offset[0]), 0, 0])
            else:  # Front/back faces
                approach_dir = np.array([0, -np.sign(offset[1]), 0])

            # Calculate orientation for face grasp
            grasp_orientation = self._calculate_grasp_orientation(
                approach_dir, np.array([1, 0, 0])
            )

            grasp_width = min(0.08, max(dimensions[0], dimensions[1]) * 0.6)

            grasp = GraspPose(
                position=grasp_pos,
                orientation=grasp_orientation,
                grasp_type=GraspType.PALM,
                quality_score=self._calculate_grasp_quality(grasp_pos, dimensions, GraspType.PALM),
                approach_direction=approach_dir,
                width=grasp_width
            )
            grasps.append(grasp)

        return grasps

    def _plan_generic_grasps(self, object_pose: Dict, dimensions: np.ndarray) -> List[GraspPose]:
        """Plan generic grasps for irregular objects"""
        grasps = []

        center = object_pose['position']

        # Plan center grasp
        grasp_pos = center + np.array([0, 0, max(dimensions) / 2])  # Above center

        approach_dir = np.array([0, 0, -1])  # From above
        grasp_orientation = self._calculate_grasp_orientation(
            approach_dir, np.array([0, 1, 0])
        )

        grasp_width = min(0.1, np.mean(dimensions) * 2)

        grasp = GraspPose(
            position=grasp_pos,
            orientation=grasp_orientation,
            grasp_type=GraspType.PALM,
            quality_score=self._calculate_grasp_quality(grasp_pos, dimensions, GraspType.PALM),
            approach_direction=approach_dir,
            width=grasp_width
        )
        grasps.append(grasp)

        return grasps

    def _calculate_grasp_orientation(self, approach_dir: np.ndarray, grasp_axis: np.ndarray) -> np.ndarray:
        """Calculate gripper orientation based on approach direction and grasp axis"""
        # Normalize input vectors
        approach = approach_dir / np.linalg.norm(approach_dir)
        grasp_ax = grasp_axis / np.linalg.norm(grasp_axis)

        # Calculate rotation to align z-axis with approach direction
        z_axis = approach
        y_axis = grasp_ax - np.dot(grasp_ax, z_axis) * z_axis
        y_axis = y_axis / np.linalg.norm(y_axis) if np.linalg.norm(y_axis) > 0.001 else np.array([1, 0, 0])
        x_axis = np.cross(y_axis, z_axis)

        # Create rotation matrix
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

        # Convert to quaternion
        quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)

        return quaternion

    def _rotation_matrix_to_quaternion(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion"""
        # Method to convert rotation matrix to quaternion
        trace = np.trace(rotation_matrix)

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
                qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                qx = 0.25 * s
                qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
                qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qy = 0.25 * s
                qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
                qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                qz = 0.25 * s

        return np.array([qx, qy, qz, qw])

    def _calculate_grasp_quality(self, position: np.ndarray, dimensions: np.ndarray, grasp_type: GraspType) -> float:
        """Calculate quality score for a grasp"""
        # Quality factors:
        # 1. Object size appropriateness
        # 2. Grasp stability
        # 3. Accessibility

        size_factor = min(1.0, np.mean(dimensions) / 0.1)  # Normalize by typical object size
        stability_factor = 0.8 if grasp_type in [GraspType.PALM, GraspType.PINCH] else 0.6
        accessibility_factor = 0.9  # Assume good accessibility for now

        quality = (size_factor * 0.4 + stability_factor * 0.4 + accessibility_factor * 0.2)
        return min(1.0, max(0.1, quality))  # Clamp between 0.1 and 1.0

class ManipulationPlanner:
    """Plan complete manipulation sequences"""

    def __init__(self):
        self.grasp_planner = GraspPlanner()
        self.max_retries = 3

    def plan_manipulation(self, object_info: Dict, object_pose: Dict, target_pose: np.ndarray) -> Dict:
        """Plan complete manipulation sequence: approach, grasp, lift, transport, place"""
        start_time = time.time()

        # Plan grasps for the object
        grasps = self.grasp_planner.plan_grasps(object_info, object_pose)

        if not grasps:
            return {
                'success': False,
                'error': 'No viable grasps found',
                'actions': []
            }

        # Select the best grasp
        best_grasp = grasps[0]

        # Plan manipulation sequence
        manipulation_sequence = self._create_manipulation_sequence(
            object_pose['position'],
            best_grasp,
            target_pose
        )

        planning_time = time.time() - start_time
        print(f"Manipulation planning completed in {planning_time:.3f}s")

        return {
            'success': True,
            'grasp_pose': best_grasp,
            'sequence': manipulation_sequence,
            'planning_time': planning_time
        }

    def _create_manipulation_sequence(self, object_pos: np.ndarray, grasp: GraspPose, target_pos: np.ndarray) -> List[Dict]:
        """Create a complete manipulation sequence"""
        sequence = []

        # 1. Approach object
        approach_pos = grasp.position + grasp.approach_direction * 0.1  # 10cm above grasp point
        sequence.append({
            'action': 'move_to',
            'position': approach_pos,
            'orientation': grasp.orientation,
            'description': 'Approach object'
        })

        # 2. Descend to grasp
        sequence.append({
            'action': 'move_to',
            'position': grasp.position,
            'orientation': grasp.orientation,
            'description': 'Move to grasp position'
        })

        # 3. Close gripper
        sequence.append({
            'action': 'grasp',
            'width': grasp.width,
            'description': 'Close gripper'
        })

        # 4. Lift object
        lift_pos = grasp.position + np.array([0, 0, 0.1])  # Lift 10cm
        sequence.append({
            'action': 'move_to',
            'position': lift_pos,
            'orientation': grasp.orientation,
            'description': 'Lift object'
        })

        # 5. Move to target
        sequence.append({
            'action': 'move_to',
            'position': target_pos + np.array([0, 0, 0.1]),  # 10cm above target
            'orientation': grasp.orientation,
            'description': 'Move to target location'
        })

        # 6. Descend to place
        place_pos = target_pos + np.array([0, 0, 0.05])  # 5cm above target
        sequence.append({
            'action': 'move_to',
            'position': place_pos,
            'orientation': grasp.orientation,
            'description': 'Move to place position'
        })

        # 7. Open gripper
        sequence.append({
            'action': 'release',
            'description': 'Open gripper'
        })

        # 8. Retract
        retract_pos = place_pos + np.array([0, 0, 0.1])  # Retract 10cm
        sequence.append({
            'action': 'move_to',
            'position': retract_pos,
            'orientation': grasp.orientation,
            'description': 'Retract gripper'
        })

        return sequence

def demonstrate_grasp_planning():
    """Demonstrate grasp planning"""
    print("Demonstrating Grasp Planning for Robotics")

    # Initialize planner
    planner = ManipulationPlanner()

    # Example object info (from detection)
    object_info = {
        'dimensions_3d': np.array([0.08, 0.08, 0.15]),  # 8x8x15 cm object
        'centroid_3d': np.array([0.5, 0.0, 0.1])      # Position in world coordinates
    }

    # Example object pose (from pose estimation)
    object_pose = {
        'position': np.array([0.5, 0.0, 0.1]),
        'orientation': np.array([0, 0, 0, 1]),  # Identity quaternion
        'object_type': 'cuboid_object',
        'confidence': 0.9
    }

    # Target position for placing the object
    target_position = np.array([0.8, 0.3, 0.1])

    # Plan manipulation
    result = planner.plan_manipulation(object_info, object_pose, target_position)

    if result['success']:
        print(f"Manipulation plan generated successfully!")
        print(f"Best grasp quality: {result['grasp_pose'].quality_score:.3f}")
        print(f"Grasp type: {result['grasp_pose'].grasp_type.value}")
        print(f"Sequence has {len(result['sequence'])} actions:")

        for i, action in enumerate(result['sequence']):
            print(f"  {i+1}. {action['description']}")
            if 'position' in action:
                print(f"     Position: [{action['position'][0]:.3f}, {action['position'][1]:.3f}, {action['position'][2]:.3f}]")
    else:
        print(f"Manipulation planning failed: {result['error']}")

if __name__ == "__main__":
    demonstrate_grasp_planning()
```

## ROS 2 Integration for Object Detection and Manipulation

### Action Servers and Clients

```python
# python/ros2_integration.py
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion
from builtin_interfaces.msg import Duration

# Custom message types would be defined here
# For this example, we'll define simplified versions

class ObjectDetectionActionServer(Node):
    """ROS 2 action server for object detection"""

    def __init__(self):
        super().__init__('object_detection_action_server')

        # Initialize object detector
        self.object_detector = YOLORobotDetector()  # From previous implementation
        self.camera_info = None

        # Setup action server
        self._action_server = ActionServer(
            self,
            # Define custom action type
            'DetectObjects',  # This would be your custom action
            'detect_objects',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Setup camera info subscriber
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera_info',
            self.camera_info_callback,
            QoSProfile(depth=1)
        )

        # Setup image subscriber for continuous detection
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            QoSProfile(depth=1)
        )

        self.get_logger().info("Object Detection Action Server initialized")

    def goal_callback(self, goal_request):
        """Accept or reject goal requests"""
        self.get_logger().info(f"Received object detection goal")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel requests"""
        self.get_logger().info("Received cancel request for object detection")
        return CancelResponse.ACCEPT

    def camera_info_callback(self, msg):
        """Update camera calibration info"""
        self.camera_info = msg

    def image_callback(self, msg):
        """Process images for continuous detection"""
        # Convert ROS Image to OpenCV
        # This would convert the image and run detection
        # For now, we'll just log the callback
        pass

    async def execute_callback(self, goal_handle):
        """Execute object detection goal"""
        self.get_logger().info("Executing object detection...")

        feedback_msg = None  # Define feedback message
        result = None  # Define result message

        try:
            # Process the detection request
            # This would involve getting an image and running detection
            image = self.get_latest_image()  # Implementation would get latest image
            detections = self.object_detector.detect_objects(image)

            # Format results
            detection_results = []
            for detection in detections:
                detection_results.append({
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'bbox': detection.bbox,
                    'center_3d': detection.center_3d
                })

            # Publish result
            goal_handle.succeed()
            # result = YourResultType(detections=detection_results)

        except Exception as e:
            self.get_logger().error(f"Error in object detection: {e}")
            goal_handle.abort()
            # result = YourResultType(detections=[])

        return result

    def get_latest_image(self):
        """Get the latest image from camera"""
        # Implementation would return latest image
        # For now, return a dummy image
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

class ManipulationActionServer(Node):
    """ROS 2 action server for manipulation planning and execution"""

    def __init__(self):
        super().__init__('manipulation_action_server')

        # Initialize manipulation planner
        self.manipulation_planner = ManipulationPlanner()

        # Setup action server
        self._action_server = ActionServer(
            self,
            # Define custom action type
            'ExecuteManipulation',  # This would be your custom action
            'execute_manipulation',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Publishers for robot control
        self.joint_trajectory_pub = self.create_publisher(
            # JointTrajectory,  # Replace with actual message type
            'joint_trajectory',
            QoSProfile(depth=1)
        )

        self.gripper_pub = self.create_publisher(
            # GripperCommand,  # Replace with actual message type
            'gripper_command',
            QoSProfile(depth=1)
        )

        self.get_logger().info("Manipulation Action Server initialized")

    def goal_callback(self, goal_request):
        """Accept or reject manipulation goals"""
        self.get_logger().info(f"Received manipulation goal: {goal_request.task_type}")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel requests"""
        self.get_logger().info("Received cancel request for manipulation")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute manipulation goal"""
        self.get_logger().info("Executing manipulation task...")

        try:
            # Plan manipulation based on goal
            if goal_handle.request.task_type == 'grasp_object':
                result = await self._execute_grasp_task(goal_handle.request)
            elif goal_handle.request.task_type == 'place_object':
                result = await self._execute_place_task(goal_handle.request)
            elif goal_handle.request.task_type == 'transport_object':
                result = await self._execute_transport_task(goal_handle.request)
            else:
                result = await self._execute_custom_task(goal_handle.request)

            goal_handle.succeed()
            return result

        except Exception as e:
            self.get_logger().error(f"Error in manipulation: {e}")
            goal_handle.abort()
            # return YourResultType(success=False, error=str(e))

    async def _execute_grasp_task(self, request):
        """Execute grasp task"""
        # Get object information from object detection
        # Plan grasp
        # Execute grasp sequence
        pass

    async def _execute_place_task(self, request):
        """Execute place task"""
        # Plan placement
        # Execute placement sequence
        pass

    async def _execute_transport_task(self, request):
        """Execute transport task"""
        # Plan transport from source to destination
        # Execute transport sequence
        pass

    async def _execute_custom_task(self, request):
        """Execute custom manipulation task"""
        # Handle custom manipulation tasks
        pass

class ObjectManipulationClient(Node):
    """Client for object detection and manipulation"""

    def __init__(self):
        super().__init__('object_manipulation_client')

        # Action clients
        self.detection_client = rclpy.action.ActionClient(
            self,
            # Your detection action type
            'DetectObjects',
            'detect_objects'
        )

        self.manipulation_client = rclpy.action.ActionClient(
            self,
            # Your manipulation action type
            'ExecuteManipulation',
            'execute_manipulation'
        )

        # Publishers for commands
        self.command_pub = self.create_publisher(String, 'manipulation_command', 10)

        self.get_logger().info("Object Manipulation Client initialized")

    def detect_objects(self, timeout_sec: float = 10.0):
        """Send object detection request"""
        goal_msg = None  # Your goal message

        if not self.detection_client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error("Detection action server not available")
            return None

        send_goal_future = self.detection_client.send_goal_async(
            goal_msg,
            feedback_callback=self.detection_feedback_callback
        )

        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().info("Detection goal rejected")
            return None

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)

        result = get_result_future.result().result
        return result

    def execute_manipulation(self, task_type: str, object_info: Dict, target_pose: Dict, timeout_sec: float = 30.0):
        """Send manipulation execution request"""
        goal_msg = None  # Your manipulation goal message

        if not self.manipulation_client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error("Manipulation action server not available")
            return None

        send_goal_future = self.manipulation_client.send_goal_async(
            goal_msg,
            feedback_callback=self.manipulation_feedback_callback
        )

        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().info("Manipulation goal rejected")
            return None

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)

        result = get_result_future.result().result
        return result

    def detection_feedback_callback(self, feedback_msg):
        """Handle detection feedback"""
        self.get_logger().info(f"Detection feedback: {feedback_msg}")

    def manipulation_feedback_callback(self, feedback_msg):
        """Handle manipulation feedback"""
        self.get_logger().info(f"Manipulation feedback: {feedback_msg}")

def main(args=None):
    """Main function to run the object detection and manipulation nodes"""
    rclpy.init(args=args)

    # Create nodes
    detection_server = ObjectDetectionActionServer()
    manipulation_server = ManipulationActionServer()
    client = ObjectManipulationClient()

    # Use multi-threaded executor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(detection_server)
    executor.add_node(manipulation_server)
    executor.add_node(client)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        detection_server.destroy_node()
        manipulation_server.destroy_node()
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Real-time Performance Optimization

### Optimized Detection Pipeline

```python
# python/performance_optimization.py
import torch
import numpy as np
import cv2
import time
from typing import List, Dict, Any, Callable
import threading
from queue import Queue, Empty
import multiprocessing as mp
from functools import wraps
import psutil
import os

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {(end - start) * 1000:.2f}ms")
        return result
    return wrapper

class OptimizedObjectDetectionPipeline:
    """Optimized pipeline for real-time object detection"""

    def __init__(self, model_path: str = "yolov5s.pt", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

        # Load model with optimizations
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.to(self.device)
        self.model.eval()

        # Apply optimizations
        if self.device == "cuda":
            self.model.half()  # Use half precision on GPU
            torch.backends.cudnn.benchmark = True

        # Pipeline components
        self.input_queue = Queue(maxsize=2)  # Limit queue size to prevent memory buildup
        self.output_queue = Queue(maxsize=2)
        self.is_running = False
        self.processing_thread = None

        # Frame skipping for performance
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0

        # Statistics
        self.stats = {
            'processed_frames': 0,
            'average_fps': 0.0,
            'average_detection_time': 0.0,
            'memory_usage': 0.0
        }

    def start_pipeline(self):
        """Start the detection pipeline"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._pipeline_loop, daemon=True)
        self.processing_thread.start()

    def stop_pipeline(self):
        """Stop the detection pipeline"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()

    def submit_frame(self, frame: np.ndarray) -> bool:
        """Submit a frame for processing"""
        try:
            self.input_queue.put_nowait(frame)
            return True
        except:
            # Queue is full, drop frame
            return False

    def get_results(self) -> List[DetectionResult]:
        """Get detection results"""
        try:
            return self.output_queue.get_nowait()
        except Empty:
            return []

    def _pipeline_loop(self):
        """Main pipeline processing loop"""
        while self.is_running:
            try:
                # Get frame from input queue
                frame = self.input_queue.get(timeout=0.01)

                # Apply frame skipping
                self.frame_count += 1
                if self.frame_count % self.frame_skip != 0:
                    continue

                # Process frame
                start_time = time.time()
                detections = self._process_frame_optimized(frame)
                processing_time = time.time() - start_time

                # Update statistics
                self.stats['processed_frames'] += 1
                self.stats['average_detection_time'] = (
                    (self.stats['average_detection_time'] * (self.stats['processed_frames'] - 1) + processing_time) /
                    self.stats['processed_frames']
                )

                # Calculate FPS
                if processing_time > 0:
                    current_fps = 1.0 / processing_time
                    if self.stats['average_fps'] == 0:
                        self.stats['average_fps'] = current_fps
                    else:
                        # Exponential moving average
                        self.stats['average_fps'] = 0.9 * self.stats['average_fps'] + 0.1 * current_fps

                # Put results in output queue
                try:
                    self.output_queue.put_nowait(detections)
                except:
                    # Output queue full, results are lost
                    pass

            except Empty:
                continue  # No frame available, continue loop
            except Exception as e:
                print(f"Error in pipeline: {e}")
                continue

    def _process_frame_optimized(self, frame: np.ndarray) -> List[DetectionResult]:
        """Optimized frame processing"""
        # Resize frame to model input size (smaller = faster)
        h, w = frame.shape[:2]
        target_size = 416  # Smaller than standard 640 for speed
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        # Pad to make it square
        delta_w = target_size - new_w
        delta_h = target_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114]
        )

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            results = self.model(img_tensor)

        # Process results
        detections = self._process_detections(results, frame.shape[:2])
        return detections

    def _process_detections(self, results, image_shape) -> List[DetectionResult]:
        """Process YOLO detection results"""
        detections = []
        pred = results.pred[0]

        if len(pred) > 0:
            for *xyxy, conf, cls in pred.tolist():
                x1, y1, x2, y2 = map(int, xyxy)
                width = x2 - x1
                height = y2 - y1

                if conf >= 0.5:  # Confidence threshold
                    detection = DetectionResult(
                        class_id=int(cls),
                        class_name=self.model.names[int(cls)],
                        confidence=conf,
                        bbox=(x1, y1, width, height),
                        center_3d=None,
                        rotation_3d=None
                    )
                    detections.append(detection)

        return detections

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        process = psutil.Process(os.getpid())
        self.stats['memory_usage'] = process.memory_info().rss / 1024 / 1024  # MB
        return self.stats.copy()

class MultiProcessDetection:
    """Multi-process object detection for maximum performance"""

    def __init__(self, num_processes: int = 2):
        self.num_processes = num_processes
        self.processes = []
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.is_running = False

    def start_detection(self):
        """Start multi-process detection"""
        self.is_running = True

        # Start worker processes
        for i in range(self.num_processes):
            p = mp.Process(target=self._worker_process, args=(i,))
            p.start()
            self.processes.append(p)

    def stop_detection(self):
        """Stop multi-process detection"""
        self.is_running = False

        # Send stop signals to all processes
        for _ in range(self.num_processes):
            self.input_queue.put(None)

        # Wait for processes to finish
        for p in self.processes:
            p.join()

    def _worker_process(self, worker_id: int):
        """Worker process for detection"""
        # Load model in each process
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()

        while self.is_running:
            try:
                frame = self.input_queue.get(timeout=1.0)

                if frame is None:  # Stop signal
                    break

                # Process frame
                detections = self._process_frame(model, frame)

                # Put results in output queue
                self.output_queue.put((worker_id, detections))

            except:
                continue

    def _process_frame(self, model, frame: np.ndarray):
        """Process a single frame with the model"""
        # Preprocess frame
        img_tensor = self._preprocess_frame(frame)

        # Run inference
        with torch.no_grad():
            results = model(img_tensor)

        # Process results (simplified)
        return results

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input"""
        # Resize and normalize
        resized = cv2.resize(frame, (640, 640))
        img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

class AdaptiveDetectionOptimizer:
    """Adaptively optimize detection based on system load"""

    def __init__(self):
        self.current_fps = 30.0
        self.target_fps = 30.0
        self.confidence_threshold = 0.5
        self.input_size = 416  # Starting input size
        self.max_input_size = 640
        self.min_input_size = 320

    def adjust_parameters(self):
        """Adjust detection parameters based on performance"""
        # Monitor system resources
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        # Adjust parameters based on load
        if cpu_percent > 80 or memory_percent > 80:
            # Reduce input size to improve performance
            if self.input_size > self.min_input_size:
                self.input_size -= 32
                self.confidence_threshold += 0.05  # Increase threshold to reduce detections
        elif cpu_percent < 50 and memory_percent < 60:
            # Increase input size for better accuracy
            if self.input_size < self.max_input_size:
                self.input_size += 32
                self.confidence_threshold -= 0.05  # Decrease threshold to allow more detections

        # Adjust frame skip based on FPS
        if self.current_fps < self.target_fps * 0.8:
            # Too slow, increase frame skipping
            pass  # Would implement frame skipping logic here
        elif self.current_fps > self.target_fps * 1.2:
            # Too fast, reduce frame skipping
            pass

    def get_current_settings(self) -> Dict[str, Any]:
        """Get current optimization settings"""
        return {
            'input_size': self.input_size,
            'confidence_threshold': self.confidence_threshold,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }

def benchmark_optimizations():
    """Benchmark different optimization strategies"""
    print("Benchmarking Object Detection Optimizations...")

    # Create a sample image for testing
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test different input sizes
    input_sizes = [320, 416, 640]

    print("\nTesting different input sizes:")
    for size in input_sizes:
        start_time = time.time()

        # Process multiple frames to get average time
        for _ in range(10):
            h, w = sample_image.shape[:2]
            scale = min(size / h, size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(sample_image, (new_w, new_h))

            # Pad to make it square
            delta_w = size - new_w
            delta_h = size - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114])

        elapsed = time.time() - start_time
        avg_time = elapsed / 10 * 1000  # Convert to ms

        print(f"  Input size {size}x{size}: {avg_time:.2f}ms per frame")

    print("\nOptimization benchmarking completed.")

def demonstrate_optimized_pipeline():
    """Demonstrate optimized detection pipeline"""
    print("Demonstrating Optimized Object Detection Pipeline")

    try:
        # Initialize optimized pipeline
        pipeline = OptimizedObjectDetectionPipeline(model_path="yolov5s.pt")

        # Start pipeline
        pipeline.start_pipeline()

        # Simulate frame submission
        for i in range(100):  # Process 100 frames
            sample_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            pipeline.submit_frame(sample_frame)

            # Get results occasionally
            if i % 10 == 0:
                results = pipeline.get_results()
                stats = pipeline.get_performance_stats()
                print(f"Frame {i}: FPS={stats['average_fps']:.1f}, Detection time={stats['average_detection_time']*1000:.1f}ms")

        # Get final stats
        final_stats = pipeline.get_performance_stats()
        print(f"\nFinal Performance Stats:")
        print(f"  Average FPS: {final_stats['average_fps']:.2f}")
        print(f"  Average detection time: {final_stats['average_detection_time']*1000:.2f}ms")
        print(f"  Memory usage: {final_stats['memory_usage']:.2f} MB")

        # Stop pipeline
        pipeline.stop_pipeline()

    except Exception as e:
        print(f"Error in optimized pipeline: {e}")
        print("Make sure YOLOv5 is properly installed and model file exists")

if __name__ == "__main__":
    benchmark_optimizations()
    demonstrate_optimized_pipeline()
```

## Best Practices for Object Detection and Manipulation

### Design Guidelines

1. **Multi-Modal Fusion**: Combine vision, depth, and tactile sensors for robust detection
2. **Real-time Performance**: Optimize for the robot's control loop frequency
3. **Safety First**: Implement safety checks before manipulation attempts
4. **Context Awareness**: Use scene context to improve detection accuracy
5. **Adaptive Parameters**: Adjust detection parameters based on computational resources

### Performance Considerations

- **Model Selection**: Choose models that balance accuracy with speed requirements
- **Resolution Trade-offs**: Adjust input resolution based on required accuracy
- **Batch Processing**: Process multiple frames in batches when possible
- **Hardware Acceleration**: Utilize GPU and specialized hardware when available
- **Memory Management**: Implement efficient memory usage for continuous operation

## Hands-On Exercise

### Exercise: Building an Object Detection and Manipulation System

1. **Setup Detection Pipeline**
   - Install and configure YOLO or DETR models
   - Implement real-time detection pipeline
   - Add performance optimization techniques

2. **Implement 3D Processing**
   - Integrate depth information with RGB detection
   - Implement point cloud processing for object segmentation
   - Add pose estimation capabilities

3. **Develop Grasp Planning**
   - Create grasp pose generation algorithms
   - Implement quality assessment for grasps
   - Add safety checks and validation

4. **Integrate with ROS 2**
   - Create action servers for detection and manipulation
   - Implement client interfaces for command submission
   - Add monitoring and feedback mechanisms

5. **Test and Optimize**
   - Test with various objects and scenarios
   - Optimize for real-time performance
   - Validate safety and reliability

## Summary

Object detection and manipulation pipelines form the foundation of robotic interaction with the environment. By combining advanced computer vision techniques with intelligent grasp planning and real-time optimization, we can create robust systems that enable humanoid robots to perceive, understand, and interact with objects in their environment. The key is balancing accuracy with performance while maintaining safety and reliability. Proper integration with robotic control systems and optimization for real-time operation are essential for practical deployment.

## Learning Path Adjustment

Based on your experience level, you may want to focus on:

- **Beginner**: Focus on basic object detection implementation and simple grasp planning
- **Intermediate**: Dive deeper into 3D processing and multi-modal fusion
- **Advanced**: Explore custom model training and advanced manipulation strategies