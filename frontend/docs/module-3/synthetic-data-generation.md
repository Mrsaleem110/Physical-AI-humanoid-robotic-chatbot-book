---
sidebar_position: 4
---

# Isaac Sim Synthetic Data Generation

## Chapter Objectives

- Understand synthetic data generation principles in Isaac Sim
- Create diverse training datasets for AI models
- Implement domain randomization techniques
- Generate multi-modal sensor data
- Optimize data generation pipelines for efficiency

## Introduction to Synthetic Data Generation

Synthetic data generation is a critical component of modern AI development, especially for robotics applications where real-world data collection can be expensive, time-consuming, or dangerous. Isaac Sim provides powerful tools for generating high-quality synthetic data that can be used to train computer vision, perception, and control models.

### Why Synthetic Data for Robotics?

1. **Safety**: Train models on dangerous scenarios without risk
2. **Cost-Effectiveness**: Reduce need for expensive real-world data collection
3. **Variety**: Generate diverse scenarios and edge cases
4. **Annotation**: Perfect ground truth annotations automatically
5. **Control**: Precise control over environmental conditions
6. **Scalability**: Generate large datasets quickly

### Isaac Sim's Synthetic Data Capabilities

- **Photorealistic Rendering**: NVIDIA RTX technology for realistic images
- **Multi-Modal Sensors**: Cameras, LiDAR, Radar, IMU, Force/Torque sensors
- **Domain Randomization**: Systematic variation of appearance and physics
- **Automatic Annotation**: Semantic segmentation, instance segmentation, depth maps
- **Large-Scale Generation**: Distributed rendering capabilities

## Synthetic Data Generation Pipeline

### Basic Data Generation Framework

```python
# python/synthetic_data_generator.py
import omni
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdPhysics
import numpy as np
import cv2
from PIL import Image
import json
import os
import random
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor import LidarRtx
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.nucleus import get_assets_root_path
import carb
from typing import List, Dict, Tuple, Optional

class SyntheticDataGenerator:
    def __init__(self, output_dir: str = "synthetic_data", dataset_name: str = "robotics_dataset"):
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.world = World(stage_units_in_meters=1.0)

        # Create output directories
        self.data_dirs = {
            'images': os.path.join(output_dir, 'images'),
            'labels': os.path.join(output_dir, 'labels'),
            'depth': os.path.join(output_dir, 'depth'),
            'semantic': os.path.join(output_dir, 'semantic'),
            'instances': os.path.join(output_dir, 'instances'),
            'metadata': os.path.join(output_dir, 'metadata'),
            'lidar': os.path.join(output_dir, 'lidar'),
            'camera_params': os.path.join(output_dir, 'camera_params')
        }

        for dir_path in self.data_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        # Initialize components
        self.cameras = []
        self.lidars = []
        self.objects = []
        self.lighting_configs = []
        self.material_configs = []

        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'generation_time': 0.0,
            'data_types': {},
            'object_counts': {}
        }

        self.get_logger().info(f"Synthetic Data Generator initialized for dataset: {dataset_name}")

    def setup_scene(self, scene_config: Dict):
        """Setup the scene based on configuration"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Setup lighting
        self.setup_lighting(scene_config.get('lighting', {}))

        # Add objects
        self.add_objects(scene_config.get('objects', []))

        # Setup sensors
        self.setup_sensors(scene_config.get('sensors', []))

    def setup_lighting(self, lighting_config: Dict):
        """Setup lighting with randomization"""
        # Create dome light
        dome_light_path = "/World/DomeLight"
        omni.kit.commands.execute(
            "CreateDomeLightCommand",
            path=dome_light_path,
            create_xform=True
        )

        dome_light = get_prim_at_path(dome_light_path)
        if dome_light.IsValid():
            # Randomize dome light color and intensity
            base_color = lighting_config.get('dome_color', (0.2, 0.2, 0.2))
            intensity_range = lighting_config.get('dome_intensity_range', (500, 3000))

            dome_light.GetAttribute("inputs:color").Set(base_color)
            intensity = random.uniform(*intensity_range)
            dome_light.GetAttribute("inputs:intensity").Set(intensity)

        # Add directional lights with randomization
        num_directional_lights = lighting_config.get('num_directional_lights', 1)
        for i in range(num_directional_lights):
            light_path = f"/World/DirectionalLight_{i}"
            omni.kit.commands.execute(
                "CreateLightCommand",
                path=light_path,
                light_type="DistantLight"
            )

            light = get_prim_at_path(light_path)
            if light.IsValid():
                # Randomize light properties
                color = self.randomize_color(lighting_config.get('light_color_range', [(0.8, 0.8, 0.8), (1.0, 1.0, 1.0)]))
                intensity = random.uniform(*lighting_config.get('light_intensity_range', (1000, 5000))
                direction = self.randomize_direction()

                light.GetAttribute("inputs:color").Set(color)
                light.GetAttribute("inputs:intensity").Set(intensity)
                light.GetAttribute("xformOp:rotateXYZ").Set(direction)

    def add_objects(self, object_configs: List[Dict]):
        """Add objects to the scene with randomization"""
        for config in object_configs:
            obj_type = config.get('type', 'cube')
            obj_name = config.get('name', f"object_{len(self.objects)}")
            obj_path = f"/World/{obj_name}"

            # Create object based on type
            if obj_type == 'cube':
                obj_prim = UsdGeom.Cube.Define(self.stage, obj_path)
                obj_prim.GetSizeAttr().Set(1.0)
            elif obj_type == 'sphere':
                obj_prim = UsdGeom.Sphere.Define(self.stage, obj_path)
                obj_prim.GetRadiusAttr().Set(0.5)
            elif obj_type == 'cylinder':
                obj_prim = UsdGeom.Cylinder.Define(self.stage, obj_path)
                obj_prim.GetRadiusAttr().Set(0.3)
                obj_prim.GetHeightAttr().Set(1.0)
            else:
                # Default to cube
                obj_prim = UsdGeom.Cube.Define(self.stage, obj_path)
                obj_prim.GetSizeAttr().Set(1.0)

            # Randomize position
            pos_range = config.get('position_range', [(-5, -5, 0), (5, 5, 2)])
            pos = [
                random.uniform(pos_range[0][0], pos_range[1][0]),
                random.uniform(pos_range[0][1], pos_range[1][1]),
                random.uniform(pos_range[0][2], pos_range[1][2])
            ]

            xform = UsdGeom.Xformable(obj_prim)
            xform.AddTranslateOp().Set(Gf.Vec3f(*pos))

            # Randomize scale
            scale_range = config.get('scale_range', (0.5, 2.0))
            scale_val = random.uniform(*scale_range)
            xform.AddScaleOp().Set(Gf.Vec3f(scale_val, scale_val, scale_val))

            # Apply random material
            self.apply_random_material(obj_path, config.get('material_config', {}))

            # Add to objects list
            self.objects.append({
                'path': obj_path,
                'type': obj_type,
                'config': config
            })

    def setup_sensors(self, sensor_configs: List[Dict]):
        """Setup sensors for data collection"""
        for config in sensor_configs:
            sensor_type = config.get('type', 'camera')
            sensor_name = config.get('name', f"sensor_{len(self.cameras) + len(self.lidars)}")

            if sensor_type == 'camera':
                camera = self.setup_camera(sensor_name, config)
                self.cameras.append(camera)
            elif sensor_type == 'lidar':
                lidar = self.setup_lidar(sensor_name, config)
                self.lidars.append(lidar)

    def setup_camera(self, name: str, config: Dict):
        """Setup a camera sensor"""
        camera_path = f"/World/Sensors/{name}"

        # Create camera prim
        camera_prim = UsdGeom.Camera.Define(self.stage, camera_path)

        # Set camera properties
        camera_prim.GetFocalLengthAttr().Set(config.get('focal_length', 24.0))
        camera_prim.GetHorizontalApertureAttr().Set(config.get('horizontal_aperture', 36.0))
        camera_prim.GetVerticalApertureAttr().Set(config.get('vertical_aperture', 24.0))
        camera_prim.GetClippingRangeAttr().Set(config.get('clipping_range', (0.1, 1000.0)))

        # Set camera transform
        position = config.get('position', (0, 0, 2))
        rotation = config.get('rotation', (0, 0, 0))

        xform = UsdGeom.Xformable(camera_prim)
        xform.AddTranslateOp().Set(Gf.Vec3f(*position))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))

        # Create Isaac Sim camera
        camera = Camera(
            prim_path=camera_path,
            frequency=config.get('frequency', 30),
            resolution=config.get('resolution', (640, 480))
        )

        return {
            'camera': camera,
            'name': name,
            'config': config
        }

    def setup_lidar(self, name: str, config: Dict):
        """Setup a LiDAR sensor"""
        lidar_path = f"/World/Sensors/{name}"

        lidar = LidarRtx(
            prim_path=lidar_path,
            position=config.get('position', (0, 0, 1)),
            orientation=config.get('orientation', (0, 0, 0, 1)),
            config=config.get('lidar_config', "Solid-State-Mixed"),
            min_range=config.get('min_range', 0.1),
            max_range=config.get('max_range', 25.0),
            fov=config.get('fov', 360)
        )

        return {
            'lidar': lidar,
            'name': name,
            'config': config
        }

    def generate_dataset(self, num_samples: int, generation_config: Dict):
        """Generate a complete dataset"""
        start_time = carb.events.acquire_application().get_current_time()

        self.get_logger().info(f"Starting dataset generation: {num_samples} samples")

        for i in range(num_samples):
            # Randomize scene
            self.randomize_scene()

            # Generate sample
            sample_data = self.generate_sample(f"sample_{i:06d}")

            # Save sample
            self.save_sample(sample_data)

            # Update statistics
            self.stats['total_samples'] += 1

            # Log progress
            if (i + 1) % 100 == 0:
                self.get_logger().info(f"Generated {i + 1}/{num_samples} samples")

        # Calculate generation time
        end_time = carb.events.acquire_application().get_current_time()
        self.stats['generation_time'] = end_time - start_time

        self.get_logger().info(f"Dataset generation completed: {num_samples} samples in {self.stats['generation_time']:.2f}s")

        # Save dataset metadata
        self.save_dataset_metadata()

        return self.stats

    def randomize_scene(self):
        """Randomize scene elements for domain randomization"""
        # Randomize lighting
        self.randomize_lighting()

        # Randomize object positions
        self.randomize_object_positions()

        # Randomize materials
        self.randomize_materials()

        # Randomize camera positions (if enabled)
        self.randomize_camera_positions()

    def randomize_lighting(self):
        """Randomize lighting conditions"""
        # This would modify existing lights with random parameters
        pass

    def randomize_object_positions(self):
        """Randomize object positions"""
        for obj in self.objects:
            obj_prim = get_prim_at_path(obj['path'])
            if obj_prim:
                # Get current transform
                xform = UsdGeom.Xformable(obj_prim)

                # Calculate new random position
                pos_range = obj['config'].get('position_range', [(-5, -5, 0), (5, 5, 2)])
                new_pos = [
                    random.uniform(pos_range[0][0], pos_range[1][0]),
                    random.uniform(pos_range[0][1], pos_range[1][1]),
                    random.uniform(pos_range[0][2], pos_range[1][2])
                ]

                # Apply new position
                xform_op = xform.GetOrderedXformOps()[0]  # Assuming first op is translate
                xform_op.Set(Gf.Vec3f(*new_pos))

    def randomize_materials(self):
        """Randomize object materials"""
        for obj in self.objects:
            self.apply_random_material(obj['path'], obj['config'].get('material_config', {}))

    def generate_sample(self, sample_id: str) -> Dict:
        """Generate a single data sample"""
        # Step the simulation to update all sensors
        self.world.step(render=True)

        sample_data = {
            'id': sample_id,
            'timestamp': carb.events.acquire_application().get_current_time(),
            'camera_data': {},
            'lidar_data': {},
            'object_poses': {},
            'scene_config': self.get_scene_config(),
            'metadata': {}
        }

        # Capture camera data
        for cam_info in self.cameras:
            camera = cam_info['camera']
            cam_name = cam_info['name']

            # Get RGB image
            rgb_image = camera.get_rgb()
            if rgb_image is not None:
                sample_data['camera_data'][cam_name] = {
                    'rgb': rgb_image,
                    'depth': camera.get_depth(),
                    'semantic': camera.get_semantic_segmentation(),
                    'instance': camera.get_instance_segmentation(),
                    'camera_params': camera.get_intrinsics()
                }

        # Capture LiDAR data
        for lidar_info in self.lidars:
            lidar = lidar_info['lidar']
            lidar_name = lidar_info['name']

            point_cloud = lidar.get_point_cloud()
            if point_cloud is not None:
                sample_data['lidar_data'][lidar_name] = {
                    'point_cloud': point_cloud,
                    'intensities': lidar.get_intensities()
                }

        # Capture object poses
        for obj in self.objects:
            # Get object pose from simulation
            sample_data['object_poses'][obj['path']] = self.get_object_pose(obj['path'])

        return sample_data

    def save_sample(self, sample_data: Dict):
        """Save a data sample to disk"""
        sample_id = sample_data['id']

        # Save camera data
        for cam_name, cam_data in sample_data['camera_data'].items():
            # Save RGB image
            if 'rgb' in cam_data and cam_data['rgb'] is not None:
                rgb_path = os.path.join(self.data_dirs['images'], f"{sample_id}_{cam_name}.png")
                Image.fromarray(cam_data['rgb']).save(rgb_path)

            # Save depth image
            if 'depth' in cam_data and cam_data['depth'] is not None:
                depth_path = os.path.join(self.data_dirs['depth'], f"{sample_id}_{cam_name}_depth.png")
                depth_normalized = ((cam_data['depth'] - cam_data['depth'].min()) /
                                  (cam_data['depth'].max() - cam_data['depth'].min()) * 255).astype(np.uint8)
                Image.fromarray(depth_normalized).save(depth_path)

            # Save semantic segmentation
            if 'semantic' in cam_data and cam_data['semantic'] is not None:
                semantic_path = os.path.join(self.data_dirs['semantic'], f"{sample_id}_{cam_name}_semantic.png")
                Image.fromarray(cam_data['semantic']).save(semantic_path)

            # Save instance segmentation
            if 'instance' in cam_data and cam_data['instance'] is not None:
                instance_path = os.path.join(self.data_dirs['instances'], f"{sample_id}_{cam_name}_instance.png")
                Image.fromarray(cam_data['instance']).save(instance_path)

            # Save camera parameters
            if 'camera_params' in cam_data:
                params_path = os.path.join(self.data_dirs['camera_params'], f"{sample_id}_{cam_name}_params.json")
                with open(params_path, 'w') as f:
                    json.dump(cam_data['camera_params'], f)

        # Save LiDAR data
        for lidar_name, lidar_data in sample_data['lidar_data'].items():
            if 'point_cloud' in lidar_data:
                # Save point cloud as numpy array
                pc_path = os.path.join(self.data_dirs['lidar'], f"{sample_id}_{lidar_name}_pc.npy")
                np.save(pc_path, lidar_data['point_cloud'])

        # Save metadata
        metadata_path = os.path.join(self.data_dirs['metadata'], f"{sample_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                'id': sample_data['id'],
                'timestamp': sample_data['timestamp'],
                'scene_config': sample_data['scene_config'],
                'object_poses': sample_data['object_poses']
            }, f, indent=2)

    def save_dataset_metadata(self):
        """Save overall dataset metadata"""
        metadata = {
            'dataset_name': self.dataset_name,
            'total_samples': self.stats['total_samples'],
            'generation_time': self.stats['generation_time'],
            'generation_config': {},
            'object_distribution': self.get_object_distribution(),
            'data_types': list(self.stats['data_types'].keys()),
            'date_created': carb.events.acquire_application().get_current_time()
        }

        metadata_path = os.path.join(self.output_dir, 'dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_object_distribution(self) -> Dict:
        """Get distribution of objects in the dataset"""
        distribution = {}
        for obj in self.objects:
            obj_type = obj['type']
            distribution[obj_type] = distribution.get(obj_type, 0) + 1
        return distribution

    def get_object_pose(self, obj_path: str) -> Dict:
        """Get object pose from simulation"""
        # This would interface with Isaac Sim to get actual pose
        return {'position': [0, 0, 0], 'orientation': [0, 0, 0, 1]}

    def get_scene_config(self) -> Dict:
        """Get current scene configuration"""
        return {
            'objects': [obj['config'] for obj in self.objects],
            'cameras': [cam['config'] for cam in self.cameras],
            'lidars': [lidar['config'] for lidar in self.lidars]
        }

    def apply_random_material(self, prim_path: str, material_config: Dict):
        """Apply a random material to a prim"""
        # This would create and apply a random material based on config
        pass

    def randomize_color(self, color_range: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """Randomize color within range"""
        min_color, max_color = color_range
        return tuple(
            random.uniform(min_color[i], max_color[i]) for i in range(3)
        )

    def randomize_direction(self) -> Tuple[float, float, float]:
        """Randomize direction/rotation"""
        return (
            random.uniform(-180, 180),
            random.uniform(-90, 90),
            random.uniform(-180, 180)
        )

    def get_logger(self):
        """Get logger instance"""
        return carb.Logger()

def main():
    """Main function to demonstrate synthetic data generation"""
    # Create data generator
    generator = SyntheticDataGenerator(
        output_dir="humanoid_robot_dataset",
        dataset_name="Humanoid_Perception_Dataset"
    )

    # Define scene configuration
    scene_config = {
        'lighting': {
            'num_directional_lights': 2,
            'dome_color': (0.2, 0.2, 0.2),
            'dome_intensity_range': (500, 3000),
            'light_color_range': [(0.8, 0.8, 0.8), (1.0, 1.0, 1.0)],
            'light_intensity_range': (1000, 5000)
        },
        'objects': [
            {
                'type': 'cube',
                'name': 'obstacle_1',
                'position_range': [(-3, -3, 0), (3, 3, 1)],
                'scale_range': (0.3, 1.0)
            },
            {
                'type': 'sphere',
                'name': 'target_1',
                'position_range': [(-2, -2, 0.5), (2, 2, 1.5)],
                'scale_range': (0.2, 0.5)
            }
        ],
        'sensors': [
            {
                'type': 'camera',
                'name': 'rgb_camera',
                'position': (0, 0, 1.5),
                'resolution': (640, 480),
                'frequency': 30
            },
            {
                'type': 'lidar',
                'name': 'front_lidar',
                'position': (0, 0, 1.0),
                'lidar_config': 'Solid-State-Mixed',
                'min_range': 0.1,
                'max_range': 25.0
            }
        ]
    }

    # Setup scene
    generator.setup_scene(scene_config)

    # Define generation configuration
    generation_config = {
        'num_samples': 1000,
        'domain_randomization': True,
        'annotation_types': ['rgb', 'depth', 'semantic', 'instance'],
        'data_augmentation': True
    }

    # Generate dataset
    stats = generator.generate_dataset(1000, generation_config)

    print(f"Dataset generation completed with stats: {stats}")

if __name__ == '__main__':
    main()
```

## Domain Randomization Techniques

### Advanced Domain Randomization

```python
# python/domain_randomization.py
import numpy as np
import random
from typing import Dict, List, Tuple, Any
import colorsys
from pxr import Gf, UsdShade, Sdf

class DomainRandomization:
    def __init__(self):
        self.randomization_params = {
            'lighting': {
                'intensity_range': (100, 5000),
                'color_temperature_range': (3000, 8000),  # Kelvin
                'position_jitter': 0.5,
                'count_range': (1, 5)
            },
            'materials': {
                'albedo_range': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
                'roughness_range': (0.0, 1.0),
                'metallic_range': (0.0, 1.0),
                'normal_map_strength_range': (0.0, 1.0)
            },
            'objects': {
                'scale_jitter': 0.2,
                'position_jitter': 0.3,
                'rotation_jitter': 15.0,  # degrees
                'count_range': (1, 10)
            },
            'camera': {
                'position_jitter': 0.1,
                'rotation_jitter': 5.0,
                'focal_length_range': (18.0, 50.0)
            },
            'environment': {
                'floor_texture_scale_range': (0.5, 2.0),
                'background_complexity': (0.0, 1.0)
            }
        }

    def randomize_lighting(self, current_config: Dict) -> Dict:
        """Randomize lighting parameters"""
        randomized = current_config.copy()

        # Randomize number of lights
        num_lights = random.randint(
            self.randomization_params['lighting']['count_range'][0],
            self.randomization_params['lighting']['count_range'][1]
        )
        randomized['num_lights'] = num_lights

        # Randomize each light
        lights = []
        for i in range(num_lights):
            light = {
                'intensity': random.uniform(
                    self.randomization_params['lighting']['intensity_range'][0],
                    self.randomization_params['lighting']['intensity_range'][1]
                ),
                'color': self.randomize_color_by_temperature(
                    random.uniform(
                        self.randomization_params['lighting']['color_temperature_range'][0],
                        self.randomization_params['lighting']['color_temperature_range'][1]
                    )
                ),
                'position': [
                    random.gauss(0, self.randomization_params['lighting']['position_jitter']),
                    random.gauss(0, self.randomization_params['lighting']['position_jitter']),
                    random.uniform(2, 10)  # Height above ground
                ],
                'type': random.choice(['directional', 'point', 'dome'])
            }
            lights.append(light)

        randomized['lights'] = lights
        return randomized

    def randomize_color_by_temperature(self, kelvin: float) -> Tuple[float, float, float]:
        """
        Convert color temperature in Kelvin to RGB
        Based on approximation algorithm
        """
        temp = kelvin / 100

        if temp <= 66:
            red = 255
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)

        if temp <= 66:
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)

        if temp >= 66:
            blue = 255
        elif temp <= 19:
            blue = 0
        else:
            blue = temp - 10
            blue = 138.5177312231 * np.log(blue) - 305.0447927307

        # Normalize to 0-1 range
        return (
            max(0, min(255, red)) / 255.0,
            max(0, min(255, green)) / 255.0,
            max(0, min(255, blue)) / 255.0
        )

    def randomize_materials(self, current_config: Dict) -> Dict:
        """Randomize material properties"""
        randomized = current_config.copy()

        # Randomize surface properties
        material = {
            'albedo': self.randomize_color_range(
                self.randomization_params['materials']['albedo_range']
            ),
            'roughness': random.uniform(
                self.randomization_params['materials']['roughness_range'][0],
                self.randomization_params['materials']['roughness_range'][1]
            ),
            'metallic': random.uniform(
                self.randomization_params['materials']['metallic_range'][0],
                self.randomization_params['materials']['metallic_range'][1]
            ),
            'normal_map_strength': random.uniform(
                self.randomization_params['materials']['normal_map_strength_range'][0],
                self.randomization_params['materials']['normal_map_strength_range'][1]
            ),
            'texture_enabled': random.choice([True, False]),
            'texture_scale': random.uniform(0.5, 2.0)
        }

        randomized['material'] = material
        return randomized

    def randomize_color_range(self, color_range: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """Randomize color within given range"""
        min_color, max_color = color_range
        return tuple(
            random.uniform(min_color[i], max_color[i]) for i in range(3)
        )

    def randomize_objects(self, current_config: Dict) -> Dict:
        """Randomize object placement and properties"""
        randomized = current_config.copy()

        # Randomize number of objects
        num_objects = random.randint(
            self.randomization_params['objects']['count_range'][0],
            self.randomization_params['objects']['count_range'][1]
        )

        objects = []
        for i in range(num_objects):
            obj = {
                'type': random.choice(['cube', 'sphere', 'cylinder', 'capsule']),
                'scale': [
                    max(0.1, random.gauss(1.0, self.randomization_params['objects']['scale_jitter'])),
                    max(0.1, random.gauss(1.0, self.randomization_params['objects']['scale_jitter'])),
                    max(0.1, random.gauss(1.0, self.randomization_params['objects']['scale_jitter']))
                ],
                'position': [
                    random.gauss(0, self.randomization_params['objects']['position_jitter']),
                    random.gauss(0, self.randomization_params['objects']['position_jitter']),
                    random.uniform(0.1, 2.0)  # Height above ground
                ],
                'rotation': [
                    random.uniform(-self.randomization_params['objects']['rotation_jitter'],
                                 self.randomization_params['objects']['rotation_jitter']),
                    random.uniform(-self.randomization_params['objects']['rotation_jitter'],
                                 self.randomization_params['objects']['rotation_jitter']),
                    random.uniform(-self.randomization_params['objects']['rotation_jitter'],
                                 self.randomization_params['objects']['rotation_jitter'])
                ],
                'material_config': self.randomize_materials({}).get('material', {})
            }
            objects.append(obj)

        randomized['objects'] = objects
        return randomized

    def randomize_camera(self, current_config: Dict) -> Dict:
        """Randomize camera parameters"""
        randomized = current_config.copy()

        camera = {
            'position': [
                random.gauss(0, self.randomization_params['camera']['position_jitter']),
                random.gauss(0, self.randomization_params['camera']['position_jitter']),
                random.uniform(1.0, 3.0)
            ],
            'rotation': [
                random.uniform(-self.randomization_params['camera']['rotation_jitter'],
                             self.randomization_params['camera']['rotation_jitter']),
                random.uniform(-self.randomization_params['camera']['rotation_jitter'],
                             self.randomization_params['camera']['rotation_jitter']),
                random.uniform(-self.randomization_params['camera']['rotation_jitter'],
                             self.randomization_params['camera']['rotation_jitter'])
            ],
            'focal_length': random.uniform(
                self.randomization_params['camera']['focal_length_range'][0],
                self.randomization_params['camera']['focal_length_range'][1]
            ),
            'resolution': random.choice([(640, 480), (1280, 720), (1920, 1080)]),
            'sensor_noise': random.uniform(0.0, 0.1)
        }

        randomized['camera'] = camera
        return randomized

    def randomize_environment(self, current_config: Dict) -> Dict:
        """Randomize environment properties"""
        randomized = current_config.copy()

        env = {
            'floor_texture_scale': random.uniform(
                self.randomization_params['environment']['floor_texture_scale_range'][0],
                self.randomization_params['environment']['floor_texture_scale_range'][1]
            ),
            'background_complexity': random.uniform(
                self.randomization_params['environment']['background_complexity'][0],
                self.randomization_params['environment']['background_complexity'][1]
            ),
            'fog_enabled': random.choice([True, False]),
            'fog_density': random.uniform(0.0, 0.1) if random.choice([True, False]) else 0.0,
            'weather_condition': random.choice(['clear', 'overcast', 'foggy', 'rainy_simulation'])
        }

        randomized['environment'] = env
        return randomized

    def apply_randomization(self, base_config: Dict) -> Dict:
        """Apply all randomization techniques to base configuration"""
        config = base_config.copy()

        # Apply each randomization in sequence
        config = self.randomize_lighting(config)
        config = self.randomize_materials(config)
        config = self.randomize_objects(config)
        config = self.randomize_camera(config)
        config = self.randomize_environment(config)

        return config

class AdvancedDomainRandomizer(DomainRandomization):
    """Advanced domain randomization with physics-aware randomization"""

    def __init__(self):
        super().__init__()
        self.physics_randomization_params = {
            'friction': (0.1, 1.0),
            'restitution': (0.0, 0.5),
            'mass_multiplier': (0.5, 2.0),
            'damping': (0.0, 0.1)
        }

    def randomize_physics_properties(self, current_config: Dict) -> Dict:
        """Randomize physics properties for realistic simulation"""
        randomized = current_config.copy()

        physics = {
            'friction': random.uniform(
                self.physics_randomization_params['friction'][0],
                self.physics_randomization_params['friction'][1]
            ),
            'restitution': random.uniform(
                self.physics_randomization_params['restitution'][0],
                self.physics_randomization_params['restitution'][1]
            ),
            'mass_multiplier': random.uniform(
                self.physics_randomization_params['mass_multiplier'][0],
                self.physics_randomization_params['mass_multiplier'][1]
            ),
            'linear_damping': random.uniform(
                self.physics_randomization_params['damping'][0],
                self.physics_randomization_params['damping'][1]
            ),
            'angular_damping': random.uniform(
                self.physics_randomization_params['damping'][0],
                self.physics_randomization_params['damping'][1]
            )
        }

        randomized['physics'] = physics
        return randomized

    def randomize_sensor_noise(self, current_config: Dict) -> Dict:
        """Add realistic sensor noise patterns"""
        randomized = current_config.copy()

        sensor_noise = {
            'camera_noise': {
                'gaussian_noise_std': random.uniform(0.0, 0.05),
                'shot_noise_factor': random.uniform(0.0, 0.1),
                'thermal_noise_std': random.uniform(0.0, 0.02),
                'motion_blur': random.choice([True, False]),
                'chromatic_aberration': random.uniform(0.0, 0.01)
            },
            'lidar_noise': {
                'range_noise_std': random.uniform(0.001, 0.01),
                'angular_noise_std': random.uniform(0.001, 0.01),
                'intensity_noise_std': random.uniform(0.01, 0.1)
            },
            'imu_noise': {
                'accelerometer_noise_density': random.uniform(1e-4, 1e-3),
                'gyroscope_noise_density': random.uniform(1e-5, 1e-4),
                'accelerometer_random_walk': random.uniform(1e-5, 1e-4),
                'gyroscope_random_walk': random.uniform(1e-6, 1e-5)
            }
        }

        randomized['sensor_noise'] = sensor_noise
        return randomized

def demonstrate_domain_randomization():
    """Demonstrate domain randomization capabilities"""
    print("Demonstrating Domain Randomization Techniques")

    # Initialize randomizer
    randomizer = AdvancedDomainRandomizer()

    # Base configuration
    base_config = {
        'scene_name': 'randomized_scene',
        'lighting': {},
        'materials': {},
        'objects': [],
        'camera': {},
        'environment': {},
        'physics': {},
        'sensor_noise': {}
    }

    # Apply randomization
    randomized_config = randomizer.apply_randomization(base_config)
    randomized_config = randomizer.randomize_physics_properties(randomized_config)
    randomized_config = randomizer.randomize_sensor_noise(randomized_config)

    print(f"Randomized scene configuration created with {len(randomized_config['objects'])} objects")
    print(f"Lighting: {len(randomized_config['lighting'].get('lights', []))} lights")
    print(f"Environment: {randomized_config['environment']['weather_condition']} weather")

    return randomized_config
```

## Multi-Modal Sensor Data Generation

### Sensor Fusion Data Generation

```python
# python/multi_modal_sensors.py
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor import LidarRtx
from omni.isaac.core.utils.prims import get_prim_at_path
import json
import os

@dataclass
class SensorData:
    """Data structure for multi-modal sensor data"""
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    semantic: Optional[np.ndarray] = None
    instance: Optional[np.ndarray] = None
    point_cloud: Optional[np.ndarray] = None
    lidar_ranges: Optional[np.ndarray] = None
    lidar_intensities: Optional[np.ndarray] = None
    imu_data: Optional[Dict] = None
    gps_data: Optional[Dict] = None
    timestamp: float = 0.0

class MultiModalSensorManager:
    """Manager for multi-modal sensor data generation"""

    def __init__(self, world):
        self.world = world
        self.cameras = []
        self.lidars = []
        self.imus = []
        self.gps_sensors = []
        self.synchronization_enabled = True

    def add_camera(self, camera_config: Dict) -> Camera:
        """Add a camera sensor"""
        camera = Camera(
            prim_path=camera_config['prim_path'],
            frequency=camera_config.get('frequency', 30),
            resolution=camera_config.get('resolution', (640, 480))
        )

        # Set camera intrinsics
        if 'intrinsics' in camera_config:
            camera.set_focal_length(camera_config['intrinsics'].get('focal_length', 24.0))
            camera.set_horizontal_aperture(camera_config['intrinsics'].get('horizontal_aperture', 36.0))
            camera.set_vertical_aperture(camera_config['intrinsics'].get('vertical_aperture', 24.0))

        self.cameras.append({
            'camera': camera,
            'config': camera_config,
            'name': camera_config.get('name', f'camera_{len(self.cameras)}')
        })

        return camera

    def add_lidar(self, lidar_config: Dict) -> LidarRtx:
        """Add a LiDAR sensor"""
        lidar = LidarRtx(
            prim_path=lidar_config['prim_path'],
            position=lidar_config.get('position', (0, 0, 1)),
            orientation=lidar_config.get('orientation', (0, 0, 0, 1)),
            config=lidar_config.get('lidar_config', "Solid-State-Mixed"),
            min_range=lidar_config.get('min_range', 0.1),
            max_range=lidar_config.get('max_range', 25.0),
            fov=lidar_config.get('fov', 360)
        )

        self.lidars.append({
            'lidar': lidar,
            'config': lidar_config,
            'name': lidar_config.get('name', f'lidar_{len(self.lidars)}')
        })

        return lidar

    def capture_multi_modal_data(self) -> Dict[str, SensorData]:
        """Capture synchronized multi-modal sensor data"""
        multi_modal_data = {}

        # Capture camera data
        for cam_info in self.cameras:
            camera = cam_info['camera']
            cam_name = cam_info['name']

            sensor_data = SensorData(timestamp=self.world.current_time)

            # Get all camera modalities
            sensor_data.rgb = camera.get_rgb()
            sensor_data.depth = camera.get_depth()
            sensor_data.semantic = camera.get_semantic_segmentation()
            sensor_data.instance = camera.get_instance_segmentation()

            multi_modal_data[cam_name] = sensor_data

        # Capture LiDAR data
        for lidar_info in self.lidars:
            lidar = lidar_info['lidar']
            lidar_name = lidar_info['name']

            if lidar_name not in multi_modal_data:
                multi_modal_data[lidar_name] = SensorData(timestamp=self.world.current_time)

            # Get LiDAR modalities
            multi_modal_data[lidar_name].point_cloud = lidar.get_point_cloud()
            multi_modal_data[lidar_name].lidar_ranges = lidar.get_ranges()
            multi_modal_data[lidar_name].lidar_intensities = lidar.get_intensities()

        # Synchronize timestamps if enabled
        if self.synchronization_enabled:
            common_timestamp = self.world.current_time
            for data in multi_modal_data.values():
                data.timestamp = common_timestamp

        return multi_modal_data

    def generate_calibration_data(self) -> Dict:
        """Generate sensor calibration data"""
        calibration_data = {
            'cameras': {},
            'lidars': {},
            'extrinsics': {}
        }

        # Camera calibration
        for cam_info in self.cameras:
            cam_name = cam_info['name']
            camera = cam_info['camera']

            # Get camera intrinsics
            intrinsics = camera.get_intrinsics()
            calibration_data['cameras'][cam_name] = {
                'intrinsics': intrinsics,
                'resolution': camera.resolution,
                'distortion': camera.get_distortion_parameters()
            }

        # LiDAR calibration
        for lidar_info in self.lidars:
            lidar_name = lidar_info['name']
            lidar = lidar_info['lidar']

            calibration_data['lidars'][lidar_name] = {
                'fov': lidar.get_fov(),
                'min_range': lidar.get_min_range(),
                'max_range': lidar.get_max_range(),
                'rotation_count': lidar.get_rotation_count()
            }

        # Calculate extrinsics (relative poses between sensors)
        for i, cam_info in enumerate(self.cameras):
            for j, lidar_info in enumerate(self.lidars):
                cam_name = cam_info['name']
                lidar_name = lidar_info['name']

                # Calculate transform between sensors
                # This would involve getting actual poses from the simulation
                calibration_data['extrinsics'][f"{cam_name}_to_{lidar_name}"] = {
                    'translation': [0.1, 0.0, 0.05],  # Example offset
                    'rotation': [0, 0, 0, 1]  # Example quaternion
                }

        return calibration_data

class DataFusionProcessor:
    """Process and fuse multi-modal sensor data"""

    def __init__(self):
        self.fusion_algorithms = {
            'camera_lidar': self.fuse_camera_lidar,
            'multi_camera': self.fuse_multi_camera,
            'sensor_array': self.fuse_sensor_array
        }

    def fuse_camera_lidar(self, camera_data: SensorData, lidar_data: SensorData) -> Dict:
        """Fuse camera and LiDAR data"""
        fused_data = {
            'rgb_with_pointcloud_overlay': None,
            'projected_pointcloud': None,
            'fused_features': None,
            'confidence_map': None
        }

        if camera_data.rgb is not None and lidar_data.point_cloud is not None:
            # Project 3D points to 2D image
            projected_points = self.project_pointcloud_to_image(
                lidar_data.point_cloud,
                camera_data.rgb.shape
            )

            # Create RGB with point cloud overlay
            overlay_image = self.create_pointcloud_overlay(
                camera_data.rgb,
                projected_points
            )

            fused_data['rgb_with_pointcloud_overlay'] = overlay_image
            fused_data['projected_pointcloud'] = projected_points

        return fused_data

    def project_pointcloud_to_image(self, pointcloud: np.ndarray, image_shape: Tuple) -> np.ndarray:
        """Project 3D point cloud to 2D image coordinates"""
        # This would use camera intrinsics to project 3D points to 2D
        # Simplified projection for demonstration
        height, width = image_shape[:2]

        # Assume simple pinhole camera model for demonstration
        # In practice, use actual camera intrinsics
        fx, fy = width / 2, height / 2
        cx, cy = width / 2, height / 2

        projected = []
        for point in pointcloud:
            x, y, z = point[:3]
            if z > 0:  # Only points in front of camera
                u = int(fx * x / z + cx)
                v = int(fy * y / z + cy)

                if 0 <= u < width and 0 <= v < height:
                    projected.append([u, v, z])  # u, v, depth

        return np.array(projected)

    def create_pointcloud_overlay(self, rgb_image: np.ndarray, projected_points: np.ndarray) -> np.ndarray:
        """Create RGB image with point cloud overlay"""
        overlay = rgb_image.copy()

        for point in projected_points:
            u, v, depth = int(point[0]), int(point[1]), point[2]
            if 0 <= u < overlay.shape[1] and 0 <= v < overlay.shape[0]:
                # Color code based on depth
                color_intensity = min(255, int(depth * 50))  # Scale depth to color
                overlay[v, u] = [color_intensity, 255 - color_intensity, 0]  # Red-blue based on depth

        return overlay

    def fuse_multi_camera(self, camera_data_list: List[SensorData]) -> Dict:
        """Fuse data from multiple cameras"""
        fused_data = {
            'panoramic_image': None,
            'stereo_depth': None,
            'multi_view_features': None
        }

        if len(camera_data_list) >= 2:
            # Create panoramic image from multiple views
            panoramic = self.create_panoramic_image([data.rgb for data in camera_data_list if data.rgb is not None])
            fused_data['panoramic_image'] = panoramic

        return fused_data

    def create_panoramic_image(self, images: List[np.ndarray]) -> np.ndarray:
        """Create panoramic image from multiple camera views"""
        if not images:
            return None

        # Simplified panoramic stitching
        # In practice, use proper image stitching algorithms
        heights = [img.shape[0] for img in images]
        max_height = max(heights) if heights else 0

        # Horizontally concatenate images (simplified)
        if len(images) == 1:
            return images[0]
        else:
            # Resize all images to same height and concatenate
            resized_images = []
            for img in images:
                if img.shape[0] != max_height:
                    scale_factor = max_height / img.shape[0]
                    new_width = int(img.shape[1] * scale_factor)
                    resized_img = cv2.resize(img, (new_width, max_height))
                    resized_images.append(resized_img)
                else:
                    resized_images.append(img)

            return np.concatenate(resized_images, axis=1)

def generate_multi_modal_dataset(generator, num_samples: int, output_dir: str):
    """Generate multi-modal sensor dataset"""
    # Initialize sensor manager
    sensor_manager = MultiModalSensorManager(generator.world)

    # Setup sensors based on generator configuration
    for cam_config in generator.cameras:
        sensor_manager.add_camera(cam_config['config'])

    for lidar_config in generator.lidars:
        sensor_manager.add_lidar(lidar_config['config'])

    # Initialize fusion processor
    fusion_processor = DataFusionProcessor()

    # Generate samples
    for i in range(num_samples):
        # Randomize scene
        generator.randomize_scene()

        # Capture multi-modal data
        multi_modal_data = sensor_manager.capture_multi_modal_data()

        # Process fused data
        fused_results = {}
        for sensor_name, data in multi_modal_data.items():
            if sensor_name.startswith('camera_') and len([n for n in multi_modal_data.keys() if n.startswith('lidar_')]) > 0:
                # Find corresponding LiDAR data for fusion
                for lidar_name, lidar_data in multi_modal_data.items():
                    if lidar_name.startswith('lidar_'):
                        fused = fusion_processor.fuse_camera_lidar(data, lidar_data)
                        fused_results[f"{sensor_name}_fused_with_{lidar_name}"] = fused
                        break

        # Save multi-modal sample
        sample_dir = os.path.join(output_dir, f"sample_{i:06d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Save individual modalities
        for sensor_name, data in multi_modal_data.items():
            modality_dir = os.path.join(sample_dir, sensor_name)
            os.makedirs(modality_dir, exist_ok=True)

            if data.rgb is not None:
                cv2.imwrite(os.path.join(modality_dir, "rgb.png"), cv2.cvtColor(data.rgb, cv2.COLOR_RGB2BGR))

            if data.depth is not None:
                np.save(os.path.join(modality_dir, "depth.npy"), data.depth)

            if data.semantic is not None:
                cv2.imwrite(os.path.join(modality_dir, "semantic.png"), data.semantic)

            if data.point_cloud is not None:
                np.save(os.path.join(modality_dir, "pointcloud.npy"), data.point_cloud)

        # Save fused data
        fused_dir = os.path.join(sample_dir, "fused")
        os.makedirs(fused_dir, exist_ok=True)

        for fusion_key, fusion_result in fused_results.items():
            if fusion_result['rgb_with_pointcloud_overlay'] is not None:
                cv2.imwrite(
                    os.path.join(fused_dir, f"{fusion_key}_overlay.png"),
                    cv2.cvtColor(fusion_result['rgb_with_pointcloud_overlay'], cv2.COLOR_RGB2BGR)
                )

        # Save calibration data
        calibration_data = sensor_manager.generate_calibration_data()
        with open(os.path.join(sample_dir, "calibration.json"), 'w') as f:
            json.dump(calibration_data, f, indent=2)

        # Save metadata
        metadata = {
            'sample_id': f"sample_{i:06d}",
            'timestamp': data.timestamp if multi_modal_data else 0.0,
            'sensor_configurations': [cam['config'] for cam in generator.cameras],
            'fusion_configurations': list(fused_results.keys())
        }

        with open(os.path.join(sample_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Progress update
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples} multi-modal samples")

# Example usage
def example_multi_modal_generation():
    """Example of multi-modal data generation"""
    print("Generating multi-modal sensor dataset...")

    # This would be integrated with the main generator
    # For now, we'll just demonstrate the concepts
    pass
```

## Large-Scale Data Generation

### Distributed Data Generation

```python
# python/distributed_generation.py
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import time
import os
from typing import Dict, List, Callable
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

class DistributedDataGenerator:
    """Distributed synthetic data generation system"""

    def __init__(self, num_workers: int = None, output_dir: str = "distributed_dataset"):
        self.num_workers = num_workers or mp.cpu_count()
        self.output_dir = output_dir
        self.manager = Manager()
        self.task_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.stats = self.manager.dict()

        # Initialize statistics
        self.stats['total_generated'] = 0
        self.stats['generation_rate'] = 0.0
        self.stats['active_workers'] = 0
        self.stats['errors'] = 0

        os.makedirs(output_dir, exist_ok=True)

    def generate_task_batches(self, total_samples: int, samples_per_batch: int = 100) -> List[Dict]:
        """Generate task batches for distributed processing"""
        batches = []
        remaining = total_samples

        batch_id = 0
        while remaining > 0:
            batch_size = min(samples_per_batch, remaining)

            batch_config = {
                'batch_id': batch_id,
                'sample_count': batch_size,
                'output_dir': os.path.join(self.output_dir, f"batch_{batch_id:04d}"),
                'randomization_config': self.generate_randomization_config(),
                'sensor_config': self.generate_sensor_config()
            }

            batches.append(batch_config)
            remaining -= batch_size
            batch_id += 1

        return batches

    def generate_randomization_config(self) -> Dict:
        """Generate randomization configuration for a batch"""
        return {
            'domain_randomization': {
                'lighting': random.choice([True, False]),
                'materials': random.choice([True, False]),
                'objects': random.choice([True, False]),
                'textures': random.choice([True, False])
            },
            'variation_intensity': random.uniform(0.3, 1.0),
            'specific_domains': random.sample(
                ['color', 'texture', 'shape', 'lighting', 'weather'],
                k=random.randint(2, 5)
            )
        }

    def generate_sensor_config(self) -> Dict:
        """Generate sensor configuration for a batch"""
        sensors = []

        # Add cameras
        num_cameras = random.randint(1, 3)
        for i in range(num_cameras):
            sensors.append({
                'type': 'camera',
                'resolution': random.choice([(640, 480), (1280, 720), (1920, 1080)]),
                'frequency': random.choice([15, 30, 60]),
                'modalities': random.sample(
                    ['rgb', 'depth', 'semantic', 'instance'],
                    k=random.randint(2, 4)
                )
            })

        # Add LiDARs
        if random.choice([True, False]):
            sensors.append({
                'type': 'lidar',
                'configuration': random.choice(['Solid-State', 'Mechanical', 'Flash']),
                'range': random.uniform(10, 100),
                'fov': random.choice([180, 360])
            })

        return {'sensors': sensors}

    def worker_process(self, worker_id: int, task_queue: Queue, result_queue: Queue, stats: Dict):
        """Worker process for generating data batches"""
        import omni
        from omni.isaac.core import World

        # Initialize Isaac Sim in this process
        try:
            # Create a world instance for this worker
            world = World(stage_units_in_meters=1.0)

            # Add ground plane
            world.scene.add_default_ground_plane()

            stats['active_workers'] += 1

            while True:
                try:
                    # Get task from queue
                    task = task_queue.get(timeout=1.0)

                    if task is None:  # Poison pill to stop worker
                        break

                    # Process the task
                    result = self.process_batch_task(task, world)
                    result_queue.put(result)

                    # Update statistics
                    stats['total_generated'] += task['sample_count']

                except Exception as e:
                    stats['errors'] += 1
                    result_queue.put({'error': str(e), 'task_id': task.get('batch_id', 'unknown')})

        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
        finally:
            stats['active_workers'] -= 1

    def process_batch_task(self, task: Dict, world) -> Dict:
        """Process a single batch generation task"""
        batch_id = task['batch_id']
        sample_count = task['sample_count']
        output_dir = task['output_dir']

        os.makedirs(output_dir, exist_ok=True)

        # Create local generator for this batch
        local_generator = SyntheticDataGenerator(
            output_dir=output_dir,
            dataset_name=f"batch_{batch_id}"
        )

        # Setup scene with batch-specific configuration
        scene_config = self.create_scene_config_for_batch(task)
        local_generator.setup_scene(scene_config)

        # Generate samples for this batch
        generation_config = {
            'num_samples': sample_count,
            'domain_randomization': task['randomization_config'],
            'sensors': task['sensor_config']['sensors']
        }

        # Generate the batch
        batch_stats = local_generator.generate_dataset(sample_count, generation_config)

        return {
            'batch_id': batch_id,
            'output_dir': output_dir,
            'samples_generated': sample_count,
            'stats': batch_stats,
            'success': True
        }

    def create_scene_config_for_batch(self, task: Dict) -> Dict:
        """Create scene configuration for a specific batch"""
        randomization = task['randomization_config']

        scene_config = {
            'lighting': {
                'num_directional_lights': 2 if randomization['domain_randomization']['lighting'] else 1,
                'dome_color': (random.uniform(0.1, 0.3), random.uniform(0.1, 0.3), random.uniform(0.1, 0.3)),
                'dome_intensity_range': (500, 3000) if randomization['domain_randomization']['lighting'] else (1000, 1000)
            },
            'objects': self.generate_object_config_for_batch(randomization),
            'sensors': task['sensor_config']['sensors']
        }

        return scene_config

    def generate_object_config_for_batch(self, randomization: Dict) -> List[Dict]:
        """Generate object configuration for a batch"""
        objects = []

        if randomization['domain_randomization']['objects']:
            num_objects = random.randint(3, 10)
        else:
            num_objects = random.randint(1, 3)

        for i in range(num_objects):
            obj_type = random.choice(['cube', 'sphere', 'cylinder', 'capsule'])
            obj_config = {
                'type': obj_type,
                'name': f"obj_{i}",
                'position_range': [(-5, -5, 0), (5, 5, 2)],
                'scale_range': (0.2, 1.5) if randomization['domain_randomization']['objects'] else (1.0, 1.0)
            }

            if randomization['domain_randomization']['materials']:
                obj_config['material_config'] = {
                    'albedo_range': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
                    'roughness_range': (0.0, 1.0),
                    'metallic_range': (0.0, 1.0)
                }

            objects.append(obj_config)

        return objects

    def run_distributed_generation(self, total_samples: int, samples_per_batch: int = 100):
        """Run distributed data generation"""
        print(f"Starting distributed generation: {total_samples} samples with {self.num_workers} workers")

        # Generate task batches
        batches = self.generate_task_batches(total_samples, samples_per_batch)
        print(f"Generated {len(batches)} batches")

        # Start worker processes
        processes = []
        for i in range(self.num_workers):
            p = Process(
                target=self.worker_process,
                args=(i, self.task_queue, self.result_queue, self.stats)
            )
            p.start()
            processes.append(p)

        # Add tasks to queue
        for batch in batches:
            self.task_queue.put(batch)

        # Add poison pills to stop workers
        for _ in range(self.num_workers):
            self.task_queue.put(None)

        # Collect results and monitor progress
        completed_batches = 0
        start_time = time.time()

        while completed_batches < len(batches):
            try:
                result = self.result_queue.get(timeout=1.0)

                if 'error' in result:
                    print(f"Error in batch {result['task_id']}: {result['error']}")
                else:
                    completed_batches += 1
                    elapsed_time = time.time() - start_time
                    rate = completed_batches / elapsed_time if elapsed_time > 0 else 0

                    print(f"Completed batch {result['batch_id']}: {result['samples_generated']} samples "
                          f"in {elapsed_time:.2f}s (Rate: {rate:.2f} batches/s)")

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Print final statistics
        print(f"\nDistributed generation completed!")
        print(f"Total samples generated: {self.stats['total_generated']}")
        print(f"Errors occurred: {self.stats['errors']}")
        print(f"Active workers at completion: {self.stats['active_workers']}")

        return self.stats

class ScalableDataPipeline:
    """Scalable pipeline for synthetic data generation"""

    def __init__(self):
        self.stages = []
        self.stage_outputs = {}
        self.pipeline_config = {}

    def add_stage(self, name: str, function: Callable, config: Dict = None):
        """Add a processing stage to the pipeline"""
        self.stages.append({
            'name': name,
            'function': function,
            'config': config or {}
        })

    def run_pipeline(self, input_data):
        """Run the complete pipeline"""
        current_data = input_data

        for stage in self.stages:
            print(f"Running pipeline stage: {stage['name']}")
            current_data = stage['function'](current_data, **stage['config'])
            self.stage_outputs[stage['name']] = current_data

        return current_data

def demonstrate_distributed_generation():
    """Demonstrate distributed data generation"""
    print("Demonstrating Distributed Data Generation")

    # Create distributed generator
    dist_gen = DistributedDataGenerator(num_workers=4, output_dir="demo_distributed_dataset")

    # Run small-scale distributed generation
    stats = dist_gen.run_distributed_generation(total_samples=500, samples_per_batch=50)

    print(f"Distributed generation stats: {stats}")

    return dist_gen

# Example usage
if __name__ == "__main__":
    # This would be run in the context of the main generator
    pass
```

## Data Quality and Validation

### Quality Assurance Pipeline

```python
# python/data_quality_assurance.py
import numpy as np
import cv2
from PIL import Image
import json
from typing import Dict, List, Tuple, Any
import os
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

class DataQualityAssessor:
    """Assess quality of synthetic data"""

    def __init__(self):
        self.quality_metrics = {
            'image_quality': ['sharpness', 'contrast', 'brightness', 'noise_level'],
            'annotation_quality': ['completeness', 'accuracy', 'consistency'],
            'dataset_diversity': ['color_diversity', 'texture_diversity', 'spatial_diversity'],
            'realism_metrics': ['domain_gap', 'perceptual_similarity']
        }

    def assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess various image quality metrics"""
        metrics = {}

        # Sharpness (using Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = float(laplacian_var)

        # Contrast (using standard deviation)
        contrast = gray.std()
        metrics['contrast'] = float(contrast)

        # Brightness (mean intensity)
        brightness = gray.mean()
        metrics['brightness'] = float(brightness)

        # Noise level (using wavelet-based estimation)
        noise_level = self.estimate_noise_level(gray)
        metrics['noise_level'] = float(noise_level)

        return metrics

    def estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in image"""
        # Simple noise estimation using wavelet coefficients
        # Take the standard deviation of the finest scale detail coefficients
        # This is a simplified approach - in practice, use more sophisticated methods
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        noise_estimate = np.std(gradient_magnitude) / 100.0  # Normalize
        return noise_estimate

    def assess_annotation_quality(self, annotations: Dict, ground_truth: Dict = None) -> Dict[str, float]:
        """Assess annotation quality"""
        metrics = {}

        # Completeness (ratio of annotated to total possible elements)
        if 'objects' in annotations:
            annotated_count = len(annotations['objects'])
            # This would need context about how many objects should be annotated
            metrics['completeness'] = min(1.0, annotated_count / 10.0)  # Placeholder

        # Accuracy (if ground truth is available)
        if ground_truth is not None:
            accuracy = self.calculate_annotation_accuracy(annotations, ground_truth)
            metrics['accuracy'] = accuracy

        # Consistency (checking for consistent labeling across frames)
        consistency = self.check_annotation_consistency(annotations)
        metrics['consistency'] = consistency

        return metrics

    def calculate_annotation_accuracy(self, annotations: Dict, ground_truth: Dict) -> float:
        """Calculate annotation accuracy against ground truth"""
        # This would implement IoU calculations, classification accuracy, etc.
        # For now, return a placeholder
        return 0.95  # Placeholder accuracy

    def check_annotation_consistency(self, annotations: Dict) -> float:
        """Check consistency of annotations"""
        # Check if annotations follow consistent patterns
        # This could involve checking for consistent object sizes, positions, etc.
        return 0.98  # Placeholder consistency

    def assess_dataset_diversity(self, dataset_samples: List[np.ndarray]) -> Dict[str, float]:
        """Assess diversity of the dataset"""
        metrics = {}

        if not dataset_samples:
            return metrics

        # Color diversity (variance in color space)
        color_diversity = self.calculate_color_diversity(dataset_samples)
        metrics['color_diversity'] = color_diversity

        # Texture diversity (using local binary patterns or similar)
        texture_diversity = self.calculate_texture_diversity(dataset_samples)
        metrics['texture_diversity'] = texture_diversity

        # Spatial diversity (distribution of features in image space)
        spatial_diversity = self.calculate_spatial_diversity(dataset_samples)
        metrics['spatial_diversity'] = spatial_diversity

        return metrics

    def calculate_color_diversity(self, images: List[np.ndarray]) -> float:
        """Calculate color diversity across dataset"""
        all_colors = []
        for img in images:
            # Sample colors from each image
            height, width = img.shape[:2]
            sample_points = 100  # Number of sample points per image
            y_coords = np.random.randint(0, height, sample_points)
            x_coords = np.random.randint(0, width, sample_points)

            sampled_colors = img[y_coords, x_coords]
            all_colors.extend(sampled_colors)

        all_colors = np.array(all_colors)

        # Calculate diversity as variance in color space
        color_variance = np.var(all_colors, axis=0)
        diversity_score = np.mean(color_variance) / 255.0  # Normalize

        return min(1.0, diversity_score)

    def calculate_texture_diversity(self, images: List[np.ndarray]) -> float:
        """Calculate texture diversity using local statistics"""
        # Use local binary patterns or similar texture descriptors
        texture_features = []

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img

            # Calculate local statistics
            local_mean = cv2.blur(gray.astype(np.float32), (10, 10))
            local_std = np.sqrt(cv2.blur((gray.astype(np.float32) - local_mean)**2, (10, 10)))

            # Sample texture features
            feature_vector = [
                np.mean(local_std),
                np.std(local_std),
                np.percentile(local_std, 90),
                np.percentile(local_std, 10)
            ]

            texture_features.append(feature_vector)

        texture_features = np.array(texture_features)

        # Calculate diversity as variance of texture features
        feature_variance = np.var(texture_features, axis=0)
        diversity_score = np.mean(feature_variance)

        return min(1.0, diversity_score / 100.0)  # Normalize

    def calculate_spatial_diversity(self, images: List[np.ndarray]) -> float:
        """Calculate spatial diversity of content distribution"""
        spatial_features = []

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img

            # Calculate spatial distribution of edges
            edges = cv2.Canny(gray, 50, 150)
            height, width = edges.shape

            # Calculate center of mass of edges
            y_coords, x_coords = np.where(edges > 0)
            if len(y_coords) > 0:
                center_y = np.mean(y_coords) / height
                center_x = np.mean(x_coords) / width
                spatial_features.append([center_x, center_y])

        if len(spatial_features) < 2:
            return 0.0

        spatial_features = np.array(spatial_features)

        # Calculate pairwise distances
        distances = pdist(spatial_features)
        diversity_score = np.mean(distances) * 2  # Scale up for better range

        return min(1.0, diversity_score)

    def validate_data_integrity(self, sample_path: str) -> Dict[str, Any]:
        """Validate integrity of a data sample"""
        validation_results = {
            'file_exists': True,
            'file_readable': True,
            'data_consistency': True,
            'annotation_alignment': True,
            'errors': []
        }

        # Check if files exist and are readable
        required_files = [
            'rgb.png',
            'depth.npy',
            'semantic.png',
            'metadata.json'
        ]

        for file_name in required_files:
            file_path = os.path.join(sample_path, file_name)
            if not os.path.exists(file_path):
                validation_results['file_exists'] = False
                validation_results['errors'].append(f"Missing file: {file_name}")
            else:
                try:
                    if file_name.endswith('.png'):
                        img = Image.open(file_path)
                        img.verify()
                    elif file_name.endswith('.npy'):
                        data = np.load(file_path)
                    elif file_name.endswith('.json'):
                        with open(file_path, 'r') as f:
                            json.load(f)
                except Exception as e:
                    validation_results['file_readable'] = False
                    validation_results['errors'].append(f"Cannot read {file_name}: {str(e)}")

        # Check data consistency
        try:
            rgb_path = os.path.join(sample_path, 'rgb.png')
            depth_path = os.path.join(sample_path, 'depth.npy')

            if os.path.exists(rgb_path) and os.path.exists(depth_path):
                rgb_img = np.array(Image.open(rgb_path))
                depth_data = np.load(depth_path)

                # Check if dimensions match
                if rgb_img.shape[:2] != depth_data.shape[:2]:
                    validation_results['data_consistency'] = False
                    validation_results['errors'].append("RGB and depth dimensions don't match")
        except Exception as e:
            validation_results['data_consistency'] = False
            validation_results['errors'].append(f"Data consistency check failed: {str(e)}")

        return validation_results

class DatasetValidator:
    """Comprehensive dataset validator"""

    def __init__(self):
        self.assessor = DataQualityAssessor()
        self.validation_report = {}

    def validate_dataset(self, dataset_path: str, sample_count: int = 100) -> Dict[str, Any]:
        """Validate entire dataset"""
        print(f"Validating dataset at: {dataset_path}")

        # Get all sample directories
        sample_dirs = [d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('sample_')]

        # Limit to sample_count for efficiency
        sample_dirs = sample_dirs[:min(sample_count, len(sample_dirs))]

        validation_results = {
            'total_samples': len(sample_dirs),
            'passed_samples': 0,
            'failed_samples': 0,
            'quality_metrics': {},
            'integrity_report': {},
            'recommendations': []
        }

        all_image_qualities = []
        all_annotation_qualities = []
        all_samples = []

        for i, sample_dir in enumerate(sample_dirs):
            sample_path = os.path.join(dataset_path, sample_dir)

            # Validate sample integrity
            integrity_result = self.assessor.validate_data_integrity(sample_path)

            if integrity_result['file_exists'] and integrity_result['file_readable']:
                # Load and assess sample quality
                rgb_path = os.path.join(sample_path, 'rgb.png')
                if os.path.exists(rgb_path):
                    try:
                        rgb_img = np.array(Image.open(rgb_path))

                        # Assess image quality
                        img_quality = self.assessor.assess_image_quality(rgb_img)
                        all_image_qualities.append(img_quality)

                        # Load annotations if available
                        metadata_path = os.path.join(sample_path, 'metadata.json')
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)

                            annotation_quality = self.assessor.assess_annotation_quality(
                                metadata.get('annotations', {})
                            )
                            all_annotation_qualities.append(annotation_quality)

                        all_samples.append(rgb_img)

                        validation_results['passed_samples'] += 1
                    except Exception as e:
                        validation_results['failed_samples'] += 1
                        integrity_result['errors'].append(f"Quality assessment failed: {str(e)}")
                else:
                    validation_results['failed_samples'] += 1
            else:
                validation_results['failed_samples'] += 1

            # Progress update
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(sample_dirs)} samples")

        # Calculate aggregate metrics
        if all_image_qualities:
            avg_img_quality = {}
            for key in all_image_qualities[0].keys():
                values = [q[key] for q in all_image_qualities]
                avg_img_quality[key] = sum(values) / len(values)

            validation_results['quality_metrics']['image_quality'] = avg_img_quality

        if all_annotation_qualities:
            avg_annotation_quality = {}
            for key in all_annotation_qualities[0].keys():
                values = [q[key] for q in all_annotation_qualities]
                avg_annotation_quality[key] = sum(values) / len(values)

            validation_results['quality_metrics']['annotation_quality'] = avg_annotation_quality

        # Assess dataset diversity if we have enough samples
        if len(all_samples) >= 10:
            diversity_metrics = self.assessor.assess_dataset_diversity(all_samples[:50])  # Limit for efficiency
            validation_results['quality_metrics']['diversity'] = diversity_metrics

        # Generate recommendations
        self.generate_recommendations(validation_results)

        return validation_results

    def generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        # Check pass rate
        total = validation_results['total_samples']
        passed = validation_results['passed_samples']
        pass_rate = passed / total if total > 0 else 0

        if pass_rate < 0.95:
            recommendations.append(f"Low pass rate ({pass_rate:.2%}), investigate data generation pipeline")

        # Check image quality
        img_quality = validation_results['quality_metrics'].get('image_quality', {})
        if img_quality.get('sharpness', 0) < 100:  # Threshold is arbitrary
            recommendations.append("Low image sharpness detected, consider improving rendering quality")

        if img_quality.get('noise_level', 1) > 0.1:  # Threshold is arbitrary
            recommendations.append("High noise levels detected, consider denoising or improving lighting")

        # Check diversity
        diversity = validation_results['quality_metrics'].get('diversity', {})
        if diversity.get('color_diversity', 0) < 0.3:  # Threshold is arbitrary
            recommendations.append("Low color diversity, consider enhancing domain randomization")

        if diversity.get('spatial_diversity', 0) < 0.3:  # Threshold is arbitrary
            recommendations.append("Low spatial diversity, consider varying object placements more")

        validation_results['recommendations'] = recommendations
        return recommendations

def validate_synthetic_dataset(dataset_path: str):
    """Validate a synthetic dataset"""
    validator = DatasetValidator()
    results = validator.validate_dataset(dataset_path, sample_count=200)  # Validate first 200 samples

    print(f"\nDataset Validation Results:")
    print(f"Total samples: {results['total_samples']}")
    print(f"Passed: {results['passed_samples']}")
    print(f"Failed: {results['failed_samples']}")
    print(f"Pass rate: {results['passed_samples']/results['total_samples']:.2%}")

    print(f"\nQuality Metrics:")
    for category, metrics in results['quality_metrics'].items():
        print(f"  {category}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")

    if results['recommendations']:
        print(f"\nRecommendations:")
        for rec in results['recommendations']:
            print(f"  - {rec}")

    return results
```

## Best Practices for Synthetic Data Generation

### Quality Guidelines

1. **Realism vs. Diversity**: Balance photorealism with domain diversity
2. **Annotation Quality**: Ensure perfect ground truth annotations
3. **Validation**: Implement comprehensive validation pipelines
4. **Scalability**: Design for large-scale generation
5. **Modularity**: Keep generation components modular and reusable

### Performance Considerations

- **GPU Utilization**: Maximize GPU usage for rendering
- **Memory Management**: Efficient memory usage for large datasets
- **I/O Optimization**: Optimize data writing and reading
- **Parallel Processing**: Use multi-processing effectively
- **Storage Efficiency**: Balance quality with storage requirements

## Hands-On Exercise

### Exercise: Building a Synthetic Data Pipeline

1. **Setup Isaac Sim Environment**
   - Install Isaac Sim and required packages
   - Configure for synthetic data generation

2. **Create Basic Generator**
   - Implement simple scene randomization
   - Add basic sensor simulation

3. **Implement Domain Randomization**
   - Add lighting variations
   - Implement material randomization
   - Add texture variations

4. **Scale Up Generation**
   - Implement multi-modal sensors
   - Add distributed generation
   - Optimize for performance

5. **Validate Generated Data**
   - Implement quality assessment
   - Run validation pipelines
   - Analyze dataset statistics

## Summary

Synthetic data generation in Isaac Sim provides powerful capabilities for creating large, diverse, and perfectly annotated datasets for robotics AI development. By leveraging domain randomization, multi-modal sensors, and distributed processing, you can create datasets that enable robust model training. The key is balancing photorealism with diversity while maintaining efficient generation pipelines. Proper validation and quality assurance ensure that generated data meets the requirements for downstream AI applications.

## Learning Path Adjustment

Based on your experience level, you may want to focus on:

- **Beginner**: Focus on basic Isaac Sim setup and simple data generation
- **Intermediate**: Dive deeper into domain randomization and multi-modal sensors
- **Advanced**: Explore distributed generation and advanced quality assurance techniques