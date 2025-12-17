---
sidebar_position: 4
---

# Sensor Simulation: LiDAR, Depth Camera, IMU

## Chapter Objectives

- Implement realistic LiDAR sensor simulation
- Create depth camera simulation with realistic noise models
- Simulate IMU sensors with drift and noise characteristics
- Integrate sensor data with ROS/ROS 2 for robotics applications

## Introduction to Sensor Simulation

Sensor simulation is crucial for humanoid robotics development as it:

- Provides realistic sensor data for algorithm development
- Enables testing without physical sensors
- Allows simulation of sensor failures and edge cases
- Supports AI training with synthetic data

### Sensor Categories for Humanoid Robots

Humanoid robots typically use several sensor types:

1. **LiDAR**: 360-degree distance measurements for mapping and navigation
2. **Depth Cameras**: 3D point clouds for object recognition and manipulation
3. **IMU**: Inertial measurements for orientation and motion
4. **Other Sensors**: GPS, force/torque, tactile, etc.

## LiDAR Simulation

### LiDAR Physics and Characteristics

LiDAR (Light Detection and Ranging) sensors emit laser pulses and measure the time of flight to determine distances. Key characteristics include:

- **Range**: Maximum distance measurement (typically 5-100m)
- **Resolution**: Angular resolution (typically 0.1°-1°)
- **Field of View**: Horizontal and vertical coverage
- **Update Rate**: How frequently measurements are taken
- **Accuracy**: Measurement precision and noise characteristics

### LiDAR Simulation in Gazebo

```xml
<!-- URDF snippet for LiDAR sensor -->
<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>lidar</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Simulation in Unity

```csharp
// Scripts/LiDARSimulator.cs
using UnityEngine;
using System.Collections.Generic;

public class LiDARSimulator : MonoBehaviour
{
    [Header("LiDAR Configuration")]
    public int beamCount = 360;
    public float minAngle = -Mathf.PI;
    public float maxAngle = Mathf.PI;
    public float maxRange = 30f;
    public float updateRate = 10f; // Hz
    public LayerMask detectionLayers = -1;

    [Header("Noise Parameters")]
    public float noiseStdDev = 0.01f;
    public float biasError = 0f;

    [Header("Visualization")]
    public bool visualizeBeams = true;
    public LineRenderer lineRenderer;

    private float[] ranges;
    private float updateInterval;
    private float lastUpdate;

    void Start()
    {
        ranges = new float[beamCount];
        updateInterval = 1f / updateRate;
        lastUpdate = -updateInterval; // Allow immediate first update

        if (lineRenderer == null)
        {
            lineRenderer = GetComponent<LineRenderer>();
            if (lineRenderer == null)
            {
                lineRenderer = gameObject.AddComponent<LineRenderer>();
                lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
                lineRenderer.startWidth = 0.01f;
                lineRenderer.endWidth = 0.01f;
                lineRenderer.positionCount = beamCount;
            }
        }
    }

    void Update()
    {
        if (Time.time - lastUpdate >= updateInterval)
        {
            SimulateLiDAR();
            lastUpdate = Time.time;
        }
    }

    void SimulateLiDAR()
    {
        for (int i = 0; i < beamCount; i++)
        {
            float angle = Mathf.Lerp(minAngle, maxAngle, (float)i / (beamCount - 1));
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));

            // Perform raycast
            RaycastHit hit;
            if (Physics.Raycast(transform.position, transform.TransformDirection(direction), out hit, maxRange, detectionLayers))
            {
                float distance = hit.distance;

                // Add noise
                float noisyDistance = AddNoise(distance);

                ranges[i] = noisyDistance;
            }
            else
            {
                ranges[i] = float.MaxValue; // No hit
            }
        }

        if (visualizeBeams)
        {
            UpdateVisualization();
        }

        // Publish simulated data (in a real implementation, this would publish to ROS)
        PublishLiDARData();
    }

    float AddNoise(float trueValue)
    {
        // Add Gaussian noise
        float noise = RandomGaussian() * noiseStdDev;
        return trueValue + noise + biasError;
    }

    float RandomGaussian()
    {
        // Box-Muller transform for Gaussian random numbers
        float u1 = Random.value;
        float u2 = Random.value;
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
    }

    void UpdateVisualization()
    {
        Vector3[] positions = new Vector3[beamCount];
        for (int i = 0; i < beamCount; i++)
        {
            float angle = Mathf.Lerp(minAngle, maxAngle, (float)i / (beamCount - 1));
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));

            if (ranges[i] < maxRange)
            {
                positions[i] = transform.position + transform.TransformDirection(direction) * ranges[i];
            }
            else
            {
                positions[i] = transform.position + transform.TransformDirection(direction) * maxRange * 0.9f;
            }
        }

        lineRenderer.SetPositions(positions);
    }

    void PublishLiDARData()
    {
        // In a real implementation, this would publish to ROS/ROS 2
        // For now, we'll just log the data
        Debug.Log($"LiDAR: {beamCount} beams, first range: {ranges[0]:F3}m");
    }

    public float[] GetRanges()
    {
        return ranges;
    }

    public float GetRange(int beamIndex)
    {
        if (beamIndex >= 0 && beamIndex < ranges.Length)
        {
            return ranges[beamIndex];
        }
        return float.MaxValue;
    }
}
```

## Depth Camera Simulation

### Depth Camera Characteristics

Depth cameras provide 3D point cloud data by measuring distance to objects in the scene. Key characteristics include:

- **Resolution**: Image dimensions (e.g., 640x480, 1280x720)
- **Field of View**: Horizontal and vertical angles
- **Depth Range**: Minimum and maximum measurable distances
- **Accuracy**: Distance measurement precision
- **Frame Rate**: How often images are captured

### Depth Camera Simulation in Unity

```csharp
// Scripts/DepthCameraSimulator.cs
using UnityEngine;
using System.Collections;

public class DepthCameraSimulator : MonoBehaviour
{
    [Header("Camera Configuration")]
    public int width = 640;
    public int height = 480;
    public float fieldOfView = 60f;
    public float nearClip = 0.1f;
    public float farClip = 10f;

    [Header("Noise Parameters")]
    public float noiseStdDev = 0.02f;
    public float biasError = 0f;
    public float depthScale = 1f;

    [Header("Output Settings")]
    public bool outputPointCloud = true;
    public bool outputDepthImage = true;

    private Camera depthCamera;
    private RenderTexture depthTexture;
    private float[,] depthData;
    private GameObject pointCloudObject;

    void Start()
    {
        SetupCamera();
        CreateDepthTexture();
        depthData = new float[width, height];
    }

    void SetupCamera()
    {
        depthCamera = GetComponent<Camera>();
        if (depthCamera == null)
        {
            depthCamera = gameObject.AddComponent<Camera>();
        }

        depthCamera.fieldOfView = fieldOfView;
        depthCamera.nearClipPlane = nearClip;
        depthCamera.farClipPlane = farClip;
        depthCamera.depth = -1; // Render after other cameras
        depthCamera.clearFlags = CameraClearFlags.SolidColor;
        depthCamera.backgroundColor = Color.white;
        depthCamera.enabled = false; // We'll render manually
    }

    void CreateDepthTexture()
    {
        depthTexture = new RenderTexture(width, height, 24, RenderTextureFormat.Depth);
        depthTexture.Create();
        depthCamera.targetTexture = depthTexture;
    }

    void Update()
    {
        SimulateDepthCamera();
    }

    void SimulateDepthCamera()
    {
        // Render the scene from this camera's perspective
        depthCamera.Render();

        // Read depth data from render texture
        ReadDepthData();

        // Add noise to simulate real sensor characteristics
        AddNoiseToDepthData();

        if (outputPointCloud)
        {
            GeneratePointCloud();
        }

        if (outputDepthImage)
        {
            OutputDepthImage();
        }
    }

    void ReadDepthData()
    {
        // Create a temporary render texture to read the depth data
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = depthTexture;

        Texture2D depthTex = new Texture2D(width, height, TextureFormat.RFloat, false);
        depthTex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        depthTex.Apply();

        // Convert texture data to depth values
        Color[] pixels = depthTex.GetPixels();
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = y * width + x;
                float rawDepth = pixels[index].r;

                // Convert raw depth to actual distance
                // This is a simplified conversion - real implementation would be more complex
                float actualDepth = ConvertRawDepthToDistance(rawDepth);
                depthData[x, y] = actualDepth;
            }
        }

        // Clean up
        RenderTexture.active = currentRT;
        DestroyImmediate(depthTex);
    }

    float ConvertRawDepthToDistance(float rawDepth)
    {
        // Convert raw depth buffer value to actual distance
        // This is a simplified conversion
        float zNear = depthCamera.nearClipPlane;
        float zFar = depthCamera.farClipPlane;

        // Convert from [0,1] to actual distance
        float linearDepth = (2.0f * zNear * zFar) / (zFar + zNear - rawDepth * (zFar - zNear));
        return linearDepth;
    }

    void AddNoiseToDepthData()
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (depthData[x, y] < farClip) // Only add noise to valid measurements
                {
                    float noise = RandomGaussian() * noiseStdDev;
                    depthData[x, y] += noise + biasError;

                    // Ensure depth is within valid range
                    depthData[x, y] = Mathf.Clamp(depthData[x, y], nearClip, farClip);
                }
            }
        }
    }

    float RandomGaussian()
    {
        // Box-Muller transform for Gaussian random numbers
        float u1 = Random.value;
        float u2 = Random.value;
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
    }

    void GeneratePointCloud()
    {
        // Generate point cloud from depth data
        Vector3[] points = new Vector3[width * height];
        int pointCount = 0;

        float fovX = fieldOfView * Mathf.Deg2Rad;
        float fovY = (fieldOfView * height / width) * Mathf.Deg2Rad;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float depth = depthData[x, y];

                if (depth > 0 && depth < farClip) // Valid depth measurement
                {
                    // Convert pixel coordinates to 3D coordinates
                    float u = (float)x / width - 0.5f; // -0.5 to 0.5
                    float v = (float)y / height - 0.5f; // -0.5 to 0.5

                    float x3d = u * depth * Mathf.Tan(fovX / 2) * 2;
                    float y3d = -v * depth * Mathf.Tan(fovY / 2) * 2; // Negative because screen coordinates are inverted
                    float z3d = depth;

                    Vector3 point = transform.TransformPoint(new Vector3(x3d, y3d, z3d));
                    points[pointCount] = point;
                    pointCount++;
                }
            }
        }

        // Resize array to actual point count
        System.Array.Resize(ref points, pointCount);

        // In a real implementation, you would publish this point cloud to ROS
        Debug.Log($"Generated point cloud with {pointCount} points");
    }

    void OutputDepthImage()
    {
        // Create a texture to visualize the depth data
        Texture2D depthVisualization = new Texture2D(width, height, TextureFormat.RGB24, false);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float depth = depthData[x, y];

                // Normalize depth for visualization (0 = near, 1 = far)
                float normalizedDepth = Mathf.InverseLerp(nearClip, farClip, depth);

                // Map to grayscale color
                Color color = Color.Lerp(Color.white, Color.black, normalizedDepth);
                depthVisualization.SetPixel(x, y, color);
            }
        }

        depthVisualization.Apply();

        // In a real implementation, you would publish this image to ROS
    }

    public float[,] GetDepthData()
    {
        return depthData;
    }

    public float GetDepthAt(int x, int y)
    {
        if (x >= 0 && x < width && y >= 0 && y < height)
        {
            return depthData[x, y];
        }
        return float.MaxValue;
    }
}
```

## IMU Simulation

### IMU Characteristics

An IMU (Inertial Measurement Unit) typically combines:

- **Accelerometer**: Measures linear acceleration
- **Gyroscope**: Measures angular velocity
- **Magnetometer**: Measures magnetic field (for heading)

Key characteristics include:

- **Bias**: Systematic offset in measurements
- **Noise**: Random fluctuations in measurements
- **Drift**: Slow accumulation of errors over time
- **Scale Factor Error**: Mismatch between input and output
- **Cross-Axis Sensitivity**: Sensitivity to inputs in other axes

### IMU Simulation Implementation

```csharp
// Scripts/IMUSimulator.cs
using UnityEngine;

[System.Serializable]
public struct IMUReading
{
    public Vector3 linearAcceleration;  // m/s^2
    public Vector3 angularVelocity;     // rad/s
    public Vector3 magneticField;       // Tesla
    public Vector3 orientation;         // Euler angles in radians
    public double timestamp;            // Simulation time
}

public class IMUSimulator : MonoBehaviour
{
    [Header("Accelerometer Parameters")]
    public float accelerometerNoiseDensity = 8.75e-4f; // m/s^2 / sqrt(Hz)
    public float accelerometerRandomWalk = 2.92e-5f;   // m/s^3 / sqrt(Hz)
    public Vector3 accelerometerBias = Vector3.zero;
    public float accelerometerScaleFactorError = 0.001f; // 0.1%

    [Header("Gyroscope Parameters")]
    public float gyroscopeNoiseDensity = 1.64e-4f; // rad/s / sqrt(Hz)
    public float gyroscopeRandomWalk = 5.42e-6f;   // rad/s^2 / sqrt(Hz)
    public Vector3 gyroscopeBias = Vector3.zero;
    public float gyroscopeScaleFactorError = 0.001f; // 0.1%

    [Header("Magnetometer Parameters")]
    public float magnetometerNoise = 0.05e-6f; // Tesla
    public Vector3 magnetometerBias = Vector3.zero;

    [Header("Simulation Parameters")]
    public float updateRate = 100f; // Hz
    public bool enableDrift = true;

    private IMUReading currentReading;
    private float updateInterval;
    private float lastUpdate;

    // Bias drift accumulators
    private Vector3 accelerometerBiasDrift = Vector3.zero;
    private Vector3 gyroscopeBiasDrift = Vector3.zero;

    void Start()
    {
        updateInterval = 1f / updateRate;
        lastUpdate = -updateInterval; // Allow immediate first update

        // Initialize with current transform values
        UpdateIMUReading();
    }

    void Update()
    {
        if (Time.time - lastUpdate >= updateInterval)
        {
            UpdateIMUReading();
            lastUpdate = Time.time;
        }
    }

    void UpdateIMUReading()
    {
        // Get true values from Unity's physics
        Vector3 trueLinearAcceleration = GetTrueLinearAcceleration();
        Vector3 trueAngularVelocity = GetTrueAngularVelocity();
        Vector3 trueMagneticField = GetTrueMagneticField();
        Vector3 trueOrientation = transform.eulerAngles * Mathf.Deg2Rad;

        // Add noise and errors
        currentReading.linearAcceleration = AddAccelerometerNoise(trueLinearAcceleration);
        currentReading.angularVelocity = AddGyroscopeNoise(trueAngularVelocity);
        currentReading.magneticField = AddMagnetometerNoise(trueMagneticField);
        currentReading.orientation = trueOrientation; // Add noise to orientation if needed
        currentReading.timestamp = Time.time;

        // Publish simulated data
        PublishIMUData();
    }

    Vector3 GetTrueLinearAcceleration()
    {
        // Calculate true linear acceleration from Unity physics
        // This is a simplified approach - in practice, you'd need to account for gravity
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            // Remove gravity from the acceleration
            return rb.velocity - Physics.gravity * Time.deltaTime;
        }
        else
        {
            // If no rigidbody, use transform changes
            return (transform.position - transform.position) / (Time.deltaTime * Time.deltaTime);
        }
    }

    Vector3 GetTrueAngularVelocity()
    {
        // Calculate true angular velocity from Unity physics
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            return rb.angularVelocity;
        }
        else
        {
            // If no rigidbody, approximate from rotation change
            return Vector3.zero; // Simplified - would need to track rotation over time
        }
    }

    Vector3 GetTrueMagneticField()
    {
        // Return a typical Earth magnetic field vector
        // This should be adjusted based on geographic location
        return new Vector3(0.22f, 0.0f, 0.45f) * 1e-6f; // Tesla
    }

    Vector3 AddAccelerometerNoise(Vector3 trueAcceleration)
    {
        Vector3 noise = new Vector3(
            RandomGaussian() * accelerometerNoiseDensity,
            RandomGaussian() * accelerometerNoiseDensity,
            RandomGaussian() * accelerometerNoiseDensity
        );

        // Add bias (with drift if enabled)
        Vector3 bias = accelerometerBias + accelerometerBiasDrift;

        // Add scale factor error
        Vector3 scaledAcceleration = new Vector3(
            trueAcceleration.x * (1 + accelerometerScaleFactorError),
            trueAcceleration.y * (1 + accelerometerScaleFactorError),
            trueAcceleration.z * (1 + accelerometerScaleFactorError)
        );

        // Add random walk to bias
        if (enableDrift)
        {
            accelerometerBiasDrift += new Vector3(
                RandomGaussian() * accelerometerRandomWalk * Mathf.Sqrt(updateInterval),
                RandomGaussian() * accelerometerRandomWalk * Mathf.Sqrt(updateInterval),
                RandomGaussian() * accelerometerRandomWalk * Mathf.Sqrt(updateInterval)
            );
        }

        return scaledAcceleration + noise + bias;
    }

    Vector3 AddGyroscopeNoise(Vector3 trueAngularVelocity)
    {
        Vector3 noise = new Vector3(
            RandomGaussian() * gyroscopeNoiseDensity,
            RandomGaussian() * gyroscopeNoiseDensity,
            RandomGaussian() * gyroscopeNoiseDensity
        );

        // Add bias (with drift if enabled)
        Vector3 bias = gyroscopeBias + gyroscopeBiasDrift;

        // Add scale factor error
        Vector3 scaledAngularVelocity = new Vector3(
            trueAngularVelocity.x * (1 + gyroscopeScaleFactorError),
            trueAngularVelocity.y * (1 + gyroscopeScaleFactorError),
            trueAngularVelocity.z * (1 + gyroscopeScaleFactorError)
        );

        // Add random walk to bias
        if (enableDrift)
        {
            gyroscopeBiasDrift += new Vector3(
                RandomGaussian() * gyroscopeRandomWalk * Mathf.Sqrt(updateInterval),
                RandomGaussian() * gyroscopeRandomWalk * Mathf.Sqrt(updateInterval),
                RandomGaussian() * gyroscopeRandomWalk * Mathf.Sqrt(updateInterval)
            );
        }

        return scaledAngularVelocity + noise + bias;
    }

    Vector3 AddMagnetometerNoise(Vector3 trueMagneticField)
    {
        Vector3 noise = new Vector3(
            RandomGaussian() * magnetometerNoise,
            RandomGaussian() * magnetometerNoise,
            RandomGaussian() * magnetometerNoise
        );

        // Add bias
        Vector3 bias = magnetometerBias;

        return trueMagneticField + noise + bias;
    }

    float RandomGaussian()
    {
        // Box-Muller transform for Gaussian random numbers
        float u1 = Random.value;
        float u2 = Random.value;
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
    }

    void PublishIMUData()
    {
        // In a real implementation, this would publish to ROS/ROS 2
        Debug.Log($"IMU: Acc=({currentReading.linearAcceleration.x:F3}, {currentReading.linearAcceleration.y:F3}, {currentReading.linearAcceleration.z:F3})");
    }

    public IMUReading GetIMUReading()
    {
        return currentReading;
    }

    public void ResetBiasDrift()
    {
        accelerometerBiasDrift = Vector3.zero;
        gyroscopeBiasDrift = Vector3.zero;
    }
}
```

## Sensor Fusion and Integration

### Multi-Sensor Data Integration

```csharp
// Scripts/SensorFusion.cs
using UnityEngine;
using System.Collections.Generic;

public class SensorFusion : MonoBehaviour
{
    [Header("Sensor References")]
    public LiDARSimulator lidar;
    public DepthCameraSimulator depthCamera;
    public IMUSimulator imu;

    [Header("Fusion Parameters")]
    public float lidarWeight = 0.4f;
    public float depthWeight = 0.3f;
    public float imuWeight = 0.3f;

    private Queue<IMUReading> imuBuffer;
    private float bufferDuration = 1.0f; // seconds

    void Start()
    {
        imuBuffer = new Queue<IMUReading>();
    }

    void Update()
    {
        // Integrate sensor data
        ProcessSensorData();
    }

    void ProcessSensorData()
    {
        // Get current readings from all sensors
        float[] lidarData = lidar != null ? lidar.GetRanges() : new float[0];
        float[,] depthData = depthCamera != null ? depthCamera.GetDepthData() : new float[0, 0];
        IMUReading imuData = imu != null ? imu.GetIMUReading() : new IMUReading();

        // Store IMU data in buffer for temporal fusion
        if (imu != null)
        {
            imuBuffer.Enqueue(imuData);

            // Remove old data from buffer
            while (imuBuffer.Count > 0 &&
                   Time.time - imuBuffer.Peek().timestamp > bufferDuration)
            {
                imuBuffer.Dequeue();
            }
        }

        // Perform sensor fusion (simplified example)
        PerformFusion(lidarData, depthData, imuData);
    }

    void PerformFusion(float[] lidarData, float[,] depthData, IMUReading imuData)
    {
        // Example fusion algorithm: combine position estimates
        Vector3 positionEstimate = Vector3.zero;

        // Calculate position from IMU integration
        if (imuBuffer.Count > 1)
        {
            IMUReading first = imuBuffer.Peek();
            IMUReading last = imuBuffer.ToArray()[imuBuffer.Count - 1];

            float deltaTime = (float)(last.timestamp - first.timestamp);
            if (deltaTime > 0)
            {
                Vector3 velocity = (last.linearAcceleration + first.linearAcceleration) * 0.5f * deltaTime;
                positionEstimate += velocity * imuWeight;
            }
        }

        // In a real implementation, you would combine this with LiDAR and depth data
        // for more accurate position estimation

        Debug.Log($"Fused position estimate: {positionEstimate}");
    }

    public Vector3 GetFusedPositionEstimate()
    {
        // Return the fused position estimate
        return transform.position; // Simplified
    }
}
```

## Best Practices for Sensor Simulation

### Realistic Noise Modeling

- **Characterize Real Sensors**: Understand the noise characteristics of your actual sensors
- **Include Bias and Drift**: Model systematic errors that accumulate over time
- **Validate Against Reality**: Compare simulated and real sensor data
- **Temperature Effects**: Consider how temperature affects sensor performance

### Performance Optimization

- **Efficient Raycasting**: Use optimized algorithms for LiDAR simulation
- **Level of Detail**: Reduce sensor resolution when performance is critical
- **Parallel Processing**: Use multi-threading for sensor simulation
- **Caching**: Cache results when possible for repeated queries

### Validation and Testing

- **Ground Truth Comparison**: Compare simulated data to known ground truth
- **Edge Case Testing**: Test with challenging scenarios (bright light, occlusions, etc.)
- **Cross-Sensor Validation**: Verify consistency between different sensor modalities
- **Temporal Consistency**: Ensure sensor data is consistent over time

## Integration with ROS/ROS 2

### ROS Message Publishing

```csharp
// Scripts/ROSPublisher.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class ROSPublisher : MonoBehaviour
{
    private ROSConnection ros;

    [Header("Topic Names")]
    public string laserScanTopic = "scan";
    public string pointCloudTopic = "point_cloud";
    public string imuTopic = "imu/data";

    [Header("Sensors")]
    public LiDARSimulator lidar;
    public DepthCameraSimulator depthCamera;
    public IMUSimulator imu;

    void Start()
    {
        ros = ROSConnection.instance;
    }

    void Update()
    {
        PublishSensorData();
    }

    void PublishSensorData()
    {
        if (lidar != null)
        {
            PublishLaserScan();
        }

        if (depthCamera != null)
        {
            PublishPointCloud();
        }

        if (imu != null)
        {
            PublishIMU();
        }
    }

    void PublishLaserScan()
    {
        var scanMsg = new sensor_msgs.LaserScanMsg();
        scanMsg.header = new std_msgs.HeaderMsg();
        scanMsg.header.stamp = new builtin_interfaces.TimeMsg();
        scanMsg.header.frame_id = "lidar_frame";

        float[] ranges = lidar.GetRanges();
        scanMsg.ranges = new float[ranges.Length];
        for (int i = 0; i < ranges.Length; i++)
        {
            scanMsg.ranges[i] = ranges[i];
        }

        scanMsg.angle_min = lidar.minAngle;
        scanMsg.angle_max = lidar.maxAngle;
        scanMsg.angle_increment = (lidar.maxAngle - lidar.minAngle) / (ranges.Length - 1);
        scanMsg.range_min = 0.1f;
        scanMsg.range_max = lidar.maxRange;

        ros.Publish(laserScanTopic, scanMsg);
    }

    void PublishPointCloud()
    {
        // Publish point cloud from depth camera
        // Implementation would convert depth data to PointCloud2 message
    }

    void PublishIMU()
    {
        IMUReading reading = imu.GetIMUReading();

        var imuMsg = new sensor_msgs.ImuMsg();
        imuMsg.header = new std_msgs.HeaderMsg();
        imuMsg.header.stamp = new builtin_interfaces.TimeMsg();
        imuMsg.header.frame_id = "imu_frame";

        // Convert to ROS message format
        imuMsg.linear_acceleration.x = reading.linearAcceleration.x;
        imuMsg.linear_acceleration.y = reading.linearAcceleration.y;
        imuMsg.linear_acceleration.z = reading.linearAcceleration.z;

        imuMsg.angular_velocity.x = reading.angularVelocity.x;
        imuMsg.angular_velocity.y = reading.angularVelocity.y;
        imuMsg.angular_velocity.z = reading.angularVelocity.z;

        // Convert Euler angles to quaternion
        imuMsg.orientation = Unity.Robotics.ROSTCPConnector.MessageExtensions.To<geometry_msgs.QuaternionMsg>(
            Quaternion.Euler(reading.orientation * Mathf.Rad2Deg)
        );

        ros.Publish(imuTopic, imuMsg);
    }
}
```

## Hands-On Exercise

1. Implement a LiDAR sensor simulation with realistic noise characteristics
2. Create a depth camera simulator with point cloud generation
3. Develop an IMU simulator with bias drift and random walk
4. Integrate all sensors with ROS/ROS 2 message publishing
5. Validate sensor data against expected real-world behavior

## Summary

Sensor simulation is fundamental to humanoid robotics development, providing realistic data for algorithm development and testing. By accurately modeling sensor characteristics including noise, bias, and drift, you can create effective simulation environments that closely match real-world conditions. Proper integration with ROS/ROS 2 enables seamless transition from simulation to real hardware. In the next module, we'll explore NVIDIA Isaac Sim for photorealistic simulation and synthetic data generation.

## Learning Path Adjustment

Based on your experience level, you may want to focus on:

- **Beginner**: Focus on basic sensor simulation and visualization
- **Intermediate**: Dive deeper into noise modeling and sensor fusion
- **Advanced**: Explore advanced sensor physics and AI training applications