---
sidebar_position: 3
---

# Unity-Based HRI Visualization

## Chapter Objectives

- Set up Unity for Human-Robot Interaction (HRI) visualization
- Create realistic humanoid robot models in Unity
- Implement intuitive user interfaces for robot control
- Develop immersive visualization environments

## Introduction to Unity for HRI

Unity has become a popular choice for Human-Robot Interaction (HRI) visualization due to its:

- **High-Quality Graphics**: Photorealistic rendering capabilities
- **Intuitive Interface**: Visual development environment
- **Asset Ecosystem**: Extensive library of 3D models and materials
- **Cross-Platform Deployment**: Web, desktop, and VR/AR support
- **Real-Time Performance**: Optimized for interactive applications

### Unity vs. Traditional Robotics Simulation

| Aspect | Unity | Traditional Robotics Simulators |
|--------|-------|--------------------------------|
| Graphics Quality | High | Moderate to High |
| HRI Focus | Excellent | Limited |
| Development Speed | Fast | Moderate |
| Physics Accuracy | Good | Excellent |
| User Experience | Excellent | Basic |

## Setting Up Unity for Robotics

### Unity Robotics Setup

1. **Install Unity Hub** and create a new 3D project
2. **Install Unity Robotics packages**:
   - Unity Robotics Hub
   - ROS-TCP-Connector
   - Unity Perception (for synthetic data)

3. **Configure project settings** for robotics applications

### Basic Robotics Project Structure

```
UnityRoboticsProject/
├── Assets/
│   ├── Scripts/
│   │   ├── RobotControl/
│   │   ├── Visualization/
│   │   └── Communication/
│   ├── Models/
│   │   ├── HumanoidRobot/
│   │   └── Environments/
│   ├── Materials/
│   ├── Prefabs/
│   └── Scenes/
└── Packages/
```

### Unity Robotics Hub Installation

```csharp
// Example setup script for Unity Robotics
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class RobotVisualizationManager : MonoBehaviour
{
    [SerializeField] private string rosIPAddress = "127.0.0.1";
    [SerializeField] private int rosPort = 10000;

    private ROSConnection rosConnection;

    void Start()
    {
        // Connect to ROS
        rosConnection = ROSConnection.instance;
        rosConnection.rosIPAddress = rosIPAddress;
        rosConnection.rosPort = rosPort;

        // Subscribe to robot state topics
        rosConnection.Subscribe<sensor_msgs.JointStateMsg>("joint_states", OnJointStateReceived);
        rosConnection.Subscribe<geometry_msgs.TwistMsg>("cmd_vel", OnVelocityCommandReceived);
    }

    void OnJointStateReceived(sensor_msgs.JointStateMsg jointState)
    {
        // Update robot visualization based on joint states
        UpdateRobotJoints(jointState);
    }

    void OnVelocityCommandReceived(geometry_msgs.TwistMsg cmdVel)
    {
        // Update robot visualization based on velocity commands
        UpdateRobotMotion(cmdVel);
    }

    void UpdateRobotJoints(sensor_msgs.JointStateMsg jointState)
    {
        // Find robot parts and update their positions
        for (int i = 0; i < jointState.name.Length; i++)
        {
            Transform joint = transform.Find(jointState.name[i]);
            if (joint != null)
            {
                // Apply joint position to transform
                joint.localRotation = Quaternion.Euler(0, 0, jointState.position[i] * Mathf.Rad2Deg);
            }
        }
    }
}
```

## Creating Humanoid Robot Models in Unity

### Robot Model Structure

A humanoid robot in Unity typically consists of:

- **Root Object**: Main parent for the entire robot
- **Links**: Individual body parts (torso, head, arms, legs)
- **Joints**: Connection points between links
- **Colliders**: Physics collision shapes
- **Materials**: Visual appearance properties

### Creating a Simple Humanoid Robot

```csharp
// Scripts/HumanoidRobot.cs
using UnityEngine;

public class HumanoidRobot : MonoBehaviour
{
    [Header("Body Parts")]
    public Transform torso;
    public Transform head;
    public Transform leftArm;
    public Transform rightArm;
    public Transform leftLeg;
    public Transform rightLeg;

    [Header("Joint Configuration")]
    public float headRotationLimit = 45f;
    public float armRotationLimit = 90f;
    public float legRotationLimit = 60f;

    void Start()
    {
        ValidateSetup();
    }

    void ValidateSetup()
    {
        if (torso == null) torso = transform.Find("torso");
        if (head == null) head = transform.Find("head");
        if (leftArm == null) leftArm = transform.Find("left_arm");
        if (rightArm == null) rightArm = transform.Find("right_arm");
        if (leftLeg == null) leftLeg = transform.Find("left_leg");
        if (rightLeg == null) rightLeg = transform.Find("right_leg");
    }

    public void SetJointPositions(float headYaw, float leftArmAngle, float rightArmAngle,
                                 float leftLegAngle, float rightLegAngle)
    {
        if (head != null)
        {
            head.localRotation = Quaternion.Euler(0, Mathf.Clamp(headYaw, -headRotationLimit, headRotationLimit), 0);
        }

        if (leftArm != null)
        {
            leftArm.localRotation = Quaternion.Euler(0, 0, Mathf.Clamp(leftArmAngle, -armRotationLimit, armRotationLimit));
        }

        if (rightArm != null)
        {
            rightArm.localRotation = Quaternion.Euler(0, 0, Mathf.Clamp(rightArmAngle, -armRotationLimit, armRotationLimit));
        }

        if (leftLeg != null)
        {
            leftLeg.localRotation = Quaternion.Euler(0, 0, Mathf.Clamp(leftLegAngle, -legRotationLimit, legRotationLimit));
        }

        if (rightLeg != null)
        {
            rightLeg.localRotation = Quaternion.Euler(0, 0, Mathf.Clamp(rightLegAngle, -legRotationLimit, legRotationLimit));
        }
    }

    public void SetTorsoPosition(Vector3 position)
    {
        transform.position = position;
    }
}
```

### Advanced Robot Rigging

For more sophisticated humanoid robots, consider using Unity's Animation Rigging package:

```csharp
// Scripts/AdvancedHumanoidController.cs
using UnityEngine;
using UnityEngine.Animations.Rigging;

[RequireComponent(typeof(Animator))]
public class AdvancedHumanoidController : MonoBehaviour
{
    private Animator animator;
    private RigBuilder rigBuilder;

    [Header("Rig Configuration")]
    public bool useRigging = true;
    public Rig[] rigs;

    [Header("IK Targets")]
    public Transform leftHandTarget;
    public Transform rightHandTarget;
    public Transform leftFootTarget;
    public Transform rightFootTarget;

    void Start()
    {
        animator = GetComponent<Animator>();
        rigBuilder = GetComponent<RigBuilder>();

        SetupRigging();
    }

    void SetupRigging()
    {
        if (useRigging && rigBuilder != null)
        {
            rigBuilder.Build();
        }
    }

    void Update()
    {
        if (useRigging)
        {
            UpdateIKTargets();
        }
    }

    void UpdateIKTargets()
    {
        // Update inverse kinematics targets based on external input
        // This could come from ROS messages or other sources
    }

    public void SetIKPositions(Vector3 leftHandPos, Vector3 rightHandPos,
                              Vector3 leftFootPos, Vector3 rightFootPos)
    {
        if (leftHandTarget != null) leftHandTarget.position = leftHandPos;
        if (rightHandTarget != null) rightHandTarget.position = rightHandPos;
        if (leftFootTarget != null) leftFootTarget.position = leftFootPos;
        if (rightFootTarget != null) rightFootTarget.position = rightFootPos;
    }
}
```

## Creating Immersive Visualization Environments

### Environment Design Principles

For effective HRI visualization, environments should be:

- **Realistic**: Represent real-world scenarios accurately
- **Interactive**: Allow user manipulation and robot interaction
- **Scalable**: Support different complexity levels
- **Performance-Optimized**: Maintain smooth frame rates

### Creating a Room Environment

```csharp
// Scripts/RoomEnvironment.cs
using UnityEngine;

public class RoomEnvironment : MonoBehaviour
{
    [Header("Room Dimensions")]
    public Vector3 roomSize = new Vector3(10f, 5f, 8f);

    [Header("Furniture")]
    public GameObject[] furniturePrefabs;
    public Transform furnitureParent;

    [Header("Lighting")]
    public Light mainLight;
    public Color ambientColor = Color.gray;

    [Header("Interactive Elements")]
    public GameObject[] interactiveObjects;

    void Start()
    {
        CreateRoomStructure();
        AddFurniture();
        SetupLighting();
        ConfigureInteractiveElements();
    }

    void CreateRoomStructure()
    {
        // Create floor
        CreateRoomWall(Vector3.zero, new Vector3(roomSize.x, 0.1f, roomSize.z), "Floor");

        // Create walls
        CreateRoomWall(new Vector3(0, roomSize.y / 2, 0),
                      new Vector3(roomSize.x, roomSize.y, 0.1f), "BackWall");
        CreateRoomWall(new Vector3(0, roomSize.y / 2, roomSize.z),
                      new Vector3(roomSize.x, roomSize.y, 0.1f), "FrontWall");
        CreateRoomWall(new Vector3(-roomSize.x / 2, roomSize.y / 2, roomSize.z / 2),
                      new Vector3(0.1f, roomSize.y, roomSize.z), "LeftWall");
        CreateRoomWall(new Vector3(roomSize.x / 2, roomSize.y / 2, roomSize.z / 2),
                      new Vector3(0.1f, roomSize.y, roomSize.z), "RightWall");
    }

    GameObject CreateRoomWall(Vector3 position, Vector3 size, string name)
    {
        GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.name = name;
        wall.transform.position = position;
        wall.transform.localScale = size;

        // Remove collider if it's the floor to allow robot movement
        if (name != "Floor")
        {
            wall.AddComponent<BoxCollider>();
        }
        else
        {
            // Make floor a trigger for robot detection
            wall.GetComponent<BoxCollider>().isTrigger = true;
        }

        return wall;
    }

    void AddFurniture()
    {
        if (furniturePrefabs.Length > 0 && furnitureParent != null)
        {
            // Randomly place furniture in the room
            foreach (GameObject prefab in furniturePrefabs)
            {
                if (prefab != null)
                {
                    Vector3 randomPos = new Vector3(
                        Random.Range(-roomSize.x / 3, roomSize.x / 3),
                        0,
                        Random.Range(0, roomSize.z / 2)
                    );

                    GameObject furniture = Instantiate(prefab, randomPos, Quaternion.identity, furnitureParent);
                    furniture.AddComponent<Rigidbody>(); // Make physics-enabled
                }
            }
        }
    }

    void SetupLighting()
    {
        if (mainLight != null)
        {
            mainLight.type = LightType.Directional;
            mainLight.color = Color.white;
            mainLight.intensity = 1f;
        }

        RenderSettings.ambientLight = ambientColor;
    }

    void ConfigureInteractiveElements()
    {
        foreach (GameObject obj in interactiveObjects)
        {
            if (obj != null)
            {
                // Add interaction components
                obj.AddComponent<InteractiveObject>();
            }
        }
    }
}
```

## HRI User Interface Design

### Creating Intuitive Control Interfaces

```csharp
// Scripts/HRIInterface.cs
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class HRIInterface : MonoBehaviour
{
    [Header("Robot Status Display")]
    public TextMeshProUGUI statusText;
    public Slider batterySlider;
    public TextMeshProUGUI positionText;

    [Header("Control Inputs")]
    public Button moveForwardButton;
    public Button moveBackwardButton;
    public Button turnLeftButton;
    public Button turnRightButton;
    public Button stopButton;

    [Header("Gesture Controls")]
    public Button waveButton;
    public Button pointButton;
    public Button gestureMenuButton;

    [Header("Camera Controls")]
    public Button firstPersonButton;
    public Button thirdPersonButton;
    public Button birdEyeButton;

    [Header("Visualization Settings")]
    public Toggle showPathToggle;
    public Toggle showSensorsToggle;
    public Slider transparencySlider;

    private HumanoidRobot robot;

    void Start()
    {
        robot = FindObjectOfType<HumanoidRobot>();
        SetupUIEvents();
        UpdateStatusDisplay();
    }

    void SetupUIEvents()
    {
        // Movement controls
        moveForwardButton.onClick.AddListener(() => MoveRobot(Vector3.forward));
        moveBackwardButton.onClick.AddListener(() => MoveRobot(Vector3.back));
        turnLeftButton.onClick.AddListener(() => TurnRobot(-1f));
        turnRightButton.onClick.AddListener(() => TurnRobot(1f));
        stopButton.onClick.AddListener(StopRobot);

        // Gesture controls
        waveButton.onClick.AddListener(() => PerformGesture("wave"));
        pointButton.onClick.AddListener(() => PerformGesture("point"));

        // Camera controls
        firstPersonButton.onClick.AddListener(() => SwitchCamera("first"));
        thirdPersonButton.onClick.AddListener(() => SwitchCamera("third"));
        birdEyeButton.onClick.AddListener(() => SwitchCamera("bird"));

        // Visualization toggles
        showPathToggle.onValueChanged.AddListener(OnPathToggleChanged);
        showSensorsToggle.onValueChanged.AddListener(OnSensorsToggleChanged);
        transparencySlider.onValueChanged.AddListener(OnTransparencyChanged);
    }

    void MoveRobot(Vector3 direction)
    {
        if (robot != null)
        {
            robot.SetTorsoPosition(robot.transform.position + direction * Time.deltaTime * 2f);
            UpdateStatusDisplay();
        }
    }

    void TurnRobot(float direction)
    {
        if (robot != null)
        {
            robot.transform.Rotate(Vector3.up, direction * 90f * Time.deltaTime);
            UpdateStatusDisplay();
        }
    }

    void StopRobot()
    {
        // Stop robot movement
        UpdateStatusDisplay();
    }

    void PerformGesture(string gestureName)
    {
        switch (gestureName)
        {
            case "wave":
                // Animate waving gesture
                StartCoroutine(AnimateWaveGesture());
                break;
            case "point":
                // Animate pointing gesture
                StartCoroutine(AnimatePointGesture());
                break;
        }
    }

    System.Collections.IEnumerator AnimateWaveGesture()
    {
        // Example wave animation
        if (robot != null)
        {
            for (float t = 0; t < 1f; t += Time.deltaTime)
            {
                float angle = Mathf.Sin(t * Mathf.PI * 4) * 30f;
                robot.SetJointPositions(0, angle, -angle, 0, 0);
                yield return null;
            }
            // Return to neutral position
            robot.SetJointPositions(0, 0, 0, 0, 0);
        }
    }

    System.Collections.IEnumerator AnimatePointGesture()
    {
        // Example pointing animation
        if (robot != null)
        {
            robot.SetJointPositions(0, 45, -45, 0, 0);
            yield return new WaitForSeconds(1f);
            robot.SetJointPositions(0, 0, 0, 0, 0);
        }
    }

    void SwitchCamera(string cameraMode)
    {
        // Implement camera switching logic
        switch (cameraMode)
        {
            case "first":
                // Switch to first-person view
                break;
            case "third":
                // Switch to third-person view
                break;
            case "bird":
                // Switch to bird's eye view
                break;
        }
    }

    void OnPathToggleChanged(bool isOn)
    {
        // Show/hide robot path visualization
    }

    void OnSensorsToggleChanged(bool isOn)
    {
        // Show/hide sensor visualization
    }

    void OnTransparencyChanged(float value)
    {
        // Adjust robot transparency
        if (robot != null)
        {
            SetTransparency(robot.gameObject, value);
        }
    }

    void SetTransparency(GameObject obj, float transparency)
    {
        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();
        foreach (Renderer renderer in renderers)
        {
            Color color = renderer.material.color;
            color.a = 1f - transparency;
            renderer.material.color = color;
        }
    }

    void UpdateStatusDisplay()
    {
        if (robot != null && statusText != null)
        {
            statusText.text = $"Status: Active\nPosition: {robot.transform.position}";
            positionText.text = $"X: {robot.transform.position.x:F2}, Y: {robot.transform.position.y:F2}, Z: {robot.transform.position.z:F2}";
        }
    }
}
```

## Advanced Visualization Techniques

### Sensor Visualization

```csharp
// Scripts/SensorVisualization.cs
using UnityEngine;

public class SensorVisualization : MonoBehaviour
{
    [Header("Sensor Configuration")]
    public float lidarRange = 10f;
    public int lidarBeams = 360;
    public float cameraFOV = 60f;
    public float cameraRange = 5f;

    [Header("Visualization Settings")]
    public Color lidarColor = Color.red;
    public Color cameraColor = Color.blue;
    public bool showSensors = true;

    private LineRenderer lidarRenderer;
    private GameObject cameraVisualization;

    void Start()
    {
        SetupLidarVisualization();
        SetupCameraVisualization();
    }

    void SetupLidarVisualization()
    {
        lidarRenderer = gameObject.AddComponent<LineRenderer>();
        lidarRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lidarRenderer.color = lidarColor;
        lidarRenderer.startWidth = 0.02f;
        lidarRenderer.endWidth = 0.02f;
        lidarRenderer.positionCount = lidarBeams + 1; // +1 to close the circle
    }

    void SetupCameraVisualization()
    {
        cameraVisualization = new GameObject("CameraVisualization");
        cameraVisualization.transform.SetParent(transform);
        // Create pyramid mesh to represent camera view frustum
    }

    void Update()
    {
        if (showSensors)
        {
            UpdateLidarVisualization();
            UpdateCameraVisualization();
        }
    }

    void UpdateLidarVisualization()
    {
        if (lidarRenderer != null)
        {
            Vector3[] positions = new Vector3[lidarBeams + 1];

            for (int i = 0; i < lidarBeams; i++)
            {
                float angle = (float)i / lidarBeams * Mathf.PI * 2f;
                Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));

                // Perform raycast to find obstacles
                if (Physics.Raycast(transform.position, direction, out RaycastHit hit, lidarRange))
                {
                    positions[i] = hit.point;
                }
                else
                {
                    positions[i] = transform.position + direction * lidarRange;
                }
            }

            // Close the circle
            positions[lidarBeams] = positions[0];

            lidarRenderer.SetPositions(positions);
        }
    }

    void UpdateCameraVisualization()
    {
        // Update camera frustum visualization based on current position and orientation
    }
}
```

### Performance Optimization

```csharp
// Scripts/VisualizationOptimizer.cs
using UnityEngine;

public class VisualizationOptimizer : MonoBehaviour
{
    [Header("Performance Settings")]
    public float maxFramerate = 60f;
    public bool useLOD = true;
    public float lodDistance = 10f;
    public bool cullDistantObjects = true;

    [Header("Quality Settings")]
    public bool enableShadows = true;
    public bool enableReflections = true;
    public int textureQuality = 2; // 0=low, 1=medium, 2=high

    private Camera mainCamera;

    void Start()
    {
        mainCamera = Camera.main;
        ApplyQualitySettings();
        Application.targetFrameRate = Mathf.RoundToInt(maxFramerate);
    }

    void ApplyQualitySettings()
    {
        QualitySettings.shadowDistance = enableShadows ? 50f : 10f;
        QualitySettings.shadowResolution = enableShadows ? ShadowResolution.High : ShadowResolution.Low;

        // Set texture quality
        QualitySettings.masterTextureLimit = 3 - textureQuality; // 0=full, 3=1/8 resolution
    }

    void Update()
    {
        if (useLOD)
        {
            UpdateLOD();
        }

        if (cullDistantObjects)
        {
            CullDistantObjects();
        }
    }

    void UpdateLOD()
    {
        // Example LOD system
        LODGroup[] lodGroups = FindObjectsOfType<LODGroup>();
        foreach (LODGroup lodGroup in lodGroups)
        {
            float distance = Vector3.Distance(mainCamera.transform.position, lodGroup.transform.position);
            lodGroup.animateCrossFading = distance < lodDistance * 2f;
        }
    }

    void CullDistantObjects()
    {
        Renderer[] renderers = FindObjectsOfType<Renderer>();
        foreach (Renderer renderer in renderers)
        {
            float distance = Vector3.Distance(mainCamera.transform.position, renderer.transform.position);
            renderer.enabled = distance < lodDistance * 3f;
        }
    }
}
```

## Best Practices for HRI Visualization

### User Experience Design

- **Intuitive Controls**: Use familiar UI patterns and clear affordances
- **Responsive Feedback**: Provide immediate visual feedback for user actions
- **Accessibility**: Consider users with different abilities and preferences
- **Consistency**: Maintain consistent visual language throughout the interface

### Performance Considerations

- **Optimize Draw Calls**: Batch similar objects and use instancing
- **Level of Detail**: Use LOD systems for distant objects
- **Occlusion Culling**: Don't render objects not visible to the camera
- **Texture Atlasing**: Combine multiple textures into single atlases

### Realism vs. Clarity

- **Visual Clarity**: Ensure important information is clearly visible
- **Performance Balance**: Balance visual quality with performance requirements
- **Focus Areas**: Highlight important robot states and sensor data
- **Color Coding**: Use consistent color schemes for different data types

## Integration with ROS/ROS 2

### ROS Bridge Implementation

```csharp
// Scripts/ROSIntegration.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class ROSIntegration : MonoBehaviour
{
    private ROSConnection ros;
    private HumanoidRobot robot;

    [Header("ROS Topics")]
    public string jointStateTopic = "joint_states";
    public string robotCommandTopic = "robot_command";
    public string sensorDataTopic = "sensor_data";

    void Start()
    {
        ros = ROSConnection.instance;
        robot = FindObjectOfType<HumanoidRobot>();

        // Subscribe to ROS topics
        ros.Subscribe<sensor_msgs.JointStateMsg>(jointStateTopic, OnJointStateReceived);
        ros.Subscribe<geometry_msgs.TwistMsg>(robotCommandTopic, OnRobotCommandReceived);
    }

    void OnJointStateReceived(sensor_msgs.JointStateMsg jointState)
    {
        // Update Unity robot visualization based on ROS joint states
        if (robot != null)
        {
            // Update each joint position based on ROS message
            for (int i = 0; i < jointState.name.Length && i < jointState.position.Length; i++)
            {
                robot.SetJointPosition(jointState.name[i], jointState.position[i]);
            }
        }
    }

    void OnRobotCommandReceived(geometry_msgs.TwistMsg cmd)
    {
        // Update visualization based on robot commands
        if (robot != null)
        {
            // Visualize intended movement
            robot.SetTargetVelocity(new Vector3((float)cmd.linear.x, (float)cmd.linear.y, (float)cmd.linear.z));
        }
    }

    public void SendRobotCommand(geometry_msgs.TwistMsg command)
    {
        ros.Publish(robotCommandTopic, command);
    }
}
```

## Hands-On Exercise

1. Create a Unity project with a humanoid robot model
2. Implement basic joint control visualization
3. Create an interactive environment with furniture
4. Develop a simple HRI interface with movement controls
5. Add sensor visualization (LiDAR, camera, etc.)

## Summary

Unity provides powerful capabilities for HRI visualization, combining high-quality graphics with intuitive user interfaces. By following best practices for environment design, performance optimization, and user experience, you can create compelling visualization systems that enhance human-robot interaction. In the next chapter, we'll explore sensor simulation including LiDAR, depth cameras, and IMU sensors.

## Learning Path Adjustment

Based on your experience level, you may want to focus on:

- **Beginner**: Focus on basic Unity setup and simple robot visualization
- **Intermediate**: Dive deeper into UI design and ROS integration
- **Advanced**: Explore advanced rendering techniques and VR/AR implementations