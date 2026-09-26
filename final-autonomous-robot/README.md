# Final Project: Autonomous Mobile Manipulation

This individual final project integrates perception, localization, navigation, vehicle control, and robotic-arm manipulation in the PROS Twin simulation environment.

[![Demonstration](assets/demo-thumbnail.jpg)](assets/demo.mp4)

**[Watch the demonstration](assets/demo.mp4)** · **[Read the seven-page technical report](report/final-technical-report.pdf)**

## Task results

| Task | Implemented behavior | Result |
| --- | --- | --- |
| Bear retrieval | Search, visual alignment, arm preparation, grasp, Nav2 return, AMCL verification, release | Completed |
| Bridge traversal | Bear retrieval, stable grasp, constrained local traversal, recovery, return navigation | Completed |
| Door interaction | Knob detection, coarse approach, fine visual alignment, handle-press sequence, forward push | Partially completed |

Task 3 reached perception, alignment, and manipulation testing, but the complete door-opening behavior was not reliable during the final demonstration.

## Control architecture

```mermaid
stateDiagram-v2
    [*] --> Search
    Search --> Align: stable target
    Align --> Search: target lost
    Align --> Grasp: centered and close
    Grasp --> Navigate
    Navigate --> Release: AMCL goal verified
    Release --> [*]
```

Nav2 handles long-range movement in mapped open space. Local commands handle close visual alignment and predictable motion in constrained areas such as the bridge and door approach.

## Reliability mechanisms

- **Multi-frame confirmation:** major transitions require repeated consistent detections.
- **Target-loss recovery:** controlled search resumes when the target leaves the camera view.
- **Depth fallback:** vertical image position supplements unstable close-range depth.
- **Dynamic home pose:** the controller records the initial AMCL pose at task start.
- **Return verification:** object release requires geometric proximity to the stored goal rather than relying only on a navigation-finish flag.
- **Plan recovery:** empty or exhausted Nav2 plans trigger goal publication and replanning instead of false completion.

## Selected source

| File | Project-specific contribution |
| --- | --- |
| [`mode_manager.py`](src/mode_manager.py) | Multi-task finite-state controller and recovery logic |
| [`nav_processing.py`](src/nav_processing.py) | Nav2 plan acquisition, tracking, replanning, and finish checks |
| [`object_detection_node.py`](src/object_detection_node.py) | YOLO/RGB-D target data generation |
| [`ros_communicator.py`](src/ros_communicator.py) | ROS 2 topic, AMCL, goal, and command integration |
| [`arm_controller_2D.py`](src/arm_controller_2D.py) | Target-frame conversion and grasp execution support |
| [`data_processor.py`](src/data_processor.py) | Sensor-data validation and AMCL processing guard |

Compared with the clean course-framework baseline at `pros_car` commit `3659e63`, the submitted project modified five Python files. The largest change was `mode_manager.py`, which grew from 87 to 1,447 lines as the task controller evolved.

## Framework attribution

The project builds on the course-provided [pros_car](https://github.com/asd56585452/pros_car) and [pros_app](https://github.com/asd56585452/pros_app) frameworks. Only selected modified files are included here; the complete upstream repositories are not redistributed.

