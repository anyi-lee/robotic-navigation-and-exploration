# HW2: Kinematic Models and Path-Tracking Control

This assignment connects path geometry to executable vehicle motion. It covers a bicycle kinematic model, curvature-aware speed planning, longitudinal control, and four lateral path-tracking methods.

## Implemented components

| Area | Implementation |
| --- | --- |
| Vehicle model | Bicycle kinematic state update |
| Speed planning | Curvature speed limits and forward/backward acceleration smoothing |
| Longitudinal control | PID speed tracking |
| Lateral control | PID, Pure Pursuit, Stanley, and LQR |
| Model conversion | Linear/angular velocity to differential wheel speeds |

Selected implementation files are available under [`src/`](src/). They depend on course-provided simulation utilities and track data that are not reproduced in full.

## Controller design

- **PID** minimizes signed cross-track error.
- **Pure Pursuit** targets a speed-dependent look-ahead point.
- **Stanley** combines heading and cross-track corrections.
- **LQR** solves a discrete linearized bicycle-model control problem for steering angle or steering rate.

## Report

[Open PDF](https://anyi-lee.github.io/robotic-navigation-and-exploration/hw2-motion-control/report/hw2-report.pdf)
