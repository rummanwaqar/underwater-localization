# Localization
Robot localization using unscented Kalman filter. This package does not stink :)

* Output state message param: `/topic/sensor/odom_state `
* Output frame: `temp_odom`

## Run
* Node: `roslaunch au_localization localizer.launch`
* Tests: `catkin_make run_tests_au_localization`

## File structure
* `types.h` - declares custom data-types
* `ufk.h` - generic UKF implementation
* `ukf_helpers.h` - implements UKF sigma point class and unscented transform function
* `localization(.h .cpp)` - implements a localization filter using UKF; process and measurement models defined here
* `localization_ros(.h .cpp)` - ROS wrapper for localization; implements all ROS inputs and outputs for localization
* `localization_node.cpp` - runs LocalizationRos class as a ros node

## ROS architecture
All sensor measurements are transformed to robot's base_link frame and added to a temporal 
priority queue. 
The UKF filter is periodically called by a timer (rate defined by filter frequency param). All 
past measurements in the queue are then sequentially passed to the filter. The node then 
publishes the state estimate, and `odom->base_link` transform.
 
![ros_architecture]

At startup, the node waits for an IMU message to initialize the filter state.

_Note: robot pose is w.r.t. world; robot velocity is w.r.t. robot (base_link)_


## Unscented Kalman Filter

### State
The robot state is defined as a 15 element vector

![state]

_Note: φ = roll; θ = pitch; ψ = yaw_

### Predict Step

#### State transition model (F)
The state transition model is used to make predict next robot state from previous state.
A 3D kinematics model with constant linear acceleration and constant angular velocity is used.

_(Reference: Section 2.2.1 from Handbook of Marine Craft Hydrodynamics and Motion Control, First Edition. Thor I. Fossen.)_

##### Linear

![rotation_1]

![rotation_2]

![predict_position]

![predict_velocity]

##### Angular

![predict_orientation]

_Note: this transformation is undefined if θ=+/-90 degrees_

[//]: # (Image References)

[ros_architecture]: ./docs/ros_arch.png "ROS Architecture"
[state]: ./docs/state.svg "\textbf{x}=\begin{bmatrix}x&y&z&\phi&\theta&\psi&\dot{x}&\dot{y}&\dot{z}&\dot{\phi}&\dot{\theta}&\dot{\psi}&\ddot{x}&\ddot{y}&\ddot{z}\end{bmatrix}"
[rotation_1]: ./docs/rotation_1.svg "R_z(\psi)R_y(\theta)R_x(\phi)s=\begin{bmatrix}cos(\psi)&-sin(\psi)&0\\sin(\psi)&cos(\psi)&0\\0&0&1\end{bmatrix}\begin{bmatrix}cos(\theta)&0&sin(\theta)\\0&1&0\\-sin(\theta)&0&cos(\theta)\end{bmatrix}\begin{bmatrix}1&0&0\\0&cos(\phi)&-sin(\phi)\\0&sin(\phi)&cos(\phi)\end{bmatrix}"
[rotation_2]: ./docs/rotation_2.svg "R_z(\psi)R_y(\theta)R_x(\phi)=\begin{bmatrix}cos(\theta)cos(\psi)&sin(\theta)sin(\phi)cos(\psi)-cos(\phi)sin(\psi)&sin(\theta)cos(\phi)cos(\psi)+sin(\phi)sin(\psi) \\cos(\theta)sin(\psi)&sin(\theta)sin(\phi)sin(\psi)+cos(\phi)cos(\psi)&sin(\theta)cos(\phi)sin(\psi)-sin(\phi)cos(\psi)\\-sin(\theta)&sin(\phi)cos(\theta)&cos(\phi)cos(\theta)\end{bmatrix}"
[predict_position]: ./docs/predict_position.svg "\begin{bmatrix}x\\y\\z\end{bmatrix}= \begin{bmatrix}x\\y\\z\end{bmatrix} + R_z(\psi)R_y(\theta)R_x(\phi)\begin{bmatrix}\dot{x}\\\dot{y}\\\dot{z}\end{bmatrix}\Delta{t}+0.5* R_z(\psi)R_y(\theta)R_x(\phi)\begin{bmatrix}\ddot{x}\\\ddot{y}\\\ddot{z}\end{bmatrix}\Delta{t}^2"
[predict_orientation]: ./docs/predict_orientation.svg "\begin{bmatrix}\phi\\\theta\\\psi\end{bmatrix}=\begin{bmatrix}\phi\\\theta\\\psi\end{bmatrix} + \begin{bmatrix}1&sin(\phi)tan(\theta)&cos(\phi)tan(\theta)\\0&cos(\phi)&-sin(\phi)\\0&\frac{sin(\phi)}{cos(\theta)}&\frac{cos(\phi)}{cos(\theta)}\end{bmatrix}\begin{bmatrix}\dot{\phi}\\\dot{\theta}\\\dot{\psi}\end{bmatrix}\Delta{t}"
[predict_velocity]: ./docs/predict_velocity.svg "\begin{bmatrix}\dot{x}\\\dot{y}\\\dot{z}\end{bmatrix}=\begin{bmatrix}\dot{x}\\\dot{y}\\\dot{z}\end{bmatrix} + \begin{bmatrix}\ddot{x}\\\ddot{y}\\\ddot{z}\end{bmatrix}\Delta{t}"

### Measurement Step

#### Measurement model (H)

A simple model is used where state is simply reduced to match measurement (aka drop state variables that 
are not part of the measurement). 

E.g. for depth sensor: the model simply returns `state z`. 

_note: for IMU the wrapping for orientation angles is handled as well_
