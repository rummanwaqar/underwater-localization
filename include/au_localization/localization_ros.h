#pragma once

#include <eigen3/Eigen/Dense>
#include <memory>
#include <queue>

#include <au_core/loader_util.h>

#include <ros/ros.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <au_core/Depth.h>
#include <au_core/Dvl.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>

#include <au_localization/localization.h>

using MeasurementQueue =
    std::priority_queue<MeasurementPtr, std::vector<MeasurementPtr>,
                        Measurement>;

/*
 * ROS wrapper for Localization class.
 * subscribes to all sensor inputs
 * publishes filtered output message
 */
class LocalizationRos {
 public:
  // constructor
  explicit LocalizationRos(const ros::NodeHandle& nh,
                           const ros::NodeHandle& private_nh);

  // destructor - closes all subscriber connections and clears the message
  // filters
  ~LocalizationRos() = default;

  // resets filter to initial state
  void reset();

 protected:
  // periodically called by ros::Timer
  void update(const ros::TimerEvent& event);

  // callback method for receiving IMU messages
  void imuCallback(const sensor_msgs::ImuConstPtr& msg);

  // callback method for receiving DVL messages
  void dvlCallback(const au_core::DvlConstPtr& msg);

  // callback method for recieving depth messages
  void depthCallback(const au_core::DepthConstPtr& msg);

  // returns the ekf output as ros message for publishing
  nav_msgs::Odometry getFilteredOdomMessage();

  // gets transform frame from message from to baselink
  tf2::Transform getTransformFrame(const std_msgs::Header& header);

  // loads ukf params from file
  void loadParams();

  // loads square eigen matrix from params
  void loadMatrixFromParams(Eigen::MatrixXd& mat, const std::string& key);

  // clears measurement queue
  void clearMeasurementQueue();

  // rotates covariance array using quaternion q
  // result copies to Eigen::Matrix rotated
  void rotateCovariance(const double* covariance, const tf2::Quaternion& q,
                        Eigen::MatrixXd& rotated);

  // ros node handles
  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;

  struct Params {
    // filter update rate
    double frequency;
    // ukf sigma parameters
    Eigen::MatrixXd initialCov;
    Eigen::MatrixXd processNoiseCov;

    Params()
        : initialCov(STATE_SIZE, STATE_SIZE),
          processNoiseCov(STATE_SIZE, STATE_SIZE) {}
  } params_;

  // localization ukf filter
  Localization filter_;

  ros::Publisher odomPub_;
  ros::Timer updateTimer_;
  // subscribers for sensors
  ros::Subscriber imuSub_;
  ros::Subscriber dvlSub_;
  ros::Subscriber depthSub_;

  // frames ids
  std::string baseLinkFrame_;
  std::string odomFrame_;

  // tf buffer for managing coord frames
  tf2_ros::Buffer tfBuffer_;
  tf2_ros::TransformListener tfListener_;
  tf2_ros::TransformBroadcaster odomBroadcaster_;

  // queue measurements until update method is trigger
  MeasurementQueue measurementQueue_;
};