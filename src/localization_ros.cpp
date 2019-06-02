#include <au_localization/localization_ros.h>

LocalizationRos::LocalizationRos(const ros::NodeHandle &nh,
                                 const ros::NodeHandle &private_nh)
    : nh_(nh),
      private_nh_(private_nh),
      filter_(1e-4, 2, 0),
      baseLinkFrame_("base_link"),
      odomFrame_("odom"),
      tfListener_(tfBuffer_) {
  loadParams();
  filter_.setInitialCovariance(params_.initialCov);
  filter_.setProcessNoise(params_.processNoiseCov);
  filter_.reset();

  try {
    baseLinkFrame_ = au_core::load_frame("robot");
    // todo: remove temp output frame
    //    odomFrame_ = au_core::load_frame("odom");
    odomFrame_ = "temp_odom";
    odomPub_ = nh_.advertise<nav_msgs::Odometry>(
        au_core::load_topic("/topic/sensor/odom_state"), 20);
    imuSub_ = nh_.subscribe<sensor_msgs::Imu>(
        au_core::load_topic("/topic/sensor/imu_full"), 5,
        &LocalizationRos::imuCallback, this);
    dvlSub_ =
        nh_.subscribe<au_core::Dvl>(au_core::load_topic("/topic/sensor/dvl"), 5,
                                    &LocalizationRos::dvlCallback, this);
    depthSub_ = nh_.subscribe<au_core::Depth>(
        au_core::load_topic("/topic/sensor/depth"), 5,
        &LocalizationRos::depthCallback, this);
  } catch (std::exception &e) {
    ROS_ERROR("Unable to load topic/frame. Error: %s", e.what());
  }

  updateTimer_ = nh_.createTimer(ros::Duration(1. / params_.frequency),
                                 &LocalizationRos::update, this);
}

void LocalizationRos::reset() {
  clearMeasurementQueue();

  tfBuffer_.clear();

  // clear all waiting callbacks
  ros::getGlobalCallbackQueue()->clear();
}

void LocalizationRos::update(const ros::TimerEvent &event) {
  // warn user if update loop takes too long
  const double last_cycle_duration =
      (event.current_real - event.last_expected).toSec();
  if (last_cycle_duration > 2. / params_.frequency) {
    ROS_WARN_STREAM("Failed to meet update rate! Last cycle took "
                    << std::setprecision(20) << last_cycle_duration);
  }

  const ros::Time currentTime = ros::Time::now();
  if (!measurementQueue_.empty()) {
    while (!measurementQueue_.empty() && ros::ok()) {
      MeasurementPtr z = measurementQueue_.top();
      // if measurement's time is later than now, wait until next iteration
      if (z->time > currentTime.toSec()) {
        break;
      }
      measurementQueue_.pop();
      // predict + update loop with measurement
      filter_.processMeasurement(*(z.get()));
    }
  } else if (filter_.isInitialized()) {  // only predict if initialized
    // no measurement call filter predict
    double deltaT = currentTime.toSec() - filter_.getLastMeasurementTime();
    if (deltaT > 100000.0) {
      ROS_WARN(
          "Delta was very large. Suspect playing from bag file. Setting to "
          "0.01");
      deltaT = 0.01;
    }
    filter_.predict(deltaT);
    ROS_WARN_THROTTLE(1.0, "No measurements recieved. Using prediction only.");
  }

  // publish message and frame transform
  if (filter_.isInitialized()) {
    nav_msgs::Odometry filteredState = getFilteredOdomMessage();
    // broadcast odom frame
    geometry_msgs::TransformStamped odomTransMsg;
    odomTransMsg.header.stamp = filteredState.header.stamp;
    odomTransMsg.header.frame_id = filteredState.header.frame_id;
    odomTransMsg.child_frame_id = filteredState.child_frame_id;
    odomTransMsg.transform.translation.x = filteredState.pose.pose.position.x;
    odomTransMsg.transform.translation.y = filteredState.pose.pose.position.y;
    odomTransMsg.transform.translation.z = filteredState.pose.pose.position.z;
    odomTransMsg.transform.rotation = filteredState.pose.pose.orientation;
    odomBroadcaster_.sendTransform(odomTransMsg);
    // publish odom message
    odomPub_.publish(filteredState);
  }
}

nav_msgs::Odometry LocalizationRos::getFilteredOdomMessage() {
  // should only be called if filter is initialized
  assert(filter_.isInitialized());

  const Eigen::VectorXd &state = filter_.getState();
  const Eigen::MatrixXd &cov = filter_.getCovariance();

  tf2::Quaternion quat;
  quat.setRPY(state(StateRoll), state(StatePitch), state(StateYaw));

  nav_msgs::Odometry odom;
  odom.pose.pose.position.x = state(StateX);
  odom.pose.pose.position.y = state(StateY);
  odom.pose.pose.position.z = state(StateZ);
  odom.pose.pose.orientation.x = quat.x();
  odom.pose.pose.orientation.y = quat.y();
  odom.pose.pose.orientation.z = quat.z();
  odom.pose.pose.orientation.w = quat.w();
  odom.twist.twist.linear.x = state(StateVx);
  odom.twist.twist.linear.y = state(StateVy);
  odom.twist.twist.linear.z = state(StateVz);
  odom.twist.twist.angular.x = state(StateVroll);
  odom.twist.twist.angular.y = state(StateVpitch);
  odom.twist.twist.angular.z = state(StateVyaw);

  for (size_t i = 0; i < 6; ++i) {
    for (size_t j = 0; j < 6; ++j) {
      odom.pose.covariance[6 * i + j] = cov(i, j);
      odom.twist.covariance[6 * i + j] = cov(i + StateVx, j + StateVx);
    }
  }

  odom.header.stamp = ros::Time(filter_.getLastMeasurementTime());
  odom.header.frame_id = odomFrame_;
  odom.child_frame_id = baseLinkFrame_;
  return odom;
}

void LocalizationRos::imuCallback(const sensor_msgs::ImuConstPtr &msg) {
  MeasurementPtr z = std::make_shared<Measurement>(IMU_SIZE);
  z->type = MeasurementTypeImu;
  z->time = msg->header.stamp.toSec();
  z->covariance.setZero();
  tf2::Transform targetFrameTrans = getTransformFrame(msg->header);

  // orientation
  // note: IMU should be mounted such that RPY is in NED coord frame
  tf2::Quaternion q;
  tf2::fromMsg(msg->orientation, q);
  tf2::Matrix3x3 orientation(q);
  double roll, pitch, yaw;
  orientation.getRPY(roll, pitch, yaw);
  z->measurement(ImuRoll) = roll;
  z->measurement(ImuPitch) = pitch;
  z->measurement(ImuYaw) = yaw;
  z->covariance.block<3, 3>(ImuRoll, ImuRoll) =
      Eigen::Vector3d(msg->orientation_covariance[0],
                      msg->orientation_covariance[4],
                      msg->orientation_covariance[8])
          .asDiagonal();

  // angular velocity
  tf2::Vector3 angularVelocity(msg->angular_velocity.x, msg->angular_velocity.y,
                               msg->angular_velocity.z);
  angularVelocity = targetFrameTrans.getBasis() * angularVelocity;
  z->measurement(ImuVroll) = angularVelocity.x();
  z->measurement(ImuVpitch) = angularVelocity.y();
  z->measurement(ImuVyaw) = angularVelocity.z();
  // rotate covariance matrix to base_link
  Eigen::MatrixXd covarianceRotated(3, 3);
  rotateCovariance(&(msg->angular_velocity_covariance[0]),
                   targetFrameTrans.getRotation(), covarianceRotated);
  z->covariance.block<3, 3>(ImuVroll, ImuVroll) = covarianceRotated;

  // linear acceleration
  tf2::Vector3 linearAcceleration(msg->linear_acceleration.x,
                                  msg->linear_acceleration.y,
                                  msg->linear_acceleration.z);
  // note: we assume that if the sensor is placed at some non-zero offset from
  // the vehicle's center, the vehicle turns with constant velocity. This is
  // because we do not have angular acceleration
  linearAcceleration = targetFrameTrans.getBasis() * linearAcceleration;
  z->measurement(ImuAx) = linearAcceleration.x();
  z->measurement(ImuAy) = linearAcceleration.y();
  z->measurement(ImuAz) = linearAcceleration.z();
  // rotate covariance matrix to base_link
  rotateCovariance(&(msg->linear_acceleration_covariance[0]),
                   targetFrameTrans.getRotation(), covarianceRotated);
  z->covariance.block<3, 3>(ImuAx, ImuAx) = covarianceRotated;

  measurementQueue_.push(z);
}

void LocalizationRos::dvlCallback(const au_core::DvlConstPtr &msg) {
  MeasurementPtr z = std::make_shared<Measurement>(DVL_SIZE);
  z->type = MeasurementTypeDvl;
  z->time = msg->header.stamp.toSec();
  tf2::Transform targetFrameTrans = getTransformFrame(msg->header);

  tf2::Vector3 linVel(msg->velocity.x, msg->velocity.y, msg->velocity.z);
  linVel = targetFrameTrans.getBasis() * linVel;
  // account for linear velocity as a result of sensor offset and
  // rotational velocity
  const Eigen::VectorXd &state = filter_.getState();
  tf2::Vector3 angVel(state(StateVroll), state(StateVpitch), state(StateVyaw));
  linVel += targetFrameTrans.getOrigin().cross(angVel);
  z->measurement(0) = linVel.x();
  z->measurement(1) = linVel.y();
  z->measurement(2) = linVel.z();
  // rotate covariance matrix to base_link
  Eigen::MatrixXd covarianceRotated(3, 3);
  rotateCovariance(&(msg->velocity_covariance[0]),
                   targetFrameTrans.getRotation(), covarianceRotated);
  z->covariance.block<3, 3>(0, 0) = covarianceRotated;

  measurementQueue_.push(z);
}

void LocalizationRos::depthCallback(const au_core::DepthConstPtr &msg) {
  MeasurementPtr z = std::make_shared<Measurement>(DEPTH_SIZE);
  z->type = MeasurementTypeDepth;
  z->time = msg->header.stamp.toSec();
  tf2::Transform targetFrameTrans = getTransformFrame(msg->header);
  // take into account positional offset of depth sensor
  z->measurement(0) = msg->depth - targetFrameTrans.getOrigin().z();
  z->covariance(0, 0) = msg->depth_variance;
  measurementQueue_.push(z);
}

void LocalizationRos::loadParams() {
  private_nh_.param("frequency", params_.frequency, 40.0);

  loadMatrixFromParams(params_.initialCov, "initial_estimate_covariance");
  loadMatrixFromParams(params_.processNoiseCov, "process_noise_covariance");
}

void LocalizationRos::loadMatrixFromParams(Eigen::MatrixXd &mat,
                                           const std::string &key) {
  size_t size = mat.rows();
  mat.setZero();
  XmlRpc::XmlRpcValue param;

  try {
    private_nh_.getParam(key, param);
    for (size_t i = 0; i < size; i++) {
      for (size_t j = 0; j < size; j++) {
        // needed if all points don't have decimal points
        std::ostringstream os;
        os << param[size * i + j];
        std::istringstream is(os.str());
        is >> mat(i, j);
      }
    }
  } catch (...) {
    ROS_ERROR("Error loading %s param", "initial_estimate_covariance");
  }
}

tf2::Transform LocalizationRos::getTransformFrame(
    const std_msgs::Header &header) {
  tf2::Transform targetFrameTrans;
  try {
    tf2::fromMsg(tfBuffer_
                     .lookupTransform(baseLinkFrame_, header.frame_id,
                                      header.stamp, ros::Duration(0.01))
                     .transform,
                 targetFrameTrans);
  } catch (tf2::TransformException &ex) {
    ROS_WARN_STREAM_THROTTLE(2.0, "Could not obtain transform from "
                                      << header.frame_id << " to "
                                      << baseLinkFrame_
                                      << ". Error: " << ex.what());
  }
  return targetFrameTrans;
}

void LocalizationRos::clearMeasurementQueue() {
  while (!measurementQueue_.empty() && ros::ok()) {
    measurementQueue_.pop();
  }
}

void LocalizationRos::rotateCovariance(const double *covariance,
                                       const tf2::Quaternion &q,
                                       Eigen::MatrixXd &rotated) {
  // create Eigen matrix with rotation q
  tf2::Matrix3x3 tfRot(q);
  Eigen::MatrixXd rot(3, 3);
  for (size_t i = 0; i < 3; ++i) {
    rot(i, 0) = tfRot.getRow(i).getX();
    rot(i, 1) = tfRot.getRow(i).getY();
    rot(i, 2) = tfRot.getRow(i).getZ();
  }
  // copy covariance to rotated
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      rotated(i, j) = covariance[3 * i + j];
    }
  }
  rotated = rot * rotated.eval() * rot.transpose();
}