#pragma once

#include <eigen3/Eigen/Dense>
#include <memory>

// indexes for state and cov matrices
enum State {
  StateX = 0,
  StateY,
  StateZ,
  StateRoll,
  StatePitch,
  StateYaw,
  StateVx,
  StateVy,
  StateVz,
  StateVroll,
  StateVpitch,
  StateVyaw,
  StateAx,
  StateAy,
  StateAz
};
const int STATE_SIZE = 15;

const int DEPTH_SIZE = 1;
const int DVL_SIZE = 3;

// indexes for imu variables
enum Imu {
  ImuRoll = 0,
  ImuPitch,
  ImuYaw,
  ImuVroll,
  ImuVpitch,
  ImuVyaw,
  ImuAx,
  ImuAy,
  ImuAz
};
const int IMU_SIZE = 9;

// types of measurements
enum MeasurementType {
  MeasurementTypeNone = 0,
  MeasurementTypeDepth,
  MeasurementTypeDvl,
  MeasurementTypeImu
};

/*
 * used to pass measurements between ros node
 * and Localization class
 */
struct Measurement {
  // type of measurement
  MeasurementType type;
  // measurement data with covariance
  Eigen::VectorXd measurement;
  Eigen::MatrixXd covariance;
  // time of measurement in seconds
  double time;
  // mahalanobis distance threshold in number of sigmas
  double mahalanobisThreshold;

  Measurement()
      : type(MeasurementTypeNone),
        time(0.0),
        mahalanobisThreshold(std::numeric_limits<double>::max()) {}

  Measurement(size_t size) : measurement(size), covariance(size, size) {
    Measurement();
  }

  // earlier time will have higher priority in queue
  bool operator()(const std::shared_ptr<Measurement>& a,
                  const std::shared_ptr<Measurement>& b) {
    return a->time > b->time;
  }
};
typedef std::shared_ptr<Measurement> MeasurementPtr;
