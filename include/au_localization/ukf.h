#pragma once
#include <au_localization/ukf_helpers.h>

/*
 * UKF implementation based on paper
 * https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf
 *
 * @param N: number of states
 */
template <int N>
class Ukf {
 public:
  /*
   * initializes the ukf class
   * @param alpha: spread of sigma points (1 < alpha < 1e-4)
   * @param beta: prior knowledge of distribution (Gaussian = 2)
   * @param kappa: secondary scaling parameter
   * @param F: state transition function
   */
  Ukf(double alpha, double beta, double kappa,
      std::function<Eigen::Matrix<double, N, 1>(
          const Eigen::Matrix<double, N, 1>& state, double deltaT)>
          F)
      : pointsFn_(alpha, beta, kappa), stateTransitionFunction_(F) {
    x_.setZero();
    P_.setIdentity();
    processNoise_.setIdentity();
  }

  /*
   * perform filter prediction step using system model
   * @param deltaT: time between k and k+1 in seconds
   */
  void predict(double deltaT) {
    // 1. calculate sigma points from current state
    sigmaPrediction_ = pointsFn_.generateSigmaPoints(x_, P_);

    // 2. predict sigma points for next state
    for (int i = 0; i < pointsFn_.getNumSigmaPoints(); i++) {
      sigmaPrediction_.col(i) =
          stateTransitionFunction_(sigmaPrediction_.col(i), deltaT);
    }

    // 3. get predicted state and covariance from sigma points
    std::tie(x_, P_) = unscented_transform<N, 2 * N + 1>(
        sigmaPrediction_, pointsFn_.getWm(), pointsFn_.getWc(), processNoise_,
        diffFn_, meanFn_);
  }

  /*
   * perform update step using measurement
   * @param M: number of measurement variables
   * @param measurement: measurement vector
   * @param covariance: measurement covariance
   * @param model: measurement model
   */
  template <int M>
  void update(Eigen::Matrix<double, M, 1>& measurement,
              Eigen::Matrix<double, M, M>& covariance,
              MeasurementModel<M, N> model) {
    // 1. pass prediction sigmas through measurement function
    Eigen::Matrix<double, M, N * 2 + 1> sigmaMeasurement;
    for (int i = 0; i < pointsFn_.getNumSigmaPoints(); i++) {
      sigmaMeasurement.col(i) = model.measurementFn(sigmaPrediction_.col(i));
    }

    // 2. prediction measurement (Hx) and measurement covariance
    Eigen::Matrix<double, M, 1> predictedMeasurement;
    Eigen::Matrix<double, M, M> P_zz;
    std::tie(predictedMeasurement, P_zz) = unscented_transform<M, 2 * N + 1>(
        sigmaMeasurement, pointsFn_.getWm(), pointsFn_.getWc(), covariance,
        model.diffFn, model.meanFn);

    // 3. compute cross variancec of state and measurements
    Eigen::Matrix<double, N, M> P_xz;
    P_xz.setZero();
    Eigen::Matrix<double, N, 1> xDiff;
    Eigen::Matrix<double, M, 1> zDiff;
    for (int i = 0; i < pointsFn_.getNumSigmaPoints(); i++) {
      if (diffFn_) {
        xDiff = diffFn_(sigmaPrediction_.col(i), x_);
      } else {
        xDiff = sigmaPrediction_.col(i) - x_;
      }
      if (model.diffFn) {
        zDiff = model.diffFn(sigmaMeasurement.col(i), predictedMeasurement);
      } else {
        zDiff = sigmaMeasurement.col(i) - predictedMeasurement;
      }
      P_xz.noalias() += pointsFn_.getWc()(i) * xDiff * zDiff.transpose();
    }

    // 4. calculate Kalman gain
    Eigen::Matrix<double, N, M> K = P_xz * P_zz.inverse();

    // 5. update state
    x_.noalias() += K * (measurement - predictedMeasurement);
    P_.noalias() -= K * P_zz * K.transpose();
  }

  /*
   * get estimated state
   */
  Eigen::Matrix<double, N, 1>& getState() { return x_; }

  /*
   * set state (should only be used for initialization)
   */
  void setState(const Eigen::Matrix<double, N, 1>& state) { x_ = state; }

  /*
   * get estimated covariance
   */
  Eigen::Matrix<double, N, N>& getCovariance() { return P_; }

  /*
   * set covariance (should only be used for initialization)
   */
  void setCovariance(const Eigen::Matrix<double, N, N>& cov) { P_ = cov; }

  /*
   * set process noise covariance
   */
  void setProcessNoise(const Eigen::Matrix<double, N, N>& noise) {
    processNoise_ = noise;
  }

  /*
   * set custom difference function for process model
   */
  void setDifferenceFunction(std::function<Eigen::Matrix<double, N, 1>(
                                 const Eigen::Matrix<double, N, 1>& a,
                                 const Eigen::Matrix<double, N, 1>& b)>
                                 func) {
    diffFn_ = func;
    pointsFn_.setDifferenceFunction(func);
  }

  /*
   * set custom mean function for process model
   */
  void setMeanFunction(
      std::function<Eigen::Matrix<double, N, 1>(
          const Eigen::Matrix<double, N, N * 2 + 1>& sigmaPoints,
          const Eigen::Matrix<double, N * 2 + 1, 1>& weight)>
          func) {
    meanFn_ = func;
  }

 protected:
  // sigma points class
  MerweScaledSigmaPoints<N> pointsFn_;

  // estimated state
  Eigen::Matrix<double, N, 1> x_;
  Eigen::Matrix<double, N, N> P_;
  // estimated prediction sigma points
  Eigen::Matrix<double, N, N * 2 + 1> sigmaPrediction_;
  // process noise covariance
  Eigen::Matrix<double, N, N> processNoise_;

  /*
   * state transition function
   */
  std::function<Eigen::Matrix<double, N, 1>(
      const Eigen::Matrix<double, N, 1>& state, double deltaT)>
      stateTransitionFunction_;

  /*
   * custom difference function
   * used to handle circular values like angles
   */
  std::function<Eigen::Matrix<double, N, 1>(
      const Eigen::Matrix<double, N, 1>& a,
      const Eigen::Matrix<double, N, 1>& b)>
      diffFn_;

  /*
   * custom mean function
   * used to handle circular values like angles
   */
  std::function<Eigen::Matrix<double, N, 1>(
      const Eigen::Matrix<double, N, N * 2 + 1>& sigmaPoints,
      const Eigen::Matrix<double, N * 2 + 1, 1>& weight)>
      meanFn_;
};