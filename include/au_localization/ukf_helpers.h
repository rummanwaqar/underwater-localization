#pragma once

#include <eigen3/Eigen/Dense>
#include <functional>
#include <tuple>

/*
 * generate sigma points and weights using Van der Merwe's method.
 * R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic Inference in
 * Dynamic State-Space Models"
 * @param N: state size
 */
template <int N>
class MerweScaledSigmaPoints {
 public:
  /*
   * initializes the weights
   * @parma alpha: controls spread of sigma points 1 < alpha < 1e-4
   * @param beta: prior knowledge of distribution (Gaussian = 2)
   * @param kappa: secondary scaling param (3 - N or 0)
   */
  MerweScaledSigmaPoints(double alpha, double beta, double kappa)
      : alpha_(alpha), beta_(beta), kappa_(kappa), lambda_(0) {
    compute_weights();
  }

  /*
   * generates sigma points for given mean and covariance
   * @param x: mean
   * @param P: covariance
   * @return sigma points
   */
  Eigen::Matrix<double, N, N * 2 + 1> generateSigmaPoints(
      const Eigen::Matrix<double, N, 1>& x,
      const Eigen::Matrix<double, N, N>& P) {
    Eigen::Matrix<double, N, N * 2 + 1> sigmaPoints;

    // square root of covariance matrix using LL decomposition
    Eigen::Matrix<double, N, N> S = ((N + lambda_) * P).llt().matrixL();

    sigmaPoints.col(0) = x;
    for (int i = 0; i < N; i++) {
      if (!subtract_) {
        sigmaPoints.col(i + 1) = x + S.col(i);
        sigmaPoints.col(N + i + 1) = x - S.col(i);
      } else {
        sigmaPoints.col(i + 1) = subtract_(x, -S.col(i));
        sigmaPoints.col(N + i + 1) = subtract_(x, S.col(i));
      }
    }

    return sigmaPoints;
  }

  /*
   * get number of sigma points
   */
  int getNumSigmaPoints() { return N * 2 + 1; }

  /*
   * get reference to mean weights
   */
  const Eigen::Matrix<double, N * 2 + 1, 1>& getWm() { return weightMean; }

  /*
   * get reference to covariance weights
   */
  const Eigen::Matrix<double, N * 2 + 1, 1>& getWc() { return weightCov; }

  /*
   * set custom difference function
   * used to handle difference for circular values such as angles
   */
  void setDifferenceFunction(std::function<Eigen::Matrix<double, N, 1>(
                                 const Eigen::Matrix<double, N, 1>& a,
                                 const Eigen::Matrix<double, N, 1>& b)>
                                 func) {
    subtract_ = func;
  }

 protected:
  /*
   * computes weights for the scaled unscented KF
   */
  void compute_weights() {
    lambda_ = alpha_ * alpha_ * (N + kappa_) - N;

    weightMean[0] = lambda_ / (N + lambda_);
    weightCov[0] = lambda_ / (N + lambda_) + (1 - alpha_ * alpha_ + beta_);

    double w_i = 1. / (2 * (N + lambda_));
    assert(w_i > 0);  // to avoid square root of negative number

    for (int i = 1; i < N * 2 + 1; ++i) {
      weightMean[i] = w_i;
      weightCov[i] = w_i;
    }
  }

  // sigma distribution parameters
  const double alpha_;
  const double beta_;
  const double kappa_;
  double lambda_;

  // sigma weights
  Eigen::Matrix<double, N * 2 + 1, 1> weightMean;
  Eigen::Matrix<double, N * 2 + 1, 1> weightCov;

  // custom difference function
  std::function<Eigen::Matrix<double, N, 1>(
      const Eigen::Matrix<double, N, 1>& a,
      const Eigen::Matrix<double, N, 1>& b)>
      subtract_;
};

/*
 * measurement model
 * @param M: no of measurement variables
 * @param N: no of state variables
 */
template <int M, int N>
struct MeasurementModel {
  // measurement function
  std::function<Eigen::Matrix<double, M, 1>(
      const Eigen::Matrix<double, N, 1>& state)>
      measurementFn;

  // custom difference function
  std::function<Eigen::Matrix<double, M, 1>(
      const Eigen::Matrix<double, M, 1>& a,
      const Eigen::Matrix<double, M, 1>& b)>
      diffFn = {};

  // custom mean function
  std::function<Eigen::Matrix<double, M, 1>(
      const Eigen::Matrix<double, M, N * 2 + 1>& sigmaPoints,
      const Eigen::Matrix<double, N * 2 + 1, 1>& weight)>
      meanFn = {};
};

/*
 * performs unscented transform
 * @param sigmaPoints
 * @param weightsMean: mean weights
 * @param weightsCov: covariance weights
 * @param noise: noise covariance
 * @param diffFn: custom difference function (optional)
 * @param meanFn: custom mean function (optional)
 */
template <int N, int NSigma>
std::tuple<Eigen::Matrix<double, N, 1>, Eigen::Matrix<double, N, N>>
unscented_transform(const Eigen::Matrix<double, N, NSigma>& sigmaPoints,
                    const Eigen::Matrix<double, NSigma, 1>& weightsMean,
                    const Eigen::Matrix<double, NSigma, 1>& weightsCov,
                    const Eigen::Matrix<double, N, N>& noise,
                    std::function<Eigen::Matrix<double, N, 1>(
                        const Eigen::Matrix<double, N, 1>& a,
                        const Eigen::Matrix<double, N, 1>& b)>
                        diffFn = {},
                    std::function<Eigen::Matrix<double, N, 1>(
                        const Eigen::Matrix<double, N, NSigma>& sigmaPoints,
                        const Eigen::Matrix<double, NSigma, 1>& weights)>
                        meanFn = {}) {
  Eigen::Matrix<double, N, 1> x;
  if (!meanFn) {
    x = sigmaPoints * weightsMean;
  } else {
    x = meanFn(sigmaPoints, weightsMean);
  }

  Eigen::Matrix<double, N, N> P;
  P.setZero();
  Eigen::Matrix<double, N, 1> diff;
  for (int i = 0; i < NSigma; ++i) {
    if (!diffFn) {
      diff = sigmaPoints.col(i) - x;
    } else {
      diff = diffFn(sigmaPoints.col(i), x);
    }
    P.noalias() += weightsCov(i) * (diff * diff.transpose());
  }
  P = P + noise;

  return std::make_tuple(x, P);
}