#include <au_localization/ukf.h>
#include <gtest/gtest.h>

TEST(UkfTests, prediction) {
  // state = {x y vx vy}
  // state transition function
  // 2d motion with constant velocity
  auto F = [](const Eigen::Matrix<double, 4, 1>& state,
              const double deltaT) -> Eigen::Matrix<double, 4, 1> {
    Eigen::Matrix<double, 4, 1> newState;
    newState(0) = state(0) + state(2) * deltaT;
    newState(1) = state(1) + state(3) * deltaT;
    newState(2) = state(2);
    newState(3) = state(3);
    return newState;
  };
  Eigen::Matrix<double, 4, 1> x;
  x << 0.0, 0.0, 0.3, 0.3;
  Eigen::Matrix<double, 4, 4> P;
  P.setIdentity();
  P *= 1e-4;
  Eigen::Matrix<double, 4, 4> Q;
  Q << 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0.01;

  Ukf<4> ukf(0.1, 2, 0, F);
  ukf.setState(x);
  ukf.setCovariance(P);
  ukf.setProcessNoise(Q);

  ukf.predict(0.1);
  ukf.predict(0.1);
  ukf.predict(0.1);

  EXPECT_FLOAT_EQ(0.09, ukf.getState()(0));
  EXPECT_FLOAT_EQ(0.09, ukf.getState()(1));
  EXPECT_FLOAT_EQ(0.3, ukf.getState()(2));
  EXPECT_FLOAT_EQ(0.3, ukf.getState()(3));

  EXPECT_FLOAT_EQ(0.300609, ukf.getCovariance().diagonal()(0));
  EXPECT_FLOAT_EQ(0.300609, ukf.getCovariance().diagonal()(1));
  EXPECT_FLOAT_EQ(0.0301, ukf.getCovariance().diagonal()(2));
  EXPECT_FLOAT_EQ(0.0301, ukf.getCovariance().diagonal()(3));
}

TEST(UkfTests, predictionAngular) {
  // state = {x y yaw}
  // state transition function
  // 2d motion with constant velocity and angle
  auto F = [](const Eigen::Matrix<double, 3, 1>& state,
              const double deltaT) -> Eigen::Matrix<double, 3, 1> {
    Eigen::Matrix<double, 3, 1> newState;
    double vel = 1;
    newState(0) = state(0) + cos(state(2)) * deltaT * vel;
    newState(1) = state(1) + sin(state(2)) * deltaT * vel;
    newState(2) = state(2);
    return newState;
  };

  auto diffFn =
      [](const Eigen::Matrix<double, 3, 1>& a,
         const Eigen::Matrix<double, 3, 1>& b) -> Eigen::Matrix<double, 3, 1> {
    Eigen::Matrix<double, 3, 1> diff = a;
    diff(0) -= b(0);
    diff(1) -= b(1);
    diff(2) -= b(2);
    while (diff(2) > M_PI) diff(2) -= M_PI * 2;
    while (diff(2) < -M_PI) diff(2) += M_PI * 2;
    return diff;
  };

  auto meanFn = [](const Eigen::Matrix<double, 3, 7>& sigmaPoints,
                   const Eigen::Matrix<double, 7, 1>& weights)
      -> Eigen::Matrix<double, 3, 1> {
    Eigen::Matrix<double, 3, 1> mean;
    mean.setZero();
    double sum_sin = 0, sum_cos = 0;
    for (int i = 0; i < sigmaPoints.cols(); i++) {
      mean(0) += sigmaPoints(0, i) * weights(i);
      mean(1) += sigmaPoints(1, i) * weights(i);
      sum_sin += sin(sigmaPoints(2, i)) * weights(i);
      sum_cos += cos(sigmaPoints(2, i)) * weights(i);
    }
    mean(2) = atan2(sum_sin, sum_cos);
    return mean;
  };

  Eigen::Matrix<double, 3, 1> x;
  x << 0.0, 0.0, M_PI * 2 + 0.3;
  Eigen::Matrix<double, 3, 3> P;
  P.setIdentity();
  P *= 1e-4;
  Eigen::Matrix<double, 3, 3> Q;
  Q << 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.01;

  Ukf<3> ukf(0.1, 2, 0, F);
  ukf.setState(x);
  ukf.setCovariance(P);
  ukf.setProcessNoise(Q);
  ukf.setDifferenceFunction(diffFn);
  ukf.setMeanFunction(meanFn);

  ukf.predict(0.1);
  ukf.predict(0.1);
  ukf.predict(0.1);

  EXPECT_NEAR(0.2, ukf.getState()(0), 0.1);
  EXPECT_NEAR(0.1, ukf.getState()(1), 0.1);
  EXPECT_FLOAT_EQ(0.3, ukf.getState()(2));

  EXPECT_NEAR(0.300, ukf.getCovariance().diagonal()(0), 0.001);
  EXPECT_NEAR(0.300, ukf.getCovariance().diagonal()(1), 0.001);
  EXPECT_FLOAT_EQ(0.0301, ukf.getCovariance().diagonal()(2));
}

TEST(UkfTests, update) {
  // state = {x y vx vy}
  // state transition function
  // 2d motion with constant velocity
  auto F = [](const Eigen::Matrix<double, 4, 1>& state,
              const double deltaT) -> Eigen::Matrix<double, 4, 1> {
    Eigen::Matrix<double, 4, 1> newState;
    newState(0) = state(0) + state(2) * deltaT;
    newState(1) = state(1) + state(3) * deltaT;
    newState(2) = state(2);
    newState(3) = state(3);
    return newState;
  };
  Eigen::Matrix<double, 4, 1> x;
  x << 0.0, 0.0, 0.3, 0.3;
  Eigen::Matrix<double, 4, 4> P;
  P.setIdentity();
  P *= 1e-4;
  Eigen::Matrix<double, 4, 4> Q;
  Q << 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0.01;

  Ukf<4> ukf(0.1, 2, 0, F);
  ukf.setState(x);
  ukf.setCovariance(P);
  ukf.setProcessNoise(Q);

  ukf.predict(0.1);

  // velocity sensor
  MeasurementModel<2, 4> measurementModel;
  measurementModel.measurementFn = [](const Eigen::Matrix<double, 4, 1>& state)
      -> Eigen::Matrix<double, 2, 1> {
    Eigen::Matrix<double, 2, 1> measurement;
    measurement(0) = state(2);
    measurement(1) = state(3);
    return measurement;
  };

  Eigen::Matrix<double, 2, 1> z;
  z << 0.6, 0.4;
  Eigen::Matrix<double, 2, 2> R;
  R << 1e-6, 0, 0, 1e-6;

  ukf.update<2>(z, R, measurementModel);

  EXPECT_NEAR(0.06, ukf.getState()(0), 0.01);
  EXPECT_NEAR(0.04, ukf.getState()(1), 0.01);
  EXPECT_NEAR(0.6, ukf.getState()(2), 0.1);
  EXPECT_NEAR(0.4, ukf.getState()(3), 0.1);

  EXPECT_NEAR(0.1001, ukf.getCovariance().diagonal()(0), 0.0001);
  EXPECT_NEAR(0.1001, ukf.getCovariance().diagonal()(1), 0.0001);
  EXPECT_NEAR(0.01001, ukf.getCovariance().diagonal()(2), 0.0001);
  EXPECT_NEAR(0.01001, ukf.getCovariance().diagonal()(3), 0.0001);
}

TEST(UkfTests, updateAngular) {
  // state = {x y yaw}
  // state transition function
  // 2d motion with constant velocity and angle
  auto F = [](const Eigen::Matrix<double, 3, 1>& state,
              const double deltaT) -> Eigen::Matrix<double, 3, 1> {
    Eigen::Matrix<double, 3, 1> newState;
    double vel = 1;
    newState(0) = state(0) + cos(state(2)) * deltaT * vel;
    newState(1) = state(1) + sin(state(2)) * deltaT * vel;
    newState(2) = state(2);
    return newState;
  };

  auto diffFn =
      [](const Eigen::Matrix<double, 3, 1>& a,
         const Eigen::Matrix<double, 3, 1>& b) -> Eigen::Matrix<double, 3, 1> {
    Eigen::Matrix<double, 3, 1> diff = a;
    diff(0) -= b(0);
    diff(1) -= b(1);
    diff(2) -= b(2);
    while (diff(2) > M_PI) diff(2) -= M_PI * 2;
    while (diff(2) < -M_PI) diff(2) += M_PI * 2;
    return diff;
  };

  auto meanFn = [](const Eigen::Matrix<double, 3, 7>& sigmaPoints,
                   const Eigen::Matrix<double, 7, 1>& weights)
      -> Eigen::Matrix<double, 3, 1> {
    Eigen::Matrix<double, 3, 1> mean;
    mean.setZero();
    double sum_sin = 0, sum_cos = 0;
    for (int i = 0; i < sigmaPoints.cols(); i++) {
      mean(0) += sigmaPoints(0, i) * weights(i);
      mean(1) += sigmaPoints(1, i) * weights(i);
      sum_sin += sin(sigmaPoints(2, i)) * weights(i);
      sum_cos += cos(sigmaPoints(2, i)) * weights(i);
    }
    mean(2) = atan2(sum_sin, sum_cos);
    return mean;
  };

  Eigen::Matrix<double, 3, 1> x;
  x << 0.0, 0.0, 0.3;
  Eigen::Matrix<double, 3, 3> P;
  P.setIdentity();
  P *= 1e-4;
  Eigen::Matrix<double, 3, 3> Q;
  Q << 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.01;

  Ukf<3> ukf(0.1, 2, 0, F);
  ukf.setState(x);
  ukf.setCovariance(P);
  ukf.setProcessNoise(Q);
  ukf.setDifferenceFunction(diffFn);
  ukf.setMeanFunction(meanFn);

  ukf.predict(0.1);

  // yaw sensor
  MeasurementModel<1, 3> measurementModel;
  measurementModel.measurementFn = [](const Eigen::Matrix<double, 3, 1>& state)
      -> Eigen::Matrix<double, 1, 1> {
    Eigen::Matrix<double, 1, 1> measurement;
    measurement(0) = state(2);
    return measurement;
  };
  measurementModel.diffFn =
      [](const Eigen::Matrix<double, 1, 1>& a,
         const Eigen::Matrix<double, 1, 1>& b) -> Eigen::Matrix<double, 1, 1> {
    Eigen::Matrix<double, 1, 1> diff = a;
    diff(0) -= b(0);
    while (diff(0) > M_PI) diff(0) -= M_PI * 2;
    while (diff(0) < -M_PI) diff(0) += M_PI * 2;
    return diff;
  };
  measurementModel.meanFn = [](const Eigen::Matrix<double, 1, 7>& sigmaPoints,
                               const Eigen::Matrix<double, 7, 1>& weights)
      -> Eigen::Matrix<double, 1, 1> {
    Eigen::Matrix<double, 1, 1> mean;
    mean.setZero();
    double sum_sin = 0, sum_cos = 0;
    for (int i = 0; i < sigmaPoints.cols(); i++) {
      sum_sin += sin(sigmaPoints(0, i)) * weights(i);
      sum_cos += cos(sigmaPoints(0, i)) * weights(i);
    }
    mean(0) = atan2(sum_sin, sum_cos);
    return mean;
  };

  Eigen::Matrix<double, 1, 1> z;
  z << -0.1;
  Eigen::Matrix<double, 1, 1> R;
  R << 1e-6;

  ukf.update<1>(z, R, measurementModel);

  EXPECT_NEAR(0.1, ukf.getState()(0), 0.01);
  EXPECT_NEAR(0.0, ukf.getState()(1), 0.01);
  EXPECT_NEAR(-0.09, ukf.getState()(2), 0.01);

  EXPECT_NEAR(0.1001, ukf.getCovariance().diagonal()(0), 0.001);
  EXPECT_NEAR(0.1001, ukf.getCovariance().diagonal()(1), 0.001);
  EXPECT_NEAR(0.01001, ukf.getCovariance().diagonal()(2), 0.001);
}

// Run all the tests
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}