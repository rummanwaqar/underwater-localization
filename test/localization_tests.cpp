#include <au_localization/localization.h>
#include <gtest/gtest.h>

TEST(LocalizationTests, prediction) {
  Localization localization(1e-2, 2, 0);
  localization.reset();

  Eigen::Matrix<double, STATE_SIZE, 1> x;
  x << 0, 0, 0, 0, 0, 0, 0.3, 0.2, 0.1, 0.5, 0.6, 0.7, 0.05, -0.05, 0;
  localization.setState(x);
  localization.predict(0.1);

  Eigen::Matrix<double, STATE_SIZE, 1> expectedX;
  expectedX << 0.3 * 0.1 + 0.05 * 0.01 * 0.5, 0.2 * 0.1 - 0.05 * 0.01 * 0.5,
      0.1 * 0.1, 0.05, 0.06, 0.07, 0.3 + 0.05 * 0.1, 0.2 - 0.05 * 0.1, 0.1, 0.5,
      0.6, 0.7, 0.05, -0.05, 0;
  Eigen::Matrix<double, STATE_SIZE, 1> expectedP;
  expectedP << 0.05, 0.05, 0.05, 0.03, 0.03, 0.06, 0.025, 0.025, 0.025, 0.01,
      0.01, 0.02, 0.01, 0.01, 0.01;

  for (int i = 0; i < STATE_SIZE; i++) {
    EXPECT_FLOAT_EQ(expectedX(i), localization.getState()(i));
    EXPECT_FLOAT_EQ(expectedP(i), localization.getCovariance().diagonal()(i));
  }
}

TEST(LocalizationTests, initialization) {
  Localization localization(1e-2, 2, 0);
  localization.reset();
  EXPECT_FALSE(localization.isInitialized());

  // depth sensor should not trigger initialization
  Measurement depth(DEPTH_SIZE);
  depth.type = MeasurementTypeDepth;
  depth.measurement << 2;
  depth.covariance << 1e-4;
  localization.processMeasurement(depth);
  EXPECT_FALSE(localization.isInitialized());

  // imu sensor should trigger initialization
  Measurement imu(IMU_SIZE);
  imu.type = MeasurementTypeImu;
  imu.measurement(ImuRoll) = imu.measurement(ImuPitch) =
      imu.measurement(ImuYaw) = M_PI_4;
  imu.measurement(ImuVroll) = imu.measurement(ImuVpitch) =
      imu.measurement(ImuVyaw) = 0.2;
  imu.measurement(ImuAx) = imu.measurement(ImuAy) = imu.measurement(ImuAz) =
      0.0;
  imu.covariance.block<3, 3>(ImuRoll, ImuRoll) =
      Eigen::Matrix3d::Identity(3, 3);
  imu.covariance.block<3, 3>(ImuVroll, ImuVroll) =
      Eigen::Matrix3d::Identity(3, 3) * 0.1;
  imu.covariance.block<3, 3>(ImuAx, ImuAx) =
      Eigen::Matrix3d::Identity(3, 3) * 0.05;
  localization.processMeasurement(imu);
  EXPECT_TRUE(localization.isInitialized());

  Eigen::VectorXd expectedState(STATE_SIZE);
  expectedState << 0, 0, 0, M_PI_4, M_PI_4, M_PI_4, 0, 0, 0, 0.2, 0.2, 0.2, 0,
      0, 0;
  Eigen::VectorXd expectedCov(STATE_SIZE);
  expectedCov << 1e-9, 1e-9, 1e-9, 1, 1, 1, 1e-9, 1e-9, 1e-9, 0.1, 0.1, 0.1,
      0.05, 0.05, 0.05;
  for (int i = 0; i < STATE_SIZE; i++) {
    EXPECT_FLOAT_EQ(expectedState(i), localization.getState()(i));
    EXPECT_FLOAT_EQ(expectedCov(i), localization.getCovariance().diagonal()(i));
  }
}

TEST(LocalizationTests, depthMeasurement) {
  Localization localization(1e-2, 2, 0);
  auto cov = Eigen::Matrix<double, STATE_SIZE, STATE_SIZE>::Identity(
                 STATE_SIZE, STATE_SIZE) *
             0.1;
  localization.setInitialCovariance(cov);
  localization.reset();

  // initialize localization class with IMU measurement
  Measurement imu(IMU_SIZE);
  imu.type = MeasurementTypeImu;
  imu.time = 0.0;
  imu.measurement(ImuRoll) = imu.measurement(ImuPitch) =
      imu.measurement(ImuYaw) = 0;
  imu.measurement(ImuVroll) = imu.measurement(ImuVpitch) =
      imu.measurement(ImuVyaw) = 0.0;
  imu.measurement(ImuAx) = imu.measurement(ImuAy) = imu.measurement(ImuAz) =
      0.01;
  imu.covariance.block<3, 3>(ImuRoll, ImuRoll) =
      Eigen::Matrix3d::Identity(3, 3);
  imu.covariance.block<3, 3>(ImuVroll, ImuVroll) =
      Eigen::Matrix3d::Identity(3, 3) * 0.1;
  imu.covariance.block<3, 3>(ImuAx, ImuAx) =
      Eigen::Matrix3d::Identity(3, 3) * 0.05;
  localization.processMeasurement(imu);

  Measurement depth(DEPTH_SIZE);
  depth.type = MeasurementTypeDepth;
  depth.time = 0.1;  // deltaT = 0.1
  depth.measurement << 2;
  depth.covariance << 1e-2;
  localization.processMeasurement(depth);

  Eigen::Matrix<double, STATE_SIZE, 1> expectedState;
  expectedState << 0, 0, 1.820, 0.0, 0.0, 0.0, 0.001, 0.001, 0.182, 0, 0, 0,
      0.01, 0.01, 0.0145;
  Eigen::Matrix<double, STATE_SIZE, 1> expectedCov;
  expectedCov << 0.15, 0.15, 0.06, 1.03, 1.03, 1.06, 0.125, 0.125, 0.125, 0.11,
      0.11, 0.12, 0.06, 0.06, 0.06;

  for (size_t i = 0; i < STATE_SIZE; i++) {
    EXPECT_NEAR(expectedState(i), localization.getState()(i), 0.001);
    EXPECT_NEAR(expectedCov(i), localization.getCovariance().diagonal()(i),
                0.01);
  }
}

TEST(LocalizationTests, dvlMeasurement) {
  Localization localization(1e-2, 2, 0);
  auto cov = Eigen::Matrix<double, STATE_SIZE, STATE_SIZE>::Identity(
                 STATE_SIZE, STATE_SIZE) *
             0.1;
  localization.setInitialCovariance(cov);
  localization.reset();

  // initialize localization class with IMU measurement
  Measurement imu(IMU_SIZE);
  imu.type = MeasurementTypeImu;
  imu.time = 0.0;
  imu.measurement(ImuRoll) = imu.measurement(ImuPitch) =
      imu.measurement(ImuYaw) = 0;
  imu.measurement(ImuVroll) = imu.measurement(ImuVpitch) =
      imu.measurement(ImuVyaw) = 0.0;
  imu.measurement(ImuAx) = imu.measurement(ImuAy) = imu.measurement(ImuAz) =
      0.01;
  imu.covariance.block<3, 3>(ImuRoll, ImuRoll) =
      Eigen::Matrix3d::Identity(3, 3);
  imu.covariance.block<3, 3>(ImuVroll, ImuVroll) =
      Eigen::Matrix3d::Identity(3, 3) * 0.1;
  imu.covariance.block<3, 3>(ImuAx, ImuAx) =
      Eigen::Matrix3d::Identity(3, 3) * 0.05;
  localization.processMeasurement(imu);

  Measurement dvl(DVL_SIZE);
  dvl.time = 0.1;  // deltaT = 0.1
  dvl.type = MeasurementTypeDvl;
  dvl.measurement << 0.2, 0.1, 0.15;
  dvl.covariance << 1e-2, 0, 0, 0, 1e-2, 0, 0, 0, 1e-2;
  localization.processMeasurement(dvl);

  Eigen::Matrix<double, STATE_SIZE, 1> expectedState;
  expectedState << 0.018, 0.009, 0.0135, 0, 0, 0, 0.18, 0.09, 0.14, 0, 0, 0,
      0.019, 0.0145, 0.017;
  Eigen::Matrix<double, STATE_SIZE, 1> expectedCov;
  expectedCov << 0.15, 0.15, 0.15, 1.03, 1.03, 1.06, 0.034, 0.034, 0.034, 0.11,
      0.11, 0.12, 0.06, 0.06, 0.06;

  for (size_t i = 0; i < STATE_SIZE; i++) {
    EXPECT_NEAR(expectedState(i), localization.getState()(i), 0.01);
    EXPECT_NEAR(expectedCov(i), localization.getCovariance().diagonal()(i),
                0.01);
  }
}

TEST(LocalizationTests, imuMeasurement) {
  Localization localization(1e-2, 2, 0);
  auto cov = Eigen::Matrix<double, STATE_SIZE, STATE_SIZE>::Identity(
                 STATE_SIZE, STATE_SIZE) *
             0.1;
  localization.setInitialCovariance(cov);
  localization.reset();

  // initialize localization class with IMU measurement
  Measurement imu(IMU_SIZE);
  imu.type = MeasurementTypeImu;
  imu.time = 0.0;
  imu.measurement(ImuRoll) = imu.measurement(ImuPitch) =
      imu.measurement(ImuYaw) = 0.5;
  imu.measurement(ImuVroll) = imu.measurement(ImuVpitch) =
      imu.measurement(ImuVyaw) = 0.1;
  imu.measurement(ImuAx) = imu.measurement(ImuAy) = imu.measurement(ImuAz) =
      0.00;
  imu.covariance.block<3, 3>(ImuRoll, ImuRoll) =
      Eigen::Matrix3d::Identity(3, 3) * 1e-2;
  imu.covariance.block<3, 3>(ImuVroll, ImuVroll) =
      Eigen::Matrix3d::Identity(3, 3) * 1e-3;
  imu.covariance.block<3, 3>(ImuAx, ImuAx) =
      Eigen::Matrix3d::Identity(3, 3) * 1e-3;
  localization.processMeasurement(imu);

  imu.time = 0.1;  // deltaT = 0.1
  imu.type = MeasurementTypeImu;
  imu.measurement << 0.8, 0.8, 0.8, 0.15, 0.15, 0.15, 0.01, 0.01, 0.01;
  localization.processMeasurement(imu);

  Eigen::Matrix<double, STATE_SIZE, 1> expectedState;
  expectedState << 0, 0, 0, 0.66, 0.65, 0.66, 0, 0, 0, 0.13, 0.13, 0.13, 0.005,
      0.005, 0.005;
  Eigen::Matrix<double, STATE_SIZE, 1> expectedCov;
  expectedCov << 0.15, 0.15, 0.15, 0.04, 0.04, 0.06, 0.125, 0.125, 0.125, 0.01,
      0.01, 0.02, 0.01, 0.01, 0.01;

  for (size_t i = 0; i < STATE_SIZE; i++) {
    EXPECT_NEAR(expectedState(i), localization.getState()(i), 0.01);
    EXPECT_NEAR(expectedCov(i), localization.getCovariance().diagonal()(i),
                0.01);
  }
}

// Run all the tests
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}