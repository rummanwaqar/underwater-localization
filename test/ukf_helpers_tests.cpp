#include <au_localization/ukf_helpers.h>
#include <gtest/gtest.h>

void EXPECT_MATRIX_EQ(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      EXPECT_NEAR(a(i, j), b(i, j), 0.001);
    }
  }
}

TEST(SigmaPointsTest, weights) {
  auto sigmaPointsFn = MerweScaledSigmaPoints<3>(0.1, 2, 0.0);
  EXPECT_EQ(7, sigmaPointsFn.getNumSigmaPoints());
  EXPECT_FLOAT_EQ(-99, sigmaPointsFn.getWm()(0));
  EXPECT_FLOAT_EQ(-96.01, sigmaPointsFn.getWc()(0));
  for (int i = 1; i < sigmaPointsFn.getNumSigmaPoints(); i++) {
    EXPECT_FLOAT_EQ(16.66666667, sigmaPointsFn.getWm()(i));
    EXPECT_FLOAT_EQ(16.66666667, sigmaPointsFn.getWc()(i));
  }
}

TEST(SigmaPointsTest, computeSigmaPoints) {
  Eigen::Matrix<double, 3, 1> x;
  x << 0.1, 0.2, 0.3;
  Eigen::Matrix<double, 3, 3> P;
  P.setIdentity();

  auto sigmaPointFn = MerweScaledSigmaPoints<3>(0.1, 2, 0.0);
  auto sigmaPoints = sigmaPointFn.generateSigmaPoints(x, P);

  Eigen::Matrix<double, 3, 7> expectedSigmaPoints;
  expectedSigmaPoints << 0.1, 0.273205, 0.1, 0.1, -0.0732051, 0.1, 0.1, 0.2,
      0.2, 0.373205, 0.2, 0.2, 0.0267949, 0.2, 0.3, 0.3, 0.3, 0.473205, 0.3,
      0.3, 0.126795;
  EXPECT_MATRIX_EQ(expectedSigmaPoints, sigmaPoints);
};

TEST(SigmaPointsTest, computeSigmaPointsDifferenceFunc) {
  Eigen::Matrix<double, 3, 1> x;
  x << 0.1, 0.2, 0.3;
  Eigen::Matrix<double, 3, 3> P;
  P.setIdentity();

  auto sigmaPointFn = MerweScaledSigmaPoints<3>(0.1, 2, 0.0);
  sigmaPointFn.setDifferenceFunction(
      [](const Eigen::Matrix<double, 3, 1>& a,
         const Eigen::Matrix<double, 3, 1>& b) -> Eigen::Matrix<double, 3, 1> {
        Eigen::Matrix<double, 3, 1> diff = a;
        diff(0) -= b(0);
        diff(1) -= b(1);
        diff(2) -= b(2);
        return diff;
      });
  auto sigmaPoints = sigmaPointFn.generateSigmaPoints(x, P);

  Eigen::Matrix<double, 3, 7> expectedSigmaPoints;
  expectedSigmaPoints << 0.1, 0.273205, 0.1, 0.1, -0.0732051, 0.1, 0.1, 0.2,
      0.2, 0.373205, 0.2, 0.2, 0.0267949, 0.2, 0.3, 0.3, 0.3, 0.473205, 0.3,
      0.3, 0.126795;
  EXPECT_MATRIX_EQ(expectedSigmaPoints, sigmaPoints);
}

TEST(UnscentedTransformTest, UnscentedTransform) {
  Eigen::Matrix<double, 3, 1> x;
  x << 0.1, 0.2, 0.3;
  Eigen::Matrix<double, 3, 3> P;
  P.setIdentity();

  auto sigmaPointFn = MerweScaledSigmaPoints<3>(0.1, 2, 0.0);
  auto sigmaPoints = sigmaPointFn.generateSigmaPoints(x, P);

  Eigen::Matrix<double, 3, 3> noise;
  noise.setIdentity();

  Eigen::Matrix<double, 3, 1> predictedX;
  Eigen::Matrix<double, 3, 3> predictedP;
  std::tie(predictedX, predictedP) = unscented_transform<3, 7>(
      sigmaPoints, sigmaPointFn.getWm(), sigmaPointFn.getWc(), noise);

  for (int i = 0; i < 3; i++) {
    EXPECT_FLOAT_EQ(x(i), predictedX(i));
    EXPECT_FLOAT_EQ(2, predictedP(i, i));
  }
}

TEST(UnscentedTransformTest, UnscentedTransformCustomFuncs) {
  Eigen::Matrix<double, 3, 1> x;
  x << 0.1, 0.2, 0.3;
  Eigen::Matrix<double, 3, 3> P;
  P.setIdentity();

  auto diffFn =
      [](const Eigen::Matrix<double, 3, 1>& a,
         const Eigen::Matrix<double, 3, 1>& b) -> Eigen::Matrix<double, 3, 1> {
    Eigen::Matrix<double, 3, 1> diff = a;
    diff(0) -= b(0);
    diff(1) -= b(1);
    diff(2) -= b(2);
    return diff;
  };

  auto meanFn = [](const Eigen::Matrix<double, 3, 7>& sigmaPoints,
                   const Eigen::Matrix<double, 7, 1>& weights)
      -> Eigen::Matrix<double, 3, 1> {
    Eigen::Matrix<double, 3, 1> mean;
    mean.setZero();
    for (int i = 0; i < sigmaPoints.cols(); i++) {
      mean(0) += sigmaPoints(0, i) * weights(i);
      mean(1) += sigmaPoints(1, i) * weights(i);
      mean(2) += sigmaPoints(2, i) * weights(i);
    }
    return mean;
  };

  auto sigmaPointFn = MerweScaledSigmaPoints<3>(0.1, 2, 0.0);
  sigmaPointFn.setDifferenceFunction(diffFn);
  auto sigmaPoints = sigmaPointFn.generateSigmaPoints(x, P);

  Eigen::Matrix<double, 3, 3> noise;
  noise.setIdentity();

  Eigen::Matrix<double, 3, 1> predictedX;
  Eigen::Matrix<double, 3, 3> predictedP;
  std::tie(predictedX, predictedP) =
      unscented_transform<3, 7>(sigmaPoints, sigmaPointFn.getWm(),
                                sigmaPointFn.getWc(), noise, diffFn, meanFn);

  for (int i = 0; i < 3; i++) {
    EXPECT_FLOAT_EQ(x(i), predictedX(i));
    EXPECT_FLOAT_EQ(2, predictedP(i, i));
  }
}

// Run all the tests
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}