import unittest
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.random import RandomState

import impactstudy.experiment as ise


class StepTest(unittest.TestCase):

    def test_step(self):
        step_target_generator = ise.StepExactTargetGenerator(50.0, 200.0, -100.0)

        df = 50.0 + pd.DataFrame(
            [[-10.0], [-5.0], [0.0], [7.0], [27.0]], columns=["x_0"]
        )

        y = step_target_generator.f(df)
        df_impact = step_target_generator.impact(df)

        self.assertTrue(pd.Series([-100.0, -100.0, 100.0, 100.0, 100.0]).equals(y))

        self.assertTrue(
            pd.DataFrame(
                [[-120.0], [-120.0], [80.0], [80.0], [80.0]], columns=["x_0"]
            ).equals(df_impact)
        )

    def test_two_step(self):
        step_target_generator_1 = ise.StepExactTargetGenerator(25.0, 100.0, 0.0)
        step_target_generator_2 = ise.StepExactTargetGenerator(75.0, -100.0, 0.0)

        target_generator = ise.AdditiveExactTargetGenerator(
            [step_target_generator_1, step_target_generator_2]
        )

        df = pd.DataFrame(
            [[0.0, 100.0], [10.0, 90.0], [25.0, 80.0], [30.0, 75.0], [50.0, 50.0]],
            columns=["x_0", "x_1"],
        )

        y = target_generator.f(df)

        self.assertTrue(pd.Series([-100.0, -100.0, 0.0, 0.0, 100.0]).equals(y))

        df_impact = target_generator.impact(df)

        df_expected_impact = pd.DataFrame(
            [
                [-60.0, -20.0],
                [-60.0, -20.0],
                [40.0, -20.0],
                [40.0, -20.0],
                [40.0, 80.0],
            ],
            columns=["x_0", "x_1"],
        )

        self.assertTrue(df_expected_impact.equals(df_impact))


class PolyTest(unittest.TestCase):

    def test_quadratic(self):
        target_generator = ise.PolynomialExactTargetGenerator([1.0, 0.0, 10.0])

        self.assertTrue(
            (np.array([1.0, 0.0, 10]) == target_generator.coefficients).all()
        )

        df = pd.DataFrame([[0.0], [1.0], [2.0], [3.0]], columns=["x_0"])

        y = target_generator.f(df)
        df_impact = target_generator.impact(df)

        expected_f = pd.Series([10.0, 11.0, 14.0, 19.0])

        df_expected_impact = pd.DataFrame(
            [[-3.5], [-2.5], [0.5], [5.5]], columns=["x_0"]
        )

        self.assertTrue(expected_f.equals(y))
        self.assertTrue(df_expected_impact.equals(df_impact))


def linear_normal_target_generator(
    a: np.array, b: float, sigma: float, seed: int | RandomState = 17
) -> ise.TargetGenerator:
    linear_exact_target_generator = ise.LinearExactTargetGenerator(a, b)
    normal_noise_generator = ise.NormalNoiseGenerator(sigma, seed=seed)

    return ise.TargetGenerator(linear_exact_target_generator, normal_noise_generator)


class LinearNormalTargetGeneratorTestCase(unittest.TestCase):
    def test_target_generator(self):
        seed = 1999
        a = np.array([1.0, 2.0, -3.0])
        sigma = 0.1
        n = 5000

        rng = np.random.default_rng(37)
        x = pd.DataFrame(np.ones((n, 3)), columns=["x_0", "x_1", "x_2"])
        c = rng.normal(size=(5, n))

        target_generator = linear_normal_target_generator(a, 0.0, sigma, seed=seed)

        y = target_generator.f_prime(x, c)

        self.assertEqual((n,), y.shape)

        # The noise gets smoothed out.
        self.assertAlmostEqual(0.0, y.mean(), delta=0.0002)
        self.assertAlmostEqual(sigma, y.std(), delta=0.001)

        self.assertLess(-0.4, y.min())
        self.assertGreater(0.4, y.max())

    def test_determinism(self):
        n = 200

        rng = np.random.default_rng(seed=12345)

        x = pd.DataFrame(np.ones((n, 3)), columns=["x_0", "x_1", "x_2"])
        c = rng.normal(size=(5, n))

        # Generate three y's, the first and last seeded the same and the middle
        # seeded differently.
        seeds = [1999, 0x1C45B81C, 1999]
        a = np.array([-1.0, 246.997, 3.0])
        sigma = 0.2

        target_generators = [
            linear_normal_target_generator(a, 0.0, sigma, seed=seed) for seed in seeds
        ]

        ys = [target_generator.f_prime(x, c) for target_generator in target_generators]

        # Same seed gives same results:
        self.assertTrue((ys[0] == ys[2]).all())
        # Different seed gives different results.
        self.assertTrue((ys[0] != ys[1]).all())


class UniformFeatureGeneratorTestCase(unittest.TestCase):
    def test_uniform_feature_generator(self):
        m = 5
        s = 3
        feature_generator = ise.UniformFeatureGenerator(m, s, seed=12345)

        df_x, df_c = feature_generator(100)

        self.assertEqual((100, 5), df_x.shape)
        self.assertEqual((100, 3), df_c.shape)

        self.assertTrue((df_x >= 0.0).all().all())
        self.assertTrue((df_x < 1.0).all().all())

        self.assertTrue((df_c >= 0.0).all().all())
        self.assertTrue((df_c < 1.0).all().all())


class UncorrelatedXCTestCase(unittest.TestCase):
    def setUp(self):
        mu = 50
        sigma = 20
        corr_xx = 0.8
        corr_cc = 0.2

        self.mu = mu
        self.sigma = sigma
        self.cov = np.array(
            [
                [sigma * sigma, sigma * sigma * corr_xx, 0.0, 0.0],
                [sigma * sigma * corr_xx, sigma * sigma, 0.0, 0.0],
                [0.0, 0.0, sigma * sigma, sigma * sigma * corr_cc],
                [0.0, 0.0, sigma * sigma * corr_cc, sigma * sigma],
            ]
        )
        self.m = 2

        self.feature_generator = ise.CorrelatedFeatureGenerator(
            self.cov, self.mu, self.m
        )

    def test_uncorrelated_xc(self):
        df_x, df_c = self.feature_generator(100)

        self.assertEqual((100, 2), df_x.shape)
        self.assertEqual((100, 2), df_c.shape)

        # Correlated X's.
        corr_xx = np.corrcoef(df_x["x_0"], df_x["x_1"])

        self.assertAlmostEqual(1.0, corr_xx[0][0], places=10)
        self.assertAlmostEqual(1.0, corr_xx[1][1], places=10)

        self.assertTrue(0.83 < corr_xx[0][1] < 0.84)
        self.assertTrue(0.83 < corr_xx[1][0] < 0.84)

        # Correlated CC
        corr_cc = np.corrcoef(df_c["c_0"], df_c["c_1"])

        self.assertAlmostEqual(1.0, corr_cc[0][0], places=10)
        self.assertAlmostEqual(1.0, corr_cc[1][1], places=10)

        self.assertTrue(0.34 < corr_cc[0][1] < 0.35)
        self.assertTrue(0.34 < corr_cc[1][0] < 0.35)

        # Uncorrelated XC
        for x_col in ["x_0", "x_1"]:
            for c_col in ["c_0", "c_1"]:
                corr_xc = np.corrcoef(df_x[x_col], df_c[c_col])

                self.assertAlmostEqual(1.0, corr_xc[0][0], places=10)
                self.assertAlmostEqual(1.0, corr_xc[1][1], places=10)

                self.assertTrue(-0.2 < corr_xc[0][1] < 0.2)
                self.assertTrue(-0.2 < corr_xc[1][0] < 0.2)


class CorrelatedXCTestCase(unittest.TestCase):
    def setUp(self):
        mu = 50
        sigma = 20
        corr = 0.7

        sigma2 = sigma * sigma
        sigma2_corr = sigma2 * corr

        self.mu = mu
        self.sigma = sigma
        self.cov = np.array(
            [
                [sigma2, 0.0, 0.0, 0.0],
                [0.0, sigma2, 0.0, 0.0],
                [0.0, 0.0, sigma2, sigma2_corr],
                [0.0, 0.0, sigma2_corr, sigma2],
            ]
        )
        self.m = 3

        self.feature_generator = ise.CorrelatedFeatureGenerator(
            self.cov, self.mu, self.m
        )

    def test_correlated_xc(self):
        df_x, df_c = self.feature_generator(100)

        self.assertEqual((100, 3), df_x.shape)
        self.assertEqual((100, 1), df_c.shape)

        # Correlated X and C
        corr_x2c0 = np.corrcoef(df_x["x_2"], df_c["c_0"])

        self.assertAlmostEqual(1.0, corr_x2c0[0][0], places=10)
        self.assertAlmostEqual(1.0, corr_x2c0[1][1], places=10)

        self.assertTrue(0.65 < corr_x2c0[0][1] < 0.75)
        self.assertTrue(0.65 < corr_x2c0[1][0] < 0.75)

        # Uncorrelated X and C
        for xcol in ["x_0", "x_1"]:
            corr_xc0 = np.corrcoef(df_x[xcol], df_c["c_0"])

            self.assertAlmostEqual(1.0, corr_xc0[0][0], places=10)
            self.assertAlmostEqual(1.0, corr_xc0[1][1], places=10)

            self.assertTrue(-0.15 < corr_xc0[0][1] < 0.15)
            self.assertTrue(-0.15 < corr_xc0[1][0] < 0.15)

        # Uncorrelated X
        for col0 in ["x_0", "x_1", "x_2"]:
            for col1 in ["x_0", "x_1", "x_2"]:
                corr_xx = np.corrcoef(df_x[col0], df_x[col1])
                self.assertAlmostEqual(1.0, corr_xx[0][0], places=10)
                self.assertAlmostEqual(1.0, corr_xx[1][1], places=10)
                if col0 == col1:
                    self.assertAlmostEqual(1.0, corr_xx[0][1], places=10)
                    self.assertAlmostEqual(1.0, corr_xx[1][0], places=10)
                else:
                    self.assertTrue(-0.15 < corr_xx[0][1] < 0.15)
                    self.assertTrue(-0.15 < corr_xx[1][0] < 0.15)

    def test_with_independence(self):
        # Add some additional independent x_i and c_i
        uniform_feature_generator = ise.UniformFeatureGenerator(
            m=2, s=2, low=0.0, high=100.0, seed=99
        )
        feature_generator = ise.ConcatenatedFeatureGenerator(
            [self.feature_generator, uniform_feature_generator]
        )

        df_x, df_c = feature_generator(100)

        self.assertEqual((100, 5), df_x.shape)
        self.assertEqual((100, 3), df_c.shape)

        for col0 in ["x_0", "x_1", "x_2"]:
            for col1 in ["x_3", "x_4"]:
                corr_xx = np.corrcoef(df_x[col0], df_x[col1])
                self.assertAlmostEqual(1.0, corr_xx[0][0], places=10)
                self.assertAlmostEqual(1.0, corr_xx[1][1], places=10)
                self.assertTrue(-0.1 < corr_xx[0][1] < 0.1)
                self.assertTrue(-0.1 < corr_xx[1][0] < 0.1)

        for col0 in ["c_0"]:
            for col1 in ["c_1", "c_2"]:
                corr_cc = np.corrcoef(df_c[col0], df_c[col1])
                self.assertAlmostEqual(1.0, corr_cc[0][0], places=10)
                self.assertAlmostEqual(1.0, corr_cc[1][1], places=10)
                self.assertTrue(-0.1 < corr_cc[0][1] < 0.1)
                self.assertTrue(-0.1 < corr_cc[1][0] < 0.1)


class ConcatenatedFeatureGeneratorTestCase(unittest.TestCase):
    def setUp(self):
        self.fg0 = ise.NormalFeatureGenerator(3, 1, 50.0, 10.0)
        self.fg1 = ise.UniformFeatureGenerator(2, 7, low=0.0, high=100.0)

        self.feature_generator = ise.ConcatenatedFeatureGenerator([self.fg0, self.fg1])

    def test_concatenated_feature_generator(self):
        df_x0, df_c0 = self.fg0(100)
        df_x1, df_c1 = self.fg1(100)

        self.assertEqual((100, 3), df_x0.shape)
        self.assertEqual((100, 1), df_c0.shape)

        self.assertEqual((100, 2), df_x1.shape)
        self.assertEqual((100, 7), df_c1.shape)

        df_x, df_c = self.feature_generator(100)

        self.assertEqual((100, 5), df_x.shape)
        self.assertEqual((100, 8), df_c.shape)

        for ii in range(3):
            self.assertTrue(df_x[f"x_{ii}"].equals(df_x0[f"x_{ii}"]))
        for ii in range(2):
            self.assertTrue(df_x[f"x_{ii + 3}"].equals(df_x1[f"x_{ii}"]))

        for ii in range(1):
            self.assertTrue(df_c[f"c_{ii}"].equals(df_c0[f"c_{ii}"]))
        for ii in range(7):
            self.assertTrue(df_c[f"c_{ii + 1}"].equals(df_c1[f"c_{ii}"]))


class ScenarioGeneratorTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.n = 100
        fg = ise.UniformFeatureGenerator(2, 2, low=0.0, high=100.0)
        tg = linear_normal_target_generator([0.5, -1.0], 0.0, 10.0)
        self.sg = ise.Scenario(fg, tg, self.n)

    def test_scenario(self):
        df_scenario = self.sg.training_data()

        self.assertEqual((self.n, 5), df_scenario.shape)


class MockFeatureGenerator(ise.FeatureGenerator):

    def __init__(self, m: int):
        super().__init__(m, 1, 17)

    def __call__(self, n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return pd.DataFrame(
            [[100.0 * ii + jj for jj in range(self._m)] for ii in range(n)],
            columns=[f"x_{ii}" for ii in range(self._m)],
        ), pd.DataFrame([[1.0]] * n, columns=["c_0"])


class MockTwoColumnTargetGenerator(ise.ExactTargetGenerator):
    def arity(self) -> int:
        return 2

    def f(self, x: pd.DataFrame) -> pd.Series:
        # We expect these column names even if they
        # were mapped out of a wider set of features.
        return 1000 * x["x_0"] + x["x_1"]

    def impact(self, x: pd.DataFrame) -> pd.DataFrame:
        df_impact = pd.DataFrame()
        df_impact["x_0"] = 1000 * x["x_0"] + x["x_1"].mean()
        df_impact["x_1"] = 1000 * x["x_0"].mean() + x["x_1"]

        return df_impact


class AdditiveFeatureTestCase(unittest.TestCase):

    def test_additive_feature(self):
        n = 3

        feature_generator = MockFeatureGenerator(6)

        additive_target_generator = ise.AdditiveExactTargetGenerator(
            [
                MockTwoColumnTargetGenerator(),
                MockTwoColumnTargetGenerator(),
                MockTwoColumnTargetGenerator(),
            ],
        )

        self.assertEqual(6, additive_target_generator.arity())

        scenario = ise.Scenario(
            feature_generator, ise.TargetGenerator(additive_target_generator), n
        )

        x_cols = scenario.x_cols()

        self.assertEqual(["x_0", "x_1", "x_2", "x_3", "x_4", "x_5"], x_cols)

        df_training_data = scenario.training_data()

        self.assertTrue(
            pd.DataFrame(
                [
                    [6_009.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0],
                    [306_309.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 1.0],
                    [606_609.0, 200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 1.0],
                ],
                columns=["y", "x_0", "x_1", "x_2", "x_3", "x_4", "x_5", "c_0"],
            ).equals(df_training_data)
        )

        df_true_impact = scenario.true_impact()

        self.assertTrue(
            pd.DataFrame(
                [
                    [-100_000.0, -100.0, -100_000.0, -100.0, -100_000.0, -100.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [100_000.0, 100.0, 100_000.0, 100.0, 100_000.0, 100.0, 0.0],
                ],
                columns=["x_0", "x_1", "x_2", "x_3", "x_4", "x_5", "c_0"],
            ).equals(df_true_impact)
        )


class KitchenSinkTestCase(unittest.TestCase):

    def test_split(self):
        for m_total in range(1, 21):
            experiment = ise.KitchenSinkExperiment(m=m_total, s=0, sigma=20.0, n=100)
            ms = experiment.ms

            self.assertEqual(m_total, sum(ms))


if __name__ == "__main__":
    unittest.main()
