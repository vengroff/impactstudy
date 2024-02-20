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


class ScenarioGeneratorTestCase(unittest.TestCase):

    def setUp(self) -> None:
        fg = ise.UniformFeatureGenerator(2, 2, low=0.0, high=100.0)
        tg = linear_normal_target_generator([0.5, -1.0], 0.0, 10.0)
        self.sg = ise.Scenario(fg, tg)

    def test_scenario(self):
        n = 100

        df_scenario = self.sg.training_data(n)

        self.assertEqual((n, 5), df_scenario.shape)


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
            feature_generator, ise.TargetGenerator(additive_target_generator)
        )

        x_cols = scenario.x_cols()

        self.assertEqual(["x_0", "x_1", "x_2", "x_3", "x_4", "x_5"], x_cols)

        df_training_data = scenario.training_data(3)

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

        df_true_impact = scenario.true_impact(3)

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


if __name__ == "__main__":
    unittest.main()
