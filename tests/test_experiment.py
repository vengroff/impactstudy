import unittest
import numpy as np
import pandas as pd
import impactstudy.experiment as ise


class LinearNormalTargetGeneratorTestCase(unittest.TestCase):
    def test_target_generator(self):
        seed = 1999
        a = np.array([1.0, 2.0, -3.0])
        sigma = 0.1
        n = 5000

        rng = np.random.default_rng(37)
        x = pd.DataFrame(np.ones((n, 3)), columns=["x_0", "x_1", "x_2"])
        c = rng.normal(size=(5, n))

        target_generator = ise.LinearNormalTargetGenerator(a, 0.0, sigma, seed=seed)

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
            ise.LinearNormalTargetGenerator(a, 0.0, sigma, seed=seed) for seed in seeds
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
        tg = ise.LinearNormalTargetGenerator([0.5, -1.0], 0.0, 10.0)
        self.sg = ise.Scenario(fg, tg)

    def test_scenario(self):
        n = 100

        df_scenario = self.sg.scenario(n)

        self.assertEqual((n, 5), df_scenario.shape)


if __name__ == "__main__":
    unittest.main()
