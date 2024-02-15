import unittest
import numpy as np
import impactstudy.experiment as ise


class LinearNormalTargetGeneratorTestCase(unittest.TestCase):
    def test_target_generator(self):
        seed = 1999
        a = np.array([1.0, 2.0, -3.0])
        sigma = 0.1
        n = 5000

        rng = np.random.default_rng(37)
        x = np.ones((3, n))
        c = rng.normal(size=(5, n))

        target_generator = ise.LinearNormalTargetGenerator(a, sigma, seed=seed)

        y = target_generator.f_prime(x, c)

        self.assertEqual((n, ), y.shape)

        # The noise gets smoothed out.
        self.assertAlmostEqual(0.0, y.mean(), delta=0.0002)
        self.assertAlmostEqual(sigma, y.std(), delta=0.001)

        self.assertLess(-0.4, y.min())
        self.assertGreater(0.4, y.max())

    def test_determinism(self):
        n = 200

        rng = np.random.default_rng(seed=12345)

        x = np.ones((3, n))
        c = rng.normal(size=(5, n))

        # Generate three y's, the first and last seeded the same and the middle
        # seeded differently.
        seeds = [1999, 0x1C45B81C, 1999]
        a = np.array([-1.0, 246.997, 3.0])
        sigma = 0.2

        target_generators = [ise.LinearNormalTargetGenerator(a, sigma, seed=seed) for seed in seeds]

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

        x, c = feature_generator(100)

        self.assertEqual((5, 100), x.shape)
        self.assertEqual((3, 100), c.shape)

        self.assertTrue((x >= 0.0).all())
        self.assertTrue((x < 1.0).all())

        self.assertTrue((c >= 0.0).all())
        self.assertTrue((c < 1.0).all())


if __name__ == '__main__':
    unittest.main()
