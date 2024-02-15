"""Experiments."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy.random import RandomState, default_rng
import pandas as pd

class TargetGenerator(ABC):

    def __init__(self, seed: int | RandomState):
        self._rng = default_rng(seed=seed)

    def f_prime(self, x: np.array, c: np.array) -> np.array:
        return self.f(x) + self.z(size=x.shape[1:])

    def setup(self):
        np.random.seed(self._seed)

    @abstractmethod
    def f(self, x: np.array) -> np.array:
        """The function $F$ from features to target."""
        raise NotImplementedError("Not implemented on abstract class.")

    @abstractmethod
    def z(self, size: int | Tuple) -> np.array:
        """The noise $Z$."""
        raise NotImplementedError("Not implemented on abstract class.")


class LinearNormalTargetGenerator(TargetGenerator):

    def __init__(self, a: np.array, sigma: float, seed: int | RandomState = 17):
        super().__init__(seed)
        self._a = a
        self._sigma = sigma

    @property
    def a(self) -> np.array:
        return self._a

    @property
    def sigma(self) -> float:
        return self._sigma

    def f(self, x: np.array) -> float:
        return np.dot(self._a, x)

    def z(self, size: int | Tuple) -> np.array:
        return self._rng.normal(0, self._sigma, size=size)


class FeatureGenerator(ABC):
    """A class to generate features and confounding featurs."""

    def __init__(self, m: int, s: int, seed: int | RandomState):
        self._m = m
        self._s = s
        self._rng = default_rng(seed)

    @abstractmethod
    def __call__(self, n: int, seed: int | RandomState) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Not implemented on abstract class.")


class UniformFeatureGenerator(FeatureGenerator):
    def __init__(self, m: int, s: int, *, low: float = 0.0, high: float = 1.0, seed: int | RandomState = 17):
        super().__init__(m=m, s=s, seed=seed)
        self._low = low
        self._high = high

    def __call__(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        x = self._rng.uniform(self._low, self._high, size=(self._m, n))
        c = self._rng.uniform(low=self._low, high=self._high, size=(self._s, n))

        return x, c


class ScenarioGenerator:

    def __init__(self, feature_generator: FeatureGenerator, target_generator: TargetGenerator):
        self._feature_generator = feature_generator
        self._target_generator = target_generator

    def scenario(self, n: int) -> pd.DataFrame:
        x, c = self._feature_generator(n)
        y = self._target_generator.f_prime(x, c)

        df = pd.DataFrame()

        df['y'] = y
        for ii in range(x.shape[0]):
            df[f'x_{ii}'] = x[ii]
        for ii in range(c.shape[0]):
            df[f'c_{ii}'] = c[ii]

        return df
