"""Experiments."""

from abc import ABC, abstractmethod
from typing import Tuple, List
from functools import cache

import numpy as np
from numpy.random import RandomState, default_rng
import pandas as pd


class TargetGenerator(ABC):

    def __init__(self, seed: int | RandomState):
        self._rng = default_rng(seed=seed)

    def f_prime(self, x: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
        f = self.f(x)
        z = self.z(size=x.shape[0])

        return f + z

    def setup(self):
        np.random.seed(self._seed)

    @abstractmethod
    def f(self, x: pd.DataFrame) -> pd.Series:
        """The function $F$ from features to target."""
        raise NotImplementedError("Not implemented on abstract class.")

    @abstractmethod
    def impact(self, x: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Not implemented on abstract class.")

    @abstractmethod
    def z(self, size: int) -> pd.Series:
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

    def f(self, df_x: pd.DataFrame) -> pd.Series:
        df_ax = df_x.mul(self._a, axis="columns")

        f = df_ax.sum(axis="columns")
        f.rename("f", inplace=True)
        return f

    def impact(self, x: pd.DataFrame) -> pd.DataFrame:
        term = np.multiply(self._a, x)
        impact = term - term.mean()

        return impact

    @cache
    def z(self, size: int) -> pd.Series:
        return pd.Series(self._rng.normal(0, self._sigma, size=size), name="z")


class FeatureGenerator(ABC):
    """A class to generate features and confounding features."""

    def __init__(self, m: int, s: int, seed: int | RandomState):
        self._m = m
        self._s = s
        self._rng = default_rng(seed)

    def x_cols(self) -> List[str]:
        return [f"x_{ii}" for ii in range(self._m)]

    def c_cols(self) -> List[str]:
        return [f"c_{ii}" for ii in range(self._s)]

    def x_prime_cols(self) -> List[str]:
        return self.x_cols() + self.c_cols()

    @abstractmethod
    def __call__(
        self, n: int, seed: int | RandomState
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError("Not implemented on abstract class.")


class UniformFeatureGenerator(FeatureGenerator):
    def __init__(
        self,
        m: int,
        s: int,
        *,
        low: float = 0.0,
        high: float = 1.0,
        seed: int | RandomState = 17,
    ):
        super().__init__(m=m, s=s, seed=seed)
        self._low = low
        self._high = high

    @cache
    def __call__(self, n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        x = self._rng.uniform(self._low, self._high, size=(n, self._m))
        c = self._rng.uniform(low=self._low, high=self._high, size=(n, self._s))

        df_x = pd.DataFrame(x, columns=self.x_cols())
        df_c = pd.DataFrame(c, columns=self.c_cols())

        return df_x, df_c


class ScenarioGenerator:

    def __init__(
        self, feature_generator: FeatureGenerator, target_generator: TargetGenerator
    ):
        self._feature_generator = feature_generator
        self._target_generator = target_generator

    def x_cols(self) -> List[str]:
        return self._feature_generator.x_cols()

    def c_cols(self) -> List[str]:
        return self._feature_generator.c_cols()

    def x_prime_cols(self) -> List[str]:
        return self._feature_generator.x_prime_cols()

    def y_col(self) -> str:
        return "y"

    def scenario(self, n: int) -> pd.DataFrame:
        df_x, df_c = self._feature_generator(n)
        y = self._target_generator.f_prime(df_x, df_c)

        df_y = pd.DataFrame()
        df_y["y"] = y

        df = pd.concat([df_y, df_x, df_c], axis="columns")

        return df

    def impact(self, n: int) -> pd.DataFrame:
        df_x, df_c = self._feature_generator(n)

        df_impact = self._target_generator.impact(df_x)
        for col in df_c.columns:
            df_impact[col] = 0.0

        return df_impact
