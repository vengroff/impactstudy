"""Experiments."""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Generator, Iterable
from functools import cache

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import RandomState, default_rng
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
)

from impactchart.model import ImpactModel, XGBoostImpactModel


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

    def __init__(
        self, a: np.array, b: float, sigma: float, seed: int | RandomState = 17
    ):
        super().__init__(seed)
        self._a = a
        self._b = b
        self._sigma = sigma

    @property
    def a(self) -> np.array:
        return self._a

    @property
    def b(self) -> float:
        return self._b

    @property
    def sigma(self) -> float:
        return self._sigma

    def f(self, df_x: pd.DataFrame) -> pd.Series:
        df_ax = df_x.mul(self._a, axis="columns")

        f = df_ax.sum(axis="columns") + self._b
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


class Scenario:

    def __init__(
        self,
        feature_generator: FeatureGenerator,
        target_generator: TargetGenerator,
        *,
        impact_model_seed: int = 0x17E3FB61,
    ):
        self._feature_generator = feature_generator
        self._target_generator = target_generator
        self._impact_model_seed = impact_model_seed

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

    @cache
    def true_impact(self, n: int) -> pd.DataFrame:
        df_x, df_c = self._feature_generator(n)

        df_impact = self._target_generator.impact(df_x)
        for col in df_c.columns:
            df_impact[col] = 0.0

        return df_impact

    @cache
    def impact_model(self, n: int) -> ImpactModel:
        df = self.scenario(n)

        df_X_prime = df[self.x_prime_cols()]
        y = df[self.y_col()]

        impact_model = XGBoostImpactModel(random_state=self._impact_model_seed)
        impact_model.fit(df_X_prime, y)

        return impact_model

    @cache
    def model_mean_impact(self, n: int) -> pd.DataFrame:
        impact_model = self.impact_model(n)

        df = self.scenario(n)
        df_X_prime = df[self.x_prime_cols()]

        return impact_model.mean_impact(df_X_prime)

    def model_impact_charts(self, n: int) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        impact_model = self.impact_model(n)

        df = self.scenario(n)
        df_X_prime = df[self.x_prime_cols()]

        impact_charts = impact_model.impact_charts(
            df_X_prime,
            subplots_kwargs=dict(figsize=(12, 8)),
        )

        df_true_impact = self.true_impact(n)

        for col in df_true_impact.columns:
            fig, ax = impact_charts[col]
            ax.scatter(
                df[col],
                df_true_impact[col],
                c="C1",
                marker=".",
                s=10,
                zorder=10,
                label="Actual impact",
            )

        return impact_charts

    def root_mean_squared_error(self, n: int) -> pd.DataFrame:
        df_true_impact = self.true_impact(n)
        df_model_impact = self.model_mean_impact(n)

        df_rmse = np.sqrt(
            pd.DataFrame(
                [
                    mean_squared_error(
                        df_true_impact, df_model_impact, multioutput="raw_values"
                    )
                ],
                columns=df_true_impact.columns,
            )
        )

        return df_rmse

    def mean_absolute_error(self, n: int) -> pd.DataFrame:
        df_true_impact = self.true_impact(n)
        df_model_impact = self.model_mean_impact(n)

        df_mae = pd.DataFrame(
            [
                mean_absolute_error(
                    df_true_impact, df_model_impact, multioutput="raw_values"
                )
            ],
            columns=df_true_impact.columns,
        )

        return df_mae

    def median_absolute_error(self, n: int) -> pd.DataFrame:
        df_true_impact = self.true_impact(n)
        df_model_impact = self.model_mean_impact(n)

        df_mae = pd.DataFrame(
            [
                median_absolute_error(
                    df_true_impact, df_model_impact, multioutput="raw_values"
                )
            ],
            columns=df_true_impact.columns,
        )

        return df_mae

    def model_errors(self, n: int) -> pd.DataFrame:
        df_rmse = self.root_mean_squared_error(n)
        df_mae = self.mean_absolute_error(n)
        df_medae = self.median_absolute_error(n)

        df_rmse["metric"] = "RMSE"
        df_mae["metric"] = "MAE"
        df_medae["metric"] = "MED_AE"

        df_errors = pd.concat([df_rmse, df_mae, df_medae], axis="rows")
        df_errors = df_errors[
            ["metric"] + [col for col in df_errors.columns if col != "metric"]
        ]

        return df_errors


class Experiment(ABC):

    @abstractmethod
    def scenarios(
        self,
    ) -> Generator[Tuple[Dict[str, int | float], Scenario], None, None]:
        raise NotImplementedError("Not implemented on abstract class.")

    def model_errors(self, n: int) -> pd.DataFrame:
        def scenario_errors() -> Generator[pd.DataFrame, None, None]:
            for tags, scenario in self.scenarios():
                df_scenario_model_errors = scenario.model_errors(n)
                for k, v in tags.items():
                    df_scenario_model_errors[k] = v
                # Mean impact across the x_i:
                df_scenario_model_errors["mu_x_i"] = df_scenario_model_errors[
                    scenario.x_cols()
                ].mean(axis="columns")
                yield df_scenario_model_errors

        df_model_errors = pd.concat(scenario_errors())

        return df_model_errors


class LinearWithNoiseExperiment(Experiment):

    def __init__(
        self,
        m: int | Iterable[int],
        s: int | Iterable[int],
        sigma: int | float | Iterable[float],
    ):
        if isinstance(m, int):
            m = [m]
        if isinstance(s, int):
            s = [s]
        if isinstance(sigma, (int, float)):
            sigma = [sigma]

        self._m = m
        self._s = s
        self._sigma = sigma

    def scenarios(
        self,
    ) -> Generator[Tuple[Dict[str, int | float], Scenario], None, None]:
        for sigma in self._sigma:
            for m in self._m:
                for s in self._s:
                    feature_generator = UniformFeatureGenerator(
                        s=s, m=m, low=0.0, high=100.0
                    )
                    target_generator = LinearNormalTargetGenerator(
                        a=np.linspace(-1.0, 1.0, m), b=0.0, sigma=sigma
                    )
                    scenario = Scenario(feature_generator, target_generator)
                    yield {"m": m, "s": s, "sigma": sigma}, scenario
