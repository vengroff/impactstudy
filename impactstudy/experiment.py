"""Experiments."""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Generator, Iterable, Optional
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
from sklearn.linear_model import LinearRegression

from impactchart.model import ImpactModel, XGBoostImpactModel


class NoiseGenerator(ABC):
    def __init__(self, seed: int | RandomState):
        self._rng = default_rng(seed=seed)

    @abstractmethod
    def z(self, size: int) -> pd.Series:
        """The noise $Z$."""
        raise NotImplementedError("Not implemented on abstract class.")


class NormalNoiseGenerator(NoiseGenerator):
    def __init__(self, sigma: float, seed: int | RandomState):
        super().__init__(seed)
        self._sigma = sigma

    @property
    def sigma(self) -> float:
        return self._sigma

    @cache
    def z(self, size: int) -> pd.Series:
        return pd.Series(self._rng.normal(0, self.sigma, size=size), name="z")


class ExactTargetGenerator(ABC):

    @abstractmethod
    def arity(self) -> int:
        raise NotImplementedError("Not implemented on abstract class.")

    @abstractmethod
    def f(self, x: pd.DataFrame) -> pd.Series:
        """The function $F$ from features to target."""
        raise NotImplementedError("Not implemented on abstract class.")

    @abstractmethod
    def impact(self, x: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Not implemented on abstract class.")


class TargetGenerator:
    def __init__(
        self,
        exact_target_generator: ExactTargetGenerator,
        noise: Optional[NoiseGenerator] = None,
    ):
        self._exact_target_generator = exact_target_generator
        self._noise = noise

    def f_prime(self, x: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
        f = self._exact_target_generator.f(x)

        if self._noise is None:
            return f

        z = self._noise.z(size=x.shape[0])
        return f + z

    def impact(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._exact_target_generator.impact(x)


class LinearExactTargetGenerator(ExactTargetGenerator):
    def __init__(self, a: np.array, b: float):
        self._a = a
        self._b = b

    def arity(self) -> int:
        return len(self._a)

    @property
    def a(self) -> np.array:
        return self._a

    @property
    def b(self) -> float:
        return self._b

    def f(self, df_x: pd.DataFrame) -> pd.Series:
        df_ax = df_x.mul(self._a, axis="columns")

        f = df_ax.sum(axis="columns") + self._b
        f.rename("f", inplace=True)
        return f

    def impact(self, x: pd.DataFrame) -> pd.DataFrame:
        term = np.multiply(self._a, x)
        impact = term - term.mean()

        return impact


class StepExactTargetGenerator(ExactTargetGenerator):
    def __init__(self, step_at: float, step_size: float, step_base: float = 0.0):
        self._step_at = step_at
        self._step_size = step_size
        self._step_base = step_base

    def arity(self) -> int:
        return 1

    def f(self, df_x: pd.DataFrame) -> pd.Series:
        return pd.Series(
            np.where(
                df_x["x_0"] < self._step_at,
                self._step_base,
                self._step_base + self._step_size,
            ),
            index=df_x.index,
            name="f",
        )

    def impact(self, x: pd.DataFrame) -> pd.DataFrame:
        f = self.f(x)
        df_impact = pd.DataFrame(f).rename({"f": "x_0"}, axis="columns")
        return df_impact


def add_normal_noise(
    exact_target_generator: ExactTargetGenerator,
    sigma: float,
    seed: int | RandomState = 17,
) -> TargetGenerator:
    normal_noise_generator = NormalNoiseGenerator(sigma, seed=seed)

    return TargetGenerator(exact_target_generator, normal_noise_generator)


class AdditiveExactTargetGenerator(ExactTargetGenerator):

    def __init__(self, exact_target_generators: Iterable[ExactTargetGenerator]):
        self._exact_target_generators = exact_target_generators

    def arity(self) -> int:
        return sum(tg.arity() for tg in self._exact_target_generators)

    def f(self, x: pd.DataFrame) -> pd.Series:
        offset = 0
        f = 0.0

        for tg in self._exact_target_generators:
            sub_x = pd.DataFrame(x[[f"x_{offset + ii}" for ii in range(tg.arity())]])
            sub_x = sub_x.rename(
                {f"x_{offset + ii}": f"x_{ii}" for ii in range(tg.arity())},
                axis="columns",
            )
            f += tg.f(sub_x)
            offset += tg.arity()

        return f

    def impact(self, x: pd.DataFrame) -> pd.DataFrame:
        offset = 0
        df_impact = pd.DataFrame()

        for tg in self._exact_target_generators:
            df_sub_x = pd.DataFrame(x[[f"x_{offset + ii}" for ii in range(tg.arity())]])
            df_sub_x.rename(
                {f"x_{offset + ii}": f"x_{ii}" for ii in range(tg.arity())},
                axis="columns",
                inplace=True,
            )
            df_sub_impact = tg.impact(df_sub_x)
            df_sub_impact.rename(
                {f"x_{ii}": f"x_{offset + ii}" for ii in range(tg.arity())},
                axis="columns",
                inplace=True,
            )
            offset += tg.arity()

            df_impact = pd.concat([df_impact, df_sub_impact], axis="columns")

        return df_impact


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
    def __call__(self, n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    def training_data(self, n: int) -> pd.DataFrame:
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
        df = self.training_data(n)

        df_X_prime = df[self.x_prime_cols()]
        y = df[self.y_col()]

        impact_model = XGBoostImpactModel(random_state=self._impact_model_seed)
        impact_model.fit(df_X_prime, y)

        return impact_model

    @cache
    def model_mean_impact(self, n: int) -> pd.DataFrame:
        impact_model = self.impact_model(n)

        df = self.training_data(n)
        df_X_prime = df[self.x_prime_cols()]

        return impact_model.mean_impact(df_X_prime)

    def model_impact_charts(
        self, n: int, linreg_overlay: bool = False
    ) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        impact_model = self.impact_model(n)

        df = self.training_data(n)
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
                c="orange",
                marker=".",
                s=10,
                zorder=10,
                label="Actual impact",
            )

            if linreg_overlay:
                df_linreg_impact = self.linreg_impact(n, col)
                ax.scatter(
                    df_linreg_impact[col],
                    df_linreg_impact["y_hat"],
                    c="purple",
                    marker=".",
                    s=10,
                    zorder=9,
                    label="Linear regression impact",
                )

        return impact_charts

    def root_mean_squared_error(
        self, n: int, df_model_impact: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        df_true_impact = self.true_impact(n)
        if df_model_impact is None:
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

    def mean_absolute_error(
        self, n: int, df_model_impact: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        df_true_impact = self.true_impact(n)
        if df_model_impact is None:
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

    def median_absolute_error(
        self, n: int, df_model_impact: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        df_true_impact = self.true_impact(n)
        if df_model_impact is None:
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

    def model_errors(self, n: int, *, linreg_errors: bool = False) -> pd.DataFrame:
        df_rmse = self.root_mean_squared_error(n)
        df_mae = self.mean_absolute_error(n)
        df_medae = self.median_absolute_error(n)

        df_rmse["metric"] = "RMSE"
        df_mae["metric"] = "MAE"
        df_medae["metric"] = "MED_AE"

        if linreg_errors:
            df_linreg_impact = self.linreg_impacts(n)

            df_linreg_rmse = self.root_mean_squared_error(n, df_linreg_impact)
            df_linreg_mae = self.mean_absolute_error(n, df_linreg_impact)
            df_linreg_medae = self.median_absolute_error(n, df_linreg_impact)

            df_linreg_rmse["metric"] = "LR_RMSE"
            df_linreg_mae["metric"] = "LR_MAE"
            df_linreg_medae["metric"] = "LR_MED_AE"

            df_errors = pd.concat(
                [
                    df_rmse,
                    df_mae,
                    df_medae,
                    df_linreg_rmse,
                    df_linreg_mae,
                    df_linreg_medae,
                ],
                axis="rows",
            )
        else:
            df_errors = pd.concat([df_rmse, df_mae, df_medae], axis="rows")

        df_errors = df_errors[
            ["metric"] + [col for col in df_errors.columns if col != "metric"]
        ]

        # Mean impact across the x_i:
        df_errors["mu_x_i"] = df_errors[self.x_cols()].mean(axis="columns")

        # Mean impact across the c_i:
        if len(self.c_cols()) > 0:
            df_errors["mu_c_i"] = df_errors[self.c_cols()].mean(axis="columns")

        return df_errors

    @cache
    def linreg_model(self, n: int) -> LinearRegression:
        linreg = LinearRegression()

        df_training = self.training_data(n)

        linreg.fit(df_training[self.x_prime_cols()], df_training[self.y_col()])

        return linreg

    def y_hat_linreg(self, n: int) -> pd.Series:
        return self.linreg_model(n).predict(self.training_data(n)[self.x_prime_cols()])

    def linreg_impacts(self, n: int) -> pd.DataFrame:
        df_linreg_impacts = pd.concat(
            (
                self.linreg_impact(n, feature)["y_hat"].rename(feature)
                for feature in self.x_prime_cols()
            ),
            axis="columns",
        )

        return df_linreg_impacts

    def linreg_impact(self, n: int, feature: str) -> pd.DataFrame:
        # Create a data frame that has the mean value for every
        # column except the feature we are interested in, where
        # the values remain the original.
        df_training = self.training_data(n)

        df_mean_x_prime = df_training[self.x_prime_cols()].copy()
        for col in df_mean_x_prime.columns:
            if col != feature:
                df_mean_x_prime[col] = df_mean_x_prime[col].mean()

        mean_y = df_training[self.y_col()].mean()

        df_mean_x_prime["y_hat"] = (
            self.linreg_model(n).predict(df_mean_x_prime) - mean_y
        )

        return df_mean_x_prime[[feature, "y_hat"]]


class Experiment(ABC):

    @abstractmethod
    def scenarios(
        self,
    ) -> Generator[Tuple[Dict[str, int | float], Scenario], None, None]:
        raise NotImplementedError("Not implemented on abstract class.")

    def model_errors(
        self, n: int, *, linreg_errors: Optional[bool] = False
    ) -> pd.DataFrame:
        def scenario_errors() -> Generator[pd.DataFrame, None, None]:
            for tags, scenario in self.scenarios():
                df_scenario_model_errors = scenario.model_errors(
                    n, linreg_errors=linreg_errors
                )
                for k, v in tags.items():
                    df_scenario_model_errors[k] = v

                yield df_scenario_model_errors

        df_model_errors = pd.concat(scenario_errors())

        return df_model_errors


class LinearWithNoiseExperiment(Experiment):

    def __init__(
        self,
        m: int | Iterable[int],
        s: int | Iterable[int],
        sigma: int | float | Iterable[float],
        seed: Optional[int] = 17,
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
        self._seed = seed

    def scenarios(
        self,
    ) -> Generator[Tuple[Dict[str, int | float], Scenario], None, None]:
        for sigma in self._sigma:
            for m in self._m:
                for s in self._s:
                    feature_generator = UniformFeatureGenerator(
                        s=s, m=m, low=0.0, high=100.0, seed=self._seed
                    )

                    linear_exact_target_generator = LinearExactTargetGenerator(
                        a=np.linspace(-1.0, 1.0, m), b=0.0
                    )
                    target_generator = add_normal_noise(
                        linear_exact_target_generator,
                        sigma,
                        seed=(17 * self._seed) % 0x7FFFFFFF,
                    )

                    scenario = Scenario(feature_generator, target_generator)
                    yield {"m": m, "s": s, "sigma": sigma}, scenario


class LinearAndStepWithNoiseExperiment(Experiment):

    def __init__(
        self,
        m_linear: int | Iterable[int],
        m_step: int | Iterable[int],
        s: int | Iterable[int],
        sigma: int | float | Iterable[float],
        seed: Optional[int] = 17,
    ):
        if isinstance(m_linear, int):
            m_linear = [m_linear]
        if isinstance(m_step, int):
            m_step = [m_step]
        if isinstance(s, int):
            s = [s]
        if isinstance(sigma, (int, float)):
            sigma = [sigma]

        self._m_linear = m_linear
        self._m_step = m_step
        self._s = s
        self._sigma = sigma
        self._seed = seed

    def scenarios(
        self,
    ) -> Generator[Tuple[Dict[str, int | float], Scenario], None, None]:
        for sigma in self._sigma:
            for m_linear in self._m_linear:
                for m_step in self._m_step:
                    for s in self._s:
                        feature_generator = UniformFeatureGenerator(
                            s=s,
                            m=m_linear + m_step,
                            low=0.0,
                            high=100.0,
                            seed=self._seed,
                        )

                        if m_linear > 0:
                            linear_exact_target_generator = LinearExactTargetGenerator(
                                a=np.linspace(-1.0, 1.0, m_linear), b=0.0
                            )
                        else:
                            linear_exact_target_generator = None

                        if m_step > 0:
                            step_exact_target_generators = [
                                StepExactTargetGenerator(
                                    50.0, 100.0 - 10.0 * ii, -50.0 + 5.0 * ii
                                )
                                for ii in range(m_step)
                            ]
                            steps_exact_target_generator = AdditiveExactTargetGenerator(
                                step_exact_target_generators
                            )
                        else:
                            steps_exact_target_generator = None

                        if linear_exact_target_generator is None:
                            exact_target_generator = steps_exact_target_generator
                        elif steps_exact_target_generator is None:
                            exact_target_generator = linear_exact_target_generator
                        else:
                            exact_target_generator = AdditiveExactTargetGenerator(
                                [
                                    steps_exact_target_generator,
                                    linear_exact_target_generator,
                                ]
                            )

                        target_generator = add_normal_noise(
                            exact_target_generator,
                            sigma,
                            seed=(17 * self._seed) % 0x7FFFFFFF,
                        )

                        scenario = Scenario(feature_generator, target_generator)
                        yield {
                            "m_linear": m_linear,
                            "m_step": m_step,
                            "s": s,
                            "sigma": sigma,
                        }, scenario
