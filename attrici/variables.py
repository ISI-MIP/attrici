import numpy as np
from loguru import logger
from scipy import stats

from attrici import distributions
from attrici.estimation.model import AttriciGLM


def check_bounds(data, lower=None, upper=None):
    if lower is not None and data.min() < lower:
        raise ValueError(data.min(), "is smaller than lower bound", lower, ".")

    if upper is not None and data.max() > upper:
        raise ValueError(data.max(), "is bigger than upper bound", upper, ".")


def mask_thresholded(data, lower_threshold=None, upper_threshold=None):
    if lower_threshold is not None:
        logger.info(
            "Mask {} values below lower bound.", (data <= lower_threshold).sum().item()
        )
        data[data <= lower_threshold] = np.nan
    if upper_threshold is not None:
        logger.info(
            "Mask {} values above upper bound.", (data >= upper_threshold).sum().item()
        )
        data[data >= upper_threshold] = np.nan


def refill_and_rescale(scaled_data, scaling):
    # TODO implement refilling of values that have been masked before
    return scaled_data * scaling["scale"]


def scale_to_unity(data):
    datamin = data.min()
    scale = data.max() - datamin
    scaled_data = (data - datamin) / scale
    return scaled_data, {"datamin": datamin.item(), "scale": scale.item()}


def rescale_from_unity(scaled_data, scaling):
    """Use a given datamin and scale to rescale to original."""
    return scaled_data * scaling["scale"] + scaling["datamin"]


def check_units(data, units):
    if "units" in data.attrs:
        if isinstance(units, str):
            if data.units != units:
                raise ValueError(
                    f"Units of data are {data.units}, but expected {units}."
                )
        elif data.units not in units:
            raise ValueError(
                f"Units of data are {data.units}, but expected one of {units}."
            )
    else:
        logger.warning("No units attribute found in data.")


def identity(x):
    return x


def invlogit(x):
    return 1 / (1 + np.exp(-x))


class Variable:
    def quantile_mapping(self, distribution_ref, distribution_cfact):
        return distribution_cfact.invcdf(distribution_ref.cdf(self.y_scaled))


class Tas(Variable):
    """Influence of GMT is modelled through a shift of
    mu parameter in the Normal distribution.
    """

    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = scale_to_unity(data)

    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "K")

    def create_model(self, statistical_model_class, predictor, modes):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Normal,
            parameters={
                "mu": AttriciGLM.PredictorDependentParam(link=identity, modes=modes),
                "sigma": AttriciGLM.PredictorIndependentParam(link=np.exp, modes=modes),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
        )

    def rescale(self, scaled_data):
        return rescale_from_unity(scaled_data, self.scaling)


class Pr(Variable):
    THRESHOLD = 0.0000011574  # in kg m-2 s-1; "dry day" == 0.1mm/day / (86400s/day)

    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = self.scale(data)

    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "kg m-2 s-1")

    def scale(self, data):
        scaled_data = data - self.THRESHOLD
        logger.info(
            "Mask {} values below lower bound.", (scaled_data <= 0).sum().item()
        )
        scaled_data[scaled_data <= 0] = np.nan
        fa, _, fscale = stats.gamma.fit(scaled_data[~np.isnan(scaled_data)], floc=0)
        scale = fscale * fa**0.5
        scaled_data = scaled_data / scale

        logger.info(
            "Min, max after scaling: {}, {}",
            scaled_data.min().item(),
            scaled_data.max().item(),
        )
        return scaled_data, {"scale": scale.item()}

    def create_model(self, statistical_model_class, predictor, modes):
        observation = self.y_scaled.sel(time=predictor.time)

        return statistical_model_class(
            distribution=distributions.BernoulliGamma,
            parameters={
                "p": AttriciGLM.PredictorDependentParam(link=invlogit, modes=modes),
                "mu": AttriciGLM.PredictorDependentParam(link=np.exp, modes=modes),
                "nu": AttriciGLM.PredictorIndependentParam(link=np.exp, modes=modes),
            },
            observed=observation,
            predictor=predictor,
        )

    def quantile_mapping(self, distribution_ref, distribution_cfact):
        # make it a numpy array, so we can combine smoothly with d data frame.
        y = self.y_scaled.values.copy()
        dry_day = np.isnan(y)
        y[dry_day] = 0
        quantile = distribution_ref.cdf(y)

        # case of p smaller p'
        # the probability of a dry day is higher in the counterfactual day
        # than in the historical day. We need to create dry days.
        drier_cf = distribution_cfact.p > distribution_ref.p
        wet_to_wet = quantile > distribution_cfact.p  # False on dry day (NA in y)
        # if the quantile of the observed rain is high enough, keep day wet
        # and use normal quantile mapping
        do_normal_qm_0 = np.logical_and(drier_cf, wet_to_wet)
        logger.info(
            "Normal qm for higher cfact dry probability: {}",
            do_normal_qm_0.sum().item(),
        )
        cfact = np.zeros(len(y))
        cfact[do_normal_qm_0] = distribution_cfact.invcdf(quantile)[do_normal_qm_0]
        # else: make it a dry day with zero precip (from np.zeros)
        # case of p' smaller p
        # the probability of a dry day is lower in the counterfactual day
        # than in the historical day. We need to create wet days.
        wetter_cf = ~drier_cf
        wet_day = ~dry_day
        # wet days stay wet, and are normally quantile mapped
        do_normal_qm_1 = np.logical_and(wetter_cf, wet_day)
        logger.info(
            "Normal qm for higher cfact wet probability: {}",
            do_normal_qm_1.sum().item(),
        )
        cfact[do_normal_qm_1] = distribution_cfact.invcdf(quantile)[do_normal_qm_1]
        # some dry days need to be made wet. take a random quantile from
        # the quantile range that was dry days before
        random_dry_day_q = np.random.rand(len(y)) * distribution_ref.p
        map_to_wet = random_dry_day_q > distribution_cfact.p
        # map these dry days to wet, which are not dry in obs and
        # wet in counterfactual
        randomly_map_to_wet = np.logical_and(~do_normal_qm_1, map_to_wet)
        cfact[randomly_map_to_wet] = distribution_cfact.invcdf(quantile)[
            randomly_map_to_wet
        ]
        # else: leave zero (from np.zeros)
        logger.info("Days originally dry: {}", dry_day.sum().item())
        logger.info("Days made wet: {}", randomly_map_to_wet.sum().item())
        logger.info("Days in cfact dry: {}", (cfact == 0).sum().item())
        logger.info(
            "Total days: {}",
            (
                (cfact == 0).sum()
                + randomly_map_to_wet.sum()
                + do_normal_qm_0.sum()
                + do_normal_qm_1.sum()
            ).item(),
        )

        return cfact

    def rescale(self, scaled_data):
        # TODO implement refilling of values that have been masked before
        data = scaled_data * self.scaling["scale"] + self.THRESHOLD
        data[scaled_data <= 0] = 0
        return data


class Rlds(Variable):
    """Influence of GMT on longwave downwelling shortwave radiation
    is modelled through a shift of mu parameter in the Normal distribution.
    """

    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = scale_to_unity(data)

    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "W m-2")

    def create_model(self, statistical_model_class, predictor, modes):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Normal,
            parameters={
                "mu": AttriciGLM.PredictorDependentParam(link=identity, modes=modes),
                "sigma": AttriciGLM.PredictorIndependentParam(link=np.exp, modes=modes),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
        )

    def rescale(self, scaled_data):
        return rescale_from_unity(scaled_data, self.scaling)


class Ps(Variable):
    """Influence of GMT on sea level pressure (ps) is modelled through a shift of
    mu parameter in the Normal distribution.
    """

    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = scale_to_unity(data)

    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "Pa")

    def create_model(self, statistical_model_class, predictor, modes):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Normal,
            parameters={
                "mu": AttriciGLM.PredictorDependentParam(link=identity, modes=modes),
                "sigma": AttriciGLM.PredictorIndependentParam(link=np.exp, modes=modes),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
        )

    def rescale(self, scaled_data):
        return rescale_from_unity(scaled_data, self.scaling)


class Hurs(Variable):
    """Influence of GMT on relative humidity (hurs) is modelled with Beta
    regression as proposed in
    https://www.tandfonline.com/doi/abs/10.1080/0266476042000214501
    """

    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = self.scale(data)

    def validate(self, data):
        check_bounds(data, lower=0.0, upper=100.0)
        check_units(data, "%")

    def scale(self, data):
        mask_thresholded(data, lower_threshold=0.01, upper_threshold=99.99)
        scale = 100.0
        scaled_data = data / scale
        logger.info(
            "Min, max after scaling: {}, {}",
            scaled_data.min().item(),
            scaled_data.max().item(),
        )
        return scaled_data, {"scale": scale}

    def create_model(self, statistical_model_class, predictor, modes):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Beta,
            parameters={
                "mu": AttriciGLM.PredictorDependentParam(link=invlogit, modes=modes),
                "phi": AttriciGLM.PredictorIndependentParam(link=np.exp, modes=modes),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
        )

    def rescale(self, scaled_data):
        return refill_and_rescale(scaled_data, self.scaling)


class Tasskew(Variable):
    """Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, data):
        self.validate(data)
        self.y_scaled = data.copy()
        self.scaling = {}
        mask_thresholded(self.y_scaled, lower_threshold=0.0001, upper_threshold=0.9999)

    def validate(self, data):
        check_bounds(data, lower=0.0, upper=1.0)
        check_units(data, {"1", "K"})

    def create_model(self, statistical_model_class, predictor, modes):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Normal,
            parameters={
                "mu": AttriciGLM.PredictorDependentParam(link=identity, modes=modes),
                "sigma": AttriciGLM.PredictorIndependentParam(link=np.exp, modes=modes),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
        )

    def quantile_mapping(self, distribution_ref, distribution_cfact):
        """
        nan values are not quantile-mapped. 100% humidity happens mainly at the poles.
        """
        res = distribution_cfact.invcdf(distribution_ref.cdf(self.y_scaled))
        res[res >= 1] = np.nan
        res[res <= 0] = np.nan
        return res

    def rescale(self, scaled_data):
        return refill_and_rescale(scaled_data, {"scale": 1.0})


class Rsds(Variable):
    """Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Normal distribution.
    """

    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = scale_to_unity(data)

    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "W m-2")

    def create_model(self, statistical_model_class, predictor, modes):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Normal,
            parameters={
                "mu": AttriciGLM.PredictorDependentParam(link=identity, modes=modes),
                "sigma": AttriciGLM.PredictorIndependentParam(link=np.exp, modes=modes),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
        )

    def quantile_mapping(self, distribution_ref, distribution_cfact):
        """
        nan values are not quantile-mapped. 0 rsds happens mainly in the polar night.
        """
        res = distribution_cfact.invcdf(distribution_ref.cdf(self.y_scaled))
        res[res <= 0] = np.nan
        return res

    def rescale(self, scaled_data):
        return rescale_from_unity(scaled_data, self.scaling)


class RsdsWeibull(Variable):
    """Influence of GMT is modelled through a shift of
    the scale parameter beta in the Weibull distribution. The shape
    parameter alpha is assumed free of a trend.
    """

    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = scale_to_unity(data)

    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "W m-2")

    def create_model(self, statistical_model_class, predictor, modes):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Weibull,
            parameters={
                "alpha": AttriciGLM.PredictorDependentParam(link=np.exp, modes=modes),
                "beta": AttriciGLM.PredictorIndependentParam(link=np.exp, modes=modes),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
        )

    def rescale(self, scaled_data):
        return rescale_from_unity(scaled_data, self.scaling)


class Tasrange(Variable):
    """Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution. TODO fix comment not Beta
    """

    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = self.scale(data)

    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "K")

    def scale(self, data):
        mask_thresholded(data, lower_threshold=0.01)
        datamin = data.min()
        scale = data.max() - datamin
        scaled_data = data / scale
        logger.info(
            "Min, max after scaling: {}, {}",
            scaled_data.min().item(),
            scaled_data.max().item(),
        )
        return scaled_data, {"scale": scale}

    def create_model(self, statistical_model_class, predictor, modes):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Gamma,
            parameters={
                "mu": AttriciGLM.PredictorDependentParam(link=np.exp, modes=modes),
                "nu": AttriciGLM.PredictorIndependentParam(link=np.exp, modes=modes),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
        )

    def rescale(self, scaled_data):
        return refill_and_rescale(scaled_data, self.scaling)


class Wind(Variable):
    """Influence of GMT is modelled through a shift of
    the scale parameter beta in the Weibull distribution. The shape
    parameter alpha is assumed free of a trend.
    """

    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = self.scale(data)

    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "m s-1")

    def scale(self, data):
        mask_thresholded(data, lower_threshold=0.01)
        datamin = data.min()
        scale = data.max() - datamin
        scaled_data = data / scale
        logger.info(
            "Min, max after scaling: {}, {}",
            scaled_data.min().item(),
            scaled_data.max().item(),
        )
        return scaled_data, {"scale": scale}

    def create_model(self, statistical_model_class, predictor, modes):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Weibull,
            parameters={
                "alpha": AttriciGLM.PredictorIndependentParam(link=np.exp, modes=modes),
                "beta": AttriciGLM.PredictorDependentParam(link=np.exp, modes=modes),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
        )

    def rescale(self, scaled_data):
        return refill_and_rescale(scaled_data, self.scaling)


def create_variable(variable, data):
    MODEL_FOR_VAR = {
        "hurs": Hurs,
        "pr": Pr,
        "ps": Ps,
        "rlds": Rlds,
        "rsds": Rsds,
        "sfcWind": Wind,
        "tas": Tas,
        "tasrange": Tasrange,
        "tasskew": Tasskew,
        "wind": Wind,
    }
    if variable not in MODEL_FOR_VAR:
        raise ValueError(f"Variable {variable} not supported.")
    return MODEL_FOR_VAR[variable](data)
