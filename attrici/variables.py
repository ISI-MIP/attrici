"""
Climate variables.

This module contains classes for different climate variables used in the fitting and
detrending process. The variables are derived from the `Variable` base class.

See also Table 1 in [Mengel et al. (2021)](https://doi.org/10.5194/gmd-14-5269-2021).
"""

import numpy as np
from loguru import logger
from scipy import stats

from attrici import distributions
from attrici.estimation.model import AttriciGLM


def check_bounds(data, lower=None, upper=None):
    """
    Check if the elements in the data array are within the specified bounds.

    Parameters
    ----------
    data : xarray.DataArray
        Data to be checked.
    lower : float or int, optional
        Lower bound. If provided, raises a `ValueError` if any element in data is
        smaller.
    upper : float or int, optional
        Upper bound. If provided, raises a `ValueError` if any element in data is
        larger.
    """
    if lower is not None and data.min() < lower:
        raise ValueError(data.min(), "is smaller than lower bound", lower, ".")

    if upper is not None and data.max() > upper:
        raise ValueError(data.max(), "is bigger than upper bound", upper, ".")


def mask_thresholded(data, lower_threshold=None, upper_threshold=None):
    """
    Mask elements in the data that are outside the specified thresholds.

    Parameters
    ----------
    data : xarray.DataArray
        Data to be masked.
    lower_threshold : float or int, optional
        Elements in data that are less than or equal to the lower threshold will
         be masked as NaN.
    upper_threshold : float or int, optional
        Elements in data that are greater than or equal to the upper threshold will
        be masked as NaN.

    Notes
    -----
    The function logs the number of elements being masked for each threshold.
    """
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
    """
    Rescale the data according to the given scaling information.

    Parameters
    ----------
    scaled_data : xarray.DataArray
        Data to be rescaled.
    scaling : dict
        Dictionary containing the scaling information - must contain the key "scale".

    Returns
    -------
    xarray.DataArray
        The rescaled data.
    """
    # refilling of values that have been masked before is done by the calling function
    # later (see `attrici.detrend`)
    return scaled_data * scaling["scale"]


def scale_to_unity(data):
    """
    Scale data linearly to unity range (map min to 0, max to 1).

    Parameters
    ----------
    data : xarray.DataArray
        Data to be scaled.

    Returns
    -------
    xarray.DataArray
        The scaled data.
    dict
        Dictionary containing the scaling information - keys are "datamin" and "scale".
    """
    datamin = data.min()
    scale = data.max() - datamin
    scaled_data = (data - datamin) / scale
    return scaled_data, {"datamin": datamin.item(), "scale": scale.item()}


def rescale_from_unity(scaled_data, scaling):
    """
    Rescale data from unity range to original range.

    Parameters
    ----------
    scaled_data : xarray.DataArray
        Data to be rescaled.
    scaling : dict
        Dictionary containing the scaling information - must contain the keys "datamin"
        and "scale".

    Returns
    -------
    xarray.DataArray
        The rescaled data.
    """
    return scaled_data * scaling["scale"] + scaling["datamin"]


def check_units(data, units):
    """
    Check if the units of the data are as expected.

    Parameters
    ----------
    data : xarray.DataArray
        Data to be checked.
    units : str or set
        The expected units (or a set of valid units).
    """
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
    """
    Identity link function.

    Parameters
    ----------
    x : array_like
        The input data.

    Returns
    -------
    array_like
        The input data.
    """
    return x


def invlogit(x):
    """
    Inverse logit link function.

    Parameters
    ----------
    x : array_like
        The input data.

    Returns
    -------
    array_like
        The inverse logit of the input data.
    """
    return 1 / (1 + np.exp(-x))


class Variable:
    """
    Climate variable base class.

    Sub-classes need to implement `create_model` and `rescale` methods and might
    overwrite the default `quantile_mapping` function. A custom validation function
    should be called in `__init__`

    Attributes
    ----------
    y_scaled : xarray.DataArray
        Rescaled input data.
    datamin : float, optional
        (Optional) minimum value of original data, used for rescaling.
    scale : float, optional
        (Optional) range between minimum and maximum value, used for rescaling.
    """

    def __init__(self, data):
        """
        Initialise a climate variable.

        Parameters
        ----------
        data: xarray.DataArray
            Observation data
        """
        raise NotImplementedError

    def validate(self, data):
        """
        Validate the input data, e.g. bounds and units.

        Parameters
        ----------
        data: xarray.DataArray
            Observation data
        """
        raise NotImplementedError

    def create_model(self, statistical_model, predictor, **kwargs):
        """
        Create a model.

        Parameters
        ----------
        statistical_model : class
            The statistical model class to be used for creating the model - should be
            a subclass of `attrici.estimation.model.AttriciGLM`.
        predictor : xarray.DataArray
            The predictor data used for the model.
        **kwargs
            Additional keyword arguments for the model.

        Returns
        -------
        attrici.estimation.model.Model
            An instance of the created model.
        """
        raise NotImplementedError

    def rescale(self, scaled_data):
        """
        Rescale data.

        Parameters
        ----------
        scaled_data : xarray.DataArray
            The data to be rescaled.

        Returns
        -------
        xarray.DataArray
            The rescaled data.
        """
        raise NotImplementedError

    def quantile_mapping(self, distribution_ref, distribution_cfact):
        """
        Map data to respective quantile in the reference distribution.

        Parameters
        ----------
        distribution_ref : Distribution
            The reference distribution.
        distribution_cfact : Distribution
            The counterfactual distribution.

        Returns
        -------
        xarray.DataArray
            Data mapped to the respective quantile in the reference distribution.
        """
        return distribution_cfact.invcdf(distribution_ref.cdf(self.y_scaled))


class Tas(Variable):
    """
    Daily mean near-surface air temperature (tas), modelled by a Normal distribution
    per time step.
    """

    # docstr-coverage:inherited
    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = scale_to_unity(data)

    # docstr-coverage:inherited
    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "K")

    # docstr-coverage:inherited
    def create_model(self, statistical_model_class, predictor, **kwargs):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Normal,
            parameters={
                "mu": AttriciGLM.Parameter(link=identity, dependent=True),
                "sigma": AttriciGLM.Parameter(link=np.exp, dependent=False),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
            **kwargs,
        )

    # docstr-coverage:inherited
    def rescale(self, scaled_data):
        return rescale_from_unity(scaled_data, self.scaling)


class Pr(Variable):
    """
    Precipitation (pr), modelled by a Bernoulli-Gamma distribution per time step.

    - For wet or dry day: Bernoulli with dry-day probability `p`
    - For intensity of precipitation on wet days: Gamma with mean value `mu`
      and shape parameter `nu`.
    """

    THRESHOLD = 0.0000011574
    """Dry day threshold in kg m-2 s-1; "dry day" == 0.1mm/day / (86400s/day)"""

    # docstr-coverage:inherited
    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = self.scale(data)

    # docstr-coverage:inherited
    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "kg m-2 s-1")

    def scale(self, data):
        """
        Precipitation specific scaling function.

        Parameters
        ----------
        data : xarray.DataArray
            Original data

        Returns
        -------
        xarray.DataArray
            The scaled data.
        dict
            The scaling information - contains the key "scale".
        """
        scaled_data = data - self.THRESHOLD
        logger.info(
            "Mask {} values below or at dry day threshold.",
            (scaled_data <= 0).sum().item(),
        )
        scaled_data[scaled_data <= 0] = np.nan
        if not scaled_data.notnull().any().item():
            return scaled_data, {"scale": np.nan}
        try:
            fa, _, fscale = stats.gamma.fit(
                scaled_data[~np.isnan(scaled_data)],
                floc=0,
            )
            scale = fscale * fa**0.5
            scaled_data = scaled_data / scale
        except ValueError as e:
            logger.warning("Failed to fit to derive scaling: {}", e)
            return scaled_data, {"scale": np.nan}

        logger.info(
            "Min, max after scaling: {}, {}",
            scaled_data.min().item(),
            scaled_data.max().item(),
        )
        return scaled_data, {"scale": scale.item()}

    # docstr-coverage:inherited
    def create_model(self, statistical_model_class, predictor, **kwargs):
        observation = self.y_scaled.sel(time=predictor.time)

        return statistical_model_class(
            distribution=distributions.BernoulliGamma,
            parameters={
                "p": AttriciGLM.Parameter(link=invlogit, dependent=True),
                "mu": AttriciGLM.Parameter(link=np.exp, dependent=True),
                "nu": AttriciGLM.Parameter(link=np.exp, dependent=False),
            },
            observed=observation,
            predictor=predictor,
            **kwargs,
        )

    # docstr-coverage:inherited
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

    # docstr-coverage:inherited
    def rescale(self, scaled_data):
        data = scaled_data * self.scaling["scale"] + self.THRESHOLD
        data[scaled_data <= 0] = 0
        return data


class Rlds(Variable):
    """
    Surface downwelling longwave radiation (rlds), modelled by a Normal distribution
    per time step.
    """

    # docstr-coverage:inherited
    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = scale_to_unity(data)

    # docstr-coverage:inherited
    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "W m-2")

    # docstr-coverage:inherited
    def create_model(self, statistical_model_class, predictor, **kwargs):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Normal,
            parameters={
                "mu": AttriciGLM.Parameter(link=identity, dependent=True),
                "sigma": AttriciGLM.Parameter(link=np.exp, dependent=False),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
            **kwargs,
        )

    # docstr-coverage:inherited
    def rescale(self, scaled_data):
        return rescale_from_unity(scaled_data, self.scaling)


class Ps(Variable):
    """
    Surface air pressure (ps), modelled by a Normal distribution per time step.
    """

    # docstr-coverage:inherited
    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = scale_to_unity(data)

    # docstr-coverage:inherited
    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "Pa")

    # docstr-coverage:inherited
    def create_model(self, statistical_model_class, predictor, **kwargs):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Normal,
            parameters={
                "mu": AttriciGLM.Parameter(link=identity, dependent=True),
                "sigma": AttriciGLM.Parameter(link=np.exp, dependent=False),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
            **kwargs,
        )

    # docstr-coverage:inherited
    def rescale(self, scaled_data):
        return rescale_from_unity(scaled_data, self.scaling)


class Hurs(Variable):
    """
    Near-surface relative humidity (hurs), modelled by a Beta distribution per time
    step as proposed in https://doi.org/10.1080/0266476042000214501.
    """

    # docstr-coverage:inherited
    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = self.scale(data)

    # docstr-coverage:inherited
    def validate(self, data):
        check_bounds(data, lower=0.0, upper=100.0)
        check_units(data, "%")

    def scale(self, data):
        """
        Hurs-specific scaling funtion.

        Parameters
        ----------
        data: xarray.DataArray
            Input data to be scaled.
        """
        mask_thresholded(data, lower_threshold=0.01, upper_threshold=99.99)
        scale = 100.0
        scaled_data = data / scale
        logger.info(
            "Min, max after scaling: {}, {}",
            scaled_data.min().item(),
            scaled_data.max().item(),
        )
        return scaled_data, {"scale": scale}

    # docstr-coverage:inherited
    def create_model(self, statistical_model_class, predictor, **kwargs):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Beta,
            parameters={
                "mu": AttriciGLM.Parameter(link=invlogit, dependent=True),
                "phi": AttriciGLM.Parameter(link=np.exp, dependent=False),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
            **kwargs,
        )

    # docstr-coverage:inherited
    def rescale(self, scaled_data):
        return refill_and_rescale(scaled_data, self.scaling)


class Tasskew(Variable):
    """
    Daily near-surface temperatureskewness (tasskew), modelled by a Normal
    distribution per time step.
    """

    # docstr-coverage:inherited
    def __init__(self, data):
        self.validate(data)
        self.y_scaled = data.copy()
        self.scaling = {}
        mask_thresholded(self.y_scaled, lower_threshold=0.0001, upper_threshold=0.9999)

    # docstr-coverage:inherited
    def validate(self, data):
        check_bounds(data, lower=0.0, upper=1.0)
        check_units(data, {"1", "K"})

    # docstr-coverage:inherited
    def create_model(self, statistical_model_class, predictor, **kwargs):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Normal,
            parameters={
                "mu": AttriciGLM.Parameter(link=identity, dependent=True),
                "sigma": AttriciGLM.Parameter(link=np.exp, dependent=False),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
            **kwargs,
        )

    # docstr-coverage:inherited
    def quantile_mapping(self, distribution_ref, distribution_cfact):
        """
        nan values are not quantile-mapped. 100% humidity happens mainly at the
        poles.
        """
        res = distribution_cfact.invcdf(distribution_ref.cdf(self.y_scaled))
        res[res >= 1] = np.nan
        res[res <= 0] = np.nan
        return res

    # docstr-coverage:inherited
    def rescale(self, scaled_data):
        return refill_and_rescale(scaled_data, {"scale": 1.0})


class Rsds(Variable):
    """
    Surface downwelling shortwave radiation (rsds), modelled by a Normal distribution
    per time.
    """

    # docstr-coverage:inherited
    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = scale_to_unity(data)

    # docstr-coverage:inherited
    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "W m-2")

    # docstr-coverage:inherited
    def create_model(self, statistical_model_class, predictor, **kwargs):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Normal,
            parameters={
                "mu": AttriciGLM.Parameter(link=identity, dependent=True),
                "sigma": AttriciGLM.Parameter(link=np.exp, dependent=False),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
            **kwargs,
        )

    # docstr-coverage:inherited
    def quantile_mapping(self, distribution_ref, distribution_cfact):
        """
        nan values are not quantile-mapped. 0 rsds happens mainly in the polar
        night.
        """
        res = distribution_cfact.invcdf(distribution_ref.cdf(self.y_scaled))
        res[res <= 0] = np.nan
        return res

    # docstr-coverage:inherited
    def rescale(self, scaled_data):
        return rescale_from_unity(scaled_data, self.scaling)


class Tasrange(Variable):
    """
    Daily near-surface temperature range (tasrange), modelled by a Gamma distribution
    per time step.
    """

    # docstr-coverage:inherited
    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = self.scale(data)

    # docstr-coverage:inherited
    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "K")

    # docstr-coverage:inherited
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

    # docstr-coverage:inherited
    def create_model(self, statistical_model_class, predictor, **kwargs):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Gamma,
            parameters={
                "mu": AttriciGLM.Parameter(link=np.exp, dependent=True),
                "nu": AttriciGLM.Parameter(link=np.exp, dependent=False),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
            **kwargs,
        )

    # docstr-coverage:inherited
    def rescale(self, scaled_data):
        return refill_and_rescale(scaled_data, self.scaling)


class Wind(Variable):
    """
    Near-surface wind speed (sfcwind), modelled by a Weibull distribution per time
    step.
    """

    # docstr-coverage:inherited
    def __init__(self, data):
        self.validate(data)
        self.y_scaled, self.scaling = self.scale(data)

    # docstr-coverage:inherited
    def validate(self, data):
        check_bounds(data, lower=0.0)
        check_units(data, "m s-1")

    # docstr-coverage:inherited
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

    # docstr-coverage:inherited
    def create_model(self, statistical_model_class, predictor, **kwargs):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Weibull,
            parameters={
                "alpha": AttriciGLM.Parameter(link=np.exp, dependent=False),
                "beta": AttriciGLM.Parameter(link=np.exp, dependent=True),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
            **kwargs,
        )

    # docstr-coverage:inherited
    def rescale(self, scaled_data):
        return refill_and_rescale(scaled_data, self.scaling)


def create_variable(variable, data):
    """
    Returns a Variable instance based on a string abbreviation.

    The variable abbreviation is mapped to the respective Variable class.
    See also `attrici.detrend`.

    Parameters
    ----------
    variable : str
        Short variable name like `tas`.
    data : xarray.DataArray
        Observation data for the variable

    Returns
    -------
    Variable
        An instance of the corresponding Variable class.
    """
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
