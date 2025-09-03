<!--- pyml disable-next-line line-length, first-line-h1 -->
[![GitHub Repository](https://img.shields.io/badge/GitHub-blue?style=for-the-badge&logo=github&logoColor=white&labelColor=%23555555&color=%23838996)](https://github.com/ISI-MIP/attrici) [![Docs](https://img.shields.io/badge/Docs-%23ff8c00?style=for-the-badge)](https://isi-mip.github.io/attrici) [![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ISI-MIP/attrici/ci.yml?style=for-the-badge)](https://github.com/ISI-MIP/attrici/actions) [![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FISI-MIP%2Fattrici%2Fmain%2Fpyproject.toml&style=for-the-badge)](https://github.com/ISI-MIP/attrici/blob/main/pyproject.toml) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=for-the-badge)](https://github.com/astral-sh/ruff)

# ATTRICI - counterfactual climate for impact attribution

Code implementing the methods described in the paper
[`ATTRICI 1.1 - counterfactual climate for impact attribution`](https://doi.org/10.5194/gmd-14-5269-2021)
in Geoscientific Model Development.

From the article's abstract:

> We here present ATTRICI (ATTRIbuting Climate Impacts), an approach to construct the required counterfactual
> stationary climate data from observational (factual) climate data. Our method identifies the long-term shifts
> in the considered daily climate variables that are correlated to global mean temperature change assuming a smooth
> annual cycle of the associated scaling coefficients for each day of the year. The produced counterfactual climate
> datasets are used as forcing data within the impact attribution setup of the Inter-Sectoral Impact Model
> Intercomparison Project (ISIMIP3a). Our method preserves the internal variability of the observed data in the sense
> that factual and counterfactual data for a given day have the same rank in their respective statistical distributions.

<https://doi.org/10.5194/gmd-14-5269-2021>

The code accompanying the manuscript is archived at [Zenodo](https://doi.org/10.5281/zenodo.3828914).

This repository contains further updates to the code. It includes optimization routines based
on the originally used [PyMC3](https://pypi.org/project/pymc3/) (PyMC3 is no longer maintained but included for
reproducibility), an updated version using [PyMC5](https://www.pymc.io) and a version based on [Scipy](https://scipy.org/).

[Documentation](https://isi-mip.github.io/attrici) is available online.
It is build for the latest commit in the current main branch.
An overview of the model architecture is in [OVERVIEW.md](https://github.com/isi-mip/attrici/tree/main/OVERVIEW.md).
See [USAGE.md](https://github.com/isi-mip/attrici/tree/main/USAGE.md) for getting started using ATTRICI.
See also the [notebooks](https://github.com/isi-mip/attrici/tree/main/notebooks) folder for further examples.

## Credits

ATTRICI uses the [PyMC](https://www.pymc.io/) package for probabilistic programming (Salvatier et al. 2016).

An early version of the code on Bayesian estimation of parameters in timeseries with periodicity in PyMC3
was inspired by [Ritchie Vink's](https://www.ritchievink.com)
[post](https://www.ritchievink.com/blog/2018/10/09/build-facebooks-prophet-in-pymc3-bayesian-time-series-analyis-with-generalized-additive-models/)
on Bayesian timeseries analysis with additive models.

The Singular Spectrum Analysis included in ATTRICI comes from [pyts](https://pyts.readthedocs.io/en/stable/generated/pyts.decomposition.SingularSpectrumAnalysis.html).

## Funding

![EU_logo](https://github.com/user-attachments/assets/e2fad699-697e-43fd-84be-032447d6dd21) This project has received funding from the European Union's HORIZON Research and Innovation Actions Programme under Grant Agreement No. 101135481 (COMPASS).

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or of the European Health and Digital Executive Agency (HADEA). Neither the European Union nor the granting authority HADEA can be held responsible for them.

## License

This code is licensed under GPLv3, see the [LICENSE.txt](https://github.com/ISI-MIP/attrici/blob/main/LICENSE.txt).
See commit history for authors.
