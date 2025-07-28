# Bootstrapping Usage Manual

## Motivation

The block‑bootstrap method implemented in the ATTRICI toolbox is designed to assess the uncertainty of the estimated trends:

- **Low spread** among bootstrap samples ⇒ a stable, well‑constrained trend estimate.
- **High spread** ⇒ a potential mismatch between the assumed GMT‑driven trend and the data, or hidden issues such as biases and spurious trends.

By mapping the bootstrap spread across space we can highlight areas where the model performs well versus areas requiring caution or further investigation.

## Method

We adopt a **block‑bootstrap algorithm** ([Mudelsee, 2019\)](https://doi.org/10.1016/j.earscirev.2018.12.005) tailored to autocorrelated climate time‑series with serial correlation. To account for non-negative data such as precipitation we use a variation where resampling is done in the percentile space instead of using residuals. We use a block size of 1 year to ensure similar seasonality in each of the blocks.

## Algorithm

Let **X** be the time‑series for a single climate variable.

1. **Fit the base model**  
   Train the GMT‑dependent parametric probability distribution on $\bf{X}$.
2. **Transform to percentiles**  
   For every data point $x \in X$ compute its percentile under the learned $CDF$:

   $$
   Z = \lbrace CDF(x) | x \in X \rbrace.
   $$

3. **Resample blocks**  
   Draw yearly blocks with replacement from $\bf{Z}$ to create $N$ bootstrap samples $\hat{Z}_i$.
4. **Back‑transform**  
   For each sample, invert the CDF to return to data space:

   $$
   \hat{X}_i = \lbrace CDF^{-1}(z) \mid z \in \hat{Z}_i \rbrace.
   $$

5. **Refit the model**  
   Train $N$ alternative models on the bootstrap series $\hat{X}_i$.
6. **Summarise**  
   For each calendar time, compute the expected value across the ensemble and derive statistics (e.g. 5th, 50th, 95th percentiles; standard deviation).

The distribution of these ensemble trends constitutes the **uncertainty envelope** around the original fit.

## Usage

The different bootstrap samples and associated parameters can be used to derive confidence intervals for the learned model / trend. A high spread between the bootstrap samples indicate poor fit on the given data and large remaining trends in the residuals. Bootstrapping can be performed on a global subset of the provided data (e.g. every 10th grid cell in each direction), which is often enough to highlight regions where the model assumptions are (not) fitting well with the data.

## References

[Mudelsee, M.: Trend analysis of climate time series: A review of methods, Earth-Sci. Rev., 190, 310–322, https://doi.org/10.1016/j.earscirev.2018.12.005, 2019\.](https://doi.org/10.1016/j.earscirev.2018.12.005)
