#! /usr/bin/env python3

import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product as it_product

from cmdstanpy import CmdStanModel

import arviz as az
import matplotlib.pyplot as plt


def calculate_gaussian_kernel_density_bandwidth_silverman_rule(
    df: pd.DataFrame) -> pd.Series:
    """
    Calculate Gaussian kernel density bandwidth based on Silverman's rule from:
        Silverman, B. W. (1986).  Density Estimation for Statistics and Data
            Analysis.  London: Chapman & Hall/CRC. p. 45
            ISBN 978-0-412-24620-3

    Wikipedia is a useful reference:

        https://en.wikipedia.org/wiki/Kernel_density_estimation

    :param df: a Pandas DataFrame where the Gaussian kernel density will be
        calculated for each column
    :return: scalar float representing bandwidth
    """

    # find interquartile range and divide it by 1.34
    iqr_div134 = (df.quantile(0.75) - df.quantile(0.25)) / 1.34

    # choose minimum of 'iqr_div134' and standard deviation for each variable
    a = pd.concat([iqr_div134, df.std()], axis=1).min(axis=1)

    h = 0.9 * a * len(df)**(-1/5)

    # check bandwidths/std on each variable

    return h


def resample_variables_by_gaussian_kernel_density(
    df: pd.DataFrame, sample_n: int) -> pd.DataFrame:
    """
    For each column in Pandas DataFrame 'df', calculates a new sample of that
        variable based on Gaussian kernel density

    :param df: a Pandas DataFrame with columns of numerical data
    :param sample_n: the number of new samples to calculate for each column
    :return: a Pandas DataFrame with 'sample_n' rows and the same number of
        columns as 'df'
    """

    bandwidths = calculate_gaussian_kernel_density_bandwidth_silverman_rule(df)
    resample = df.sample(n=sample_n, replace=True)
    density_resample = np.random.normal(
        loc=resample, scale=bandwidths, size=(sample_n, df.shape[1]))

    density_resample = pd.DataFrame(density_resample, columns=df.columns)

    return density_resample


def create_grid_regular_intervals_two_variables(
    df: pd.DataFrame, intervals_num: int) -> pd.DataFrame:
    """
    1) Accepts Pandas DataFrame where first two columns are numerical values
    2) Finds the range of each of these columns and divides each range into
        equally spaced intervals; the number of intervals is specified by
        'intervals_num'
    3) Creates new DataFrame with two columns where the rows represent the
        Cartesian product of the equally spaced intervals

    :param df: a Pandas DataFrame where first two columns are numerical values
    :param intervals_num: scalar integer; the number of equally spaced intervals
        to create for each column
    :return:
    """

    intervals_df = df.apply(
        lambda d: np.linspace(start=d.min(), stop=d.max(), num=intervals_num))

    # the following code works much like 'expand.grid' in R, but it handles only
    #   two variables
    cartesian_product = list(
        it_product(intervals_df.iloc[:, 0], intervals_df.iloc[:, 1]))

    product_df = pd.DataFrame.from_records(
        cartesian_product, columns=df.columns)

    return product_df


def run_arviz_plots(
    az_data: az.InferenceData, draws_df: pd.DataFrame, 
    parameter_names: list[str], show: bool, output_path: Path) -> None:
    """
    Save a series of plots of the Bayesian model using the ArviZ library
    """

    az.style.use('arviz-darkgrid')


    # plot chain autocorrelation
    ##################################################

    az.plot_autocorr(az_data, var_names=parameter_names, show=show)
    output_filepath = output_path / 'plot_autocorr.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_autocorr(
        az_data, var_names=parameter_names, combined=True, show=show)
    output_filepath = output_path / 'plot_autocorr_combined_chains.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot energy
    ##################################################

    az.plot_energy(az_data, show=show)
    output_filepath = output_path / 'plot_energy.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot parameter density
    ##################################################

    az.plot_density(
        az_data, var_names=parameter_names, outline=False, shade=0.7,
        hdi_prob=0.9, point_estimate='mean', show=show)
    output_filepath = output_path / 'plot_density.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot distribution
    ##################################################

    az.plot_dist(
        draws_df[parameter_names[1]+'[1]'], rug=True,
        quantiles=[0.25, 0.5, 0.75], show=show)
    output_filepath = output_path / 'plot_distribution.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_dist(
        draws_df[parameter_names[1]+'[1]'], rug=True, cumulative=True,
        quantiles=[0.25, 0.5, 0.75], show=show)
    output_filepath = output_path / 'plot_distribution_cumulative.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot ESS across local parts of distribution
    ##################################################

    az.plot_ess(
        az_data, var_names=parameter_names, kind='local', n_points=10,
        rug=True, extra_methods=True, show=show)
    output_filepath = output_path / 'plot_ess_local.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_ess(
        az_data, var_names=parameter_names, kind='quantile', n_points=10,
        rug=True, extra_methods=True, show=show)
    output_filepath = output_path / 'plot_ess_quantile.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_ess(
        az_data, var_names=parameter_names, kind='evolution', n_points=10,
        rug=True, extra_methods=True, show=show)
    output_filepath = output_path / 'plot_ess_evolution.png'
    plt.savefig(output_filepath)
    plt.close()


    # forest plots
    ##################################################

    # look at model estimations of parameters, r-hat, and ess
    az.plot_forest(
        az_data, kind='forestplot', var_names=parameter_names,
        linewidth=6, markersize=8,
        hdi_prob=0.9, r_hat=True, ess=True, show=show)
    output_filepath = output_path / 'plot_forest.png'
    plt.savefig(output_filepath)
    plt.close()

    # look at model estimations of parameters, r-hat, and ess
    az.plot_forest(
        az_data, kind='ridgeplot', var_names=parameter_names,
        hdi_prob=0.9, r_hat=True, ess=True,
        ridgeplot_alpha=0.5, ridgeplot_overlap=2, ridgeplot_kind='auto',
        show=show)
    output_filepath = output_path / 'plot_forest_ridge.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot KDE
    ##################################################

    az.plot_kde(
        draws_df[parameter_names[0]], draws_df[parameter_names[1]+'[1]'],
        contour=True, show=show)
    output_filepath = output_path / 'plot_kde_contour.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_kde(
        draws_df[parameter_names[0]], draws_df[parameter_names[1]+'[1]'],
        contour=False, show=show)
    output_filepath = output_path / 'plot_kde_no_contour.png'
    plt.savefig(output_filepath)
    plt.close()


    # MCSE statistics and plots
    ##################################################

    az.mcse(az_data, var_names=parameter_names, method='mean')
    az.mcse(az_data, var_names=parameter_names, method='sd')
    az.mcse(az_data, var_names=parameter_names, method='quantile', prob=0.1)

    az.plot_mcse(
        az_data, var_names=parameter_names, errorbar=True, n_points=10)
    output_filepath = output_path / 'plot_mcse_errorbar.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_mcse(
        az_data, var_names=parameter_names, extra_methods=True,
        n_points=10)
    output_filepath = output_path / 'plot_mcse_extra_methods.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot pair
    ##################################################

    az.plot_pair(
        az_data, var_names=parameter_names, kind='scatter',
        divergences=True, show=show)
    output_filepath = output_path / 'plot_pair_scatter.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_pair(
        az_data, var_names=parameter_names, kind='kde',
        divergences=True, show=show)
    output_filepath = output_path / 'plot_pair_kde.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot parameters in parallel
    ##################################################

    az.plot_parallel(
        az_data, var_names=parameter_names, colornd='blue', show=show)
    output_filepath = output_path / 'plot_parallel.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot parameters in parallel
    ##################################################

    az.plot_posterior(
        az_data, var_names=parameter_names, hdi_prob=0.9,
        point_estimate='mean', show=show)
    output_filepath = output_path / 'plot_posterior.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot predictive check
    ##################################################

    az.plot_ppc(
        az_data, kind='kde', data_pairs={'y': 'predict_y_given_x'},
        random_seed=483742, show=show)
    output_filepath = output_path / 'plot_predictive_check_kde.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_ppc(
        az_data, kind='cumulative', data_pairs={'y': 'predict_y_given_x'},
        random_seed=483742, show=show)
    output_filepath = output_path / 'plot_predictive_check_cumulative.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_ppc(
        az_data, kind='scatter', data_pairs={'y': 'predict_y_given_x'},
        random_seed=483742, jitter=0.5, show=show)
    output_filepath = output_path / 'plot_predictive_check_scatter.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot chain rank order statistics
    ##################################################
    # each chain should show approximately a uniform distribution:
    #   https://arxiv.org/pdf/1903.08008
    #   Vehtari, Gelman, Simpson, Carpenter, Bürkner (2020)
    #   Rank-normalization, folding, and localization: An improved R for
    #       assessing convergence of MCMC

    az.plot_rank(
        az_data, var_names=parameter_names, kind='bars', show=show)
    output_filepath = output_path / 'plot_rank_bars.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_rank(
        az_data, var_names=parameter_names, kind='vlines', show=show)
    output_filepath = output_path / 'plot_rank_vlines.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot traces
    ##################################################

    az.plot_trace(
        az_data, var_names=parameter_names, legend=False, show=show)
    output_filepath = output_path / 'plot_trace.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot distributions on violin plot
    ##################################################

    az.plot_violin(
        az_data, var_names=parameter_names, rug=True,
        hdi_prob=0.9, show=show)
    output_filepath = output_path / 'plot_violin.png'
    plt.savefig(output_filepath)
    plt.close()


def run_bernoulli_example():
    """
    Run the "Hello, World" basic example to ensure that CmdStanPy is working:

        https://mc-stan.org/cmdstanpy/users-guide/hello_world.html
    """

    ##################################################
    # SET PATHS
    ##################################################

    input_path = Path.cwd() / 'input'

    src_path = Path.cwd() / 'src'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(exist_ok=True, parents=True)


    ##################################################
    # RUN MODEL
    ##################################################

    stan_filepath = src_path / 'bernoulli.stan'
    model = CmdStanModel(stan_file=stan_filepath)

    data_filepath = input_path / 'bernoulli.data.json'
    fit = model.sample(data=data_filepath)

    output_filepath = output_path / 'bernoulli_summary.csv'
    fit.summary().to_csv(output_filepath)


def main():
    """
    Original example from:

        Statistical Rethinking:  A Bayesian Course with Examples in R and Stan, 
            2016, Richard McElreath, CRC Press, Taylor & Francis Group, Boca 
            Raton, ISBN 978-1-4822-5344-3, chapter 5, page 122
    """


    ##################################################
    # SET PATHS
    ##################################################

    input_path = Path.cwd() / 'input'
    input_filepath = input_path / 'WaffleDivorce.csv'

    src_path = Path.cwd() / 'src'

    output_path = Path.cwd() / 'output'
    output_path.mkdir(exist_ok=True, parents=True)


    ##################################################
    #
    ##################################################

    data_df = pd.read_csv(input_filepath, sep=';')
    # data_df = pl.read_csv(input_filepath, separator=';')

    y = data_df['Divorce']
    x = data_df[['Marriage', 'MedianAgeMarriage']]
    n = x.shape[0]
    k = x.shape[1]

    # standardize predictor variables
    x = (x - x.mean()) / x.std()

    # generate new predictor variables based on Gaussian density or
    #   regularly-spaced intervals
    new_x_n = 3
    new_x_density = resample_variables_by_gaussian_kernel_density(x, new_x_n)
    new_x_regular = create_grid_regular_intervals_two_variables(x, new_x_n)

    # Boolean flag to indicate whether to have Stan make model predictions at
    #   new 'x' values
    use_new_x = True

    if use_new_x:
        stan_data = {
            'N': n, 'K': k, 'x': x, 'y': y,
            'predict_y_constant_x_n': new_x_regular.shape[0],
            'predict_y_constant_x': new_x_regular,
            'predict_y_density_x_n': new_x_density.shape[0],
            'predict_y_density_x': new_x_density}
        stan_filename = 'ch5_p122_multiple_regression_w_new_xs.stan'
    else:
        stan_data = {'N': n, 'K': k, 'x': x, 'y': y}
        stan_filename = 'ch5_p122_multiple_regression_wo_new_xs.stan'


    stan_filepath = src_path / stan_filename
    model = CmdStanModel(stan_file=stan_filepath)

    # fit_model = model.sample(
    #     data=stan_data, chains=2, thin=2, seed=22074,
    #     iter_warmup=100, iter_sampling=200, output_dir=output_path)
    fit_model = model.sample(
        data=stan_data, chains=4, thin=2, seed=21520,
        iter_warmup=4000, iter_sampling=8000, output_dir=output_path)
    # fit_model = model.sample(
    #     data=stan_data, chains=4, thin=1, seed=81398,
    #     iter_warmup=1000, iter_sampling=2000, output_dir=output_path)


    # tabular summaries
    ##################################################

    # parameter estimates are nearly identical to those in book, page 125

    # text summary of means, sd, se, and quantiles for parameters, n_eff, & Rhat
    fit_df = fit_model.summary()
    output_filepath = output_path / 'summary.csv'
    fit_df.to_csv(output_filepath, index=True)


    # all samples for all parameters, predicted values, and diagnostics
    #   number of rows = number of 'iter_sampling' in 'CmdStanModel.sample' call
    draws_df = fit_model.draws_pd()
    output_filepath = output_path / 'draws.csv'
    draws_df.to_csv(output_filepath, index=True)


    # set plot parameters
    ##################################################

    # https://python.arviz.org/en/latest/getting_started/Introduction.html
    az_data = az.from_cmdstanpy(
        posterior=fit_model,
        #posterior_predictive=['predict_y_given_x', 'predicted_y_constant_x', 'predicted_y_density_x'],
        posterior_predictive='predict_y_given_x',
        #posterior_predictive='predicted_y_constant_x',
        #posterior_predictive='predicted_y_density_x',
        observed_data={'y': stan_data['y']})


    show = False
    parameter_names = ['alpha', 'beta', 'sigma']
    run_arviz_plots(az_data, draws_df, parameter_names, show, output_path)


if __name__ == '__main__':
    run_bernoulli_example()
    main()
