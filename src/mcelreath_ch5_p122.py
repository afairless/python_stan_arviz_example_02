#! /usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from cmdstanpy import CmdStanModel
import arviz as az
import matplotlib.pyplot as plt

try:
    from common import (
        read_text_file, 
        write_list_to_text_file,
        )

except:
    from src.common import (
        read_text_file, 
        write_list_to_text_file,
        )


def calculate_gaussian_kernel_density_bandwidth_silverman_rule(a_df):
    """
    Calculate Gaussian kernel density bandwidth based on Silverman's rule from:
        Silverman, B. W. (1986).  Density Estimation for Statistics and Data
            Analysis.  London: Chapman & Hall/CRC. p. 45
            ISBN 978-0-412-24620-3

    Wikipedia is a useful reference:

        https://en.wikipedia.org/wiki/Kernel_density_estimation

    :param a_df: a Pandas DataFrame where the Gaussian kernel density will be
        calculated for each column
    :return: scalar float representing bandwidth
    """

    from pandas import concat as pd_concat

    # find interquartile range and divide it by 1.34
    iqr_div134 = (a_df.quantile(0.75) - a_df.quantile(0.25)) / 1.34

    # choose minimum of 'iqr_div134' and standard deviation for each variable
    a = pd_concat([iqr_div134, a_df.std()], axis=1).min(axis=1)

    h = 0.9 * a * len(a_df)**(-1/5)

    # check bandwidths/std on each variable

    return h


def resample_variables_by_gaussian_kernel_density(a_df, sample_n):
    """
    For each column in Pandas DataFrame 'a_df', calculates a new sample of that
        variable based on Gaussian kernel density

    :param a_df: a Pandas DataFrame with columns of numerical data
    :param sample_n: the number of new samples to calculate for each column
    :return: a Pandas DataFrame with 'sample_n' rows and the same number of
        columns as 'a_df'
    """

    from numpy.random import normal as np_normal
    from pandas import DataFrame as pd_df

    bandwidths = calculate_gaussian_kernel_density_bandwidth_silverman_rule(a_df)
    re_sample = a_df.sample(n=sample_n, replace=True)
    density_re_sample = np_normal(
        loc=re_sample, scale=bandwidths, size=(sample_n, a_df.shape[1]))

    density_re_sample = pd_df(density_re_sample, columns=a_df.columns)

    return density_re_sample


def create_grid_regular_intervals_two_variables(a_df, intervals_num):
    """
    1) Accepts Pandas DataFrame where first two columns are numerical values
    2) Finds the range of each of these columns and divides each range into
        equally spaced intervals; the number of intervals is specified by
        'intervals_num'
    3) Creates new DataFrame with two columns where the rows represent the
        Cartesian product of the equally spaced intervals

    :param a_df: a Pandas DataFrame where first two columns are numerical values
    :param intervals_num: scalar integer; the number of equally spaced intervals
        to create for each column
    :return:
    """

    import pandas as pd
    from numpy import linspace as np_linspace
    from itertools import product as it_product

    intervals_df = a_df.apply(lambda d: np_linspace(
        start=d.min(), stop=d.max(), num=intervals_num))

    # the following code works much like 'expand.grid' in R, but it handles only
    #   two variables
    cartesian_product = list(it_product(intervals_df.iloc[:, 0],
                                        intervals_df.iloc[:, 1]))

    product_df = pd.DataFrame.from_records(
        cartesian_product, columns=a_df.columns)

    return product_df


def run_arviz_plots(
    az_data: az.InferenceData, draws_df: pd.DataFrame, 
    parameter_names: list[str], show: bool, output_path: Path) -> None:
    """
    Save a series of plots of the Bayesian model using the ArviZ library
    """


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


    # HPD plot
    ##################################################

    # look at model estimations of parameters, r-hat, and ess
    predicted_y_colnames = [e for e in draws_df.columns if 'y_given_x' in e]
    predicted_y_df = draws_df[predicted_y_colnames]

    # x_col_idx = 0
    # plt.scatter(x.iloc[:, x_col_idx], y)
    # az.plot_hpd(
    #     x.iloc[:, x_col_idx], predicted_y_df, credible_interval=0.5, show=show)
    # az.plot_hpd(
    #     x.iloc[:, x_col_idx], predicted_y_df, credible_interval=0.9, show=show)
    # output_filepath = output_path / 'plot_hpd_x0.png'
    # plt.savefig(output_filepath)

    # plt.close()
    # x_col_idx = 1
    # plt.scatter(x.iloc[:, x_col_idx], y)
    # az.plot_hpd(
    #     x.iloc[:, x_col_idx], predicted_y_df, credible_interval=0.5, show=show)
    # az.plot_hpd(
    #     x.iloc[:, x_col_idx], predicted_y_df, credible_interval=0.9, show=show)
    # output_filepath = output_path / 'plot_hpd_x1.png'
    # plt.savefig(output_filepath)
    # plt.close()


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


    '''
    # I haven't figured out how to calculate the MCSE statistics directly from
    #   the 'draws_df'
    
    # STD / sqrt(N), but this doesn't exactly match any of the statistics from 
    #   'az.mcse'
    draws_df.alpha.std() / np.sqrt(len(draws_df))

    # I thought MCSE by quantile could be calculated by using only the samples
    #   in that quantile, but haven't been able to get that to work
    var = 'beta[2]'
    var = 'sigma'
    q = 0.05
    n = (draws_df[var] < draws_df[var].quantile(q=q)).sum()
    m = draws_df.loc[draws_df[var] < draws_df[var].quantile(q=q)].loc[:, var].std()
    m / np.sqrt(n)

    draws_df['beta[1]'].quantile(q=0.05)
    (draws_df['beta[2]'] < draws_df['beta[2]'].quantile(q=0.05)).sum()
    '''


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










    # miscellaneous notes
    ##################################################

    '''
    fit_model.summary()
    fit_model.summary().keys()
    fit_model.stansummary()

    fit_model.constrain_pars() # needs an argument
    fit_model.constrained_param_names()
    fit_model.data
    fit_model.date

    # gets ordered dictionary of 'iter' # values per parameter
    fit_model.extract()

    fit_model.flatnames
    fit_model.get_adaptation_info()
    fit_model.get_inits()
    fit_model.get_inv_metric()
    fit_model.get_last_position()

    # 2 sequences (# chains?) of 300 each (iter #)
    fit_model.get_logposterior()

    # returns array shape 165, 2
    # '2' might be # chains; 165 might be parameters
    fit_model.get_posterior_mean()

    # returns list of 2 ordered dicts, each of length 6
    # returns list of 6 parameters for each chain:
    #   accept_stat, stepsize, treedepth, n_leapfrog, divergent, energy
    fit_model.get_sampler_params()[1].keys()

    # both return the random seed
    fit_model.get_seed()
    fit_model.random_seed

    fit_model.get_stancode()

    # both get model object
    fit_model.get_stanmodel()
    fit_model.stanmodel

    fit_model.get_stepsize()  # apparently 1 step size for each chain
    fit_model.grad_log_prob()  # requires argument
    fit_model.inits
    fit_model.log_prob()   # requires argument
    fit_model.mode  # returns single scalar
    fit_model.model_name  # returns string
    fit_model.model_pars  # returns list of names of parameters
    fit_model.par_dims  # returns list of lists, each giving dims of parameters
    fit_model.plot()
    fit_model.sim   # ordered dictionary of parameters and permutations
    fit_model.stan_args  # returns dict of all arguments

    fit_model.to_dataframe()
    fit_model.traceplot()
    fit_model.unconstrain_pars() # requires argument
    fit_model.unconstrained_param_names()



    # visual style options
    az.style.library.keys()

    az.style.use('arviz-whitegrid')

    az.plot_autocorr(fit_model) # plots each chain w/ itself & w/ other chains; check
    az.plot_density(fit_model)
    az.plot_energy(fit_model)
    az.plot_ess(fit_model)
    az.plot_forest(fit_model) # plots params for each chain; check
    az.plot_hpd()
    az.plot_kde(fit_model)
    az.plot_mcse(fit_model)
    az.plot_pair(fit_model) # scatterplot for each param combo
    az.plot_parallel(fit_model) # plots params; check 'w/ & w/o divergences'
    az.plot_posterior(fit_model)
    az.plot_ppc(fit_model)
    az.plot_rank(fit_model) # plots 'rank' of params by chain & iter
    az.plot_trace(fit_model)
    az.plot_kde(fit_model.extract()['beta'][:, 0],
                fit_model.extract()['beta'][:, 1])
    az.plot_dist(fit_model.extract()['beta'][:, 0],
                 fit_model.extract()['beta'][:, 1])
    az.plot_violin(fit_model)
    plt.savefig('violin.png')
    plt.close()


    az.plot_compare(fit_model)
    az.plot_elpd(fit_model)
    az.plot_joint(fit_model) # reduce # variables to 2
    az.plot_khat(fit_model)
    az.plot_loo_pit(fit_model)

    fit_model.extract().keys()
    fit_model.extract()['beta'][:, 0]

    az_data = az.from_pystan(
        posterior=fit_model,
        posterior_predictive='y_hat',
        observed_data=['y'],
        log_likelihood={'y': 'log_lik'},

    )
    '''



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


    az.style.use('arviz-darkgrid')
    parameter_names = ['alpha', 'beta', 'sigma']
    show = False

    run_arviz_plots(az_data, draws_df, parameter_names, show, output_path)








if __name__ == '__main__':
    run_bernoulli_example()
    main()
