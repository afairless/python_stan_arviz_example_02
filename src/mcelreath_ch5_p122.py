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
    """


    ##################################################
    # SET OUTPUT PATH
    ##################################################

    output_path = Path.cwd() / 'output'
    output_path.mkdir(exist_ok=True, parents=True)

    input_path = Path.cwd() / 'input'
    input_filepath = input_path / 'WaffleDivorce.csv'


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
            'N': n, 'K': k, 'x': x.to_json(), 'y': y.to_json(),
            'predict_y_constant_x_n': new_x_regular.shape[0],
            'predict_y_constant_x': new_x_regular.to_json(),
            'predict_y_density_x_n': new_x_density.shape[0],
            'predict_y_density_x': new_x_density.to_json()}
        stan_filename = 'ch5_p122_multiple_regression_w_new_xs.stan'
    else:
        stan_data = {'N': n, 'K': k, 'x': x, 'y': y}
        stan_filename = 'ch5_p122_multiple_regression_wo_new_xs.stan'


    dir(stan)
    dir(stan.model)
    dir(stan.fit)
    dir(stan.model.stan)
    dir(stan.model.stan.common)

    stan_filepath = Path.cwd() / 'src' / stan_filename
    with open(stan_filepath) as f:
        stan_code = f.read()
    stan_model = stan.build(program_code=stan_code, data=stan_data)

    #fit_model = stan_model.sampling(
    #    data=stan_data, iter=300, chains=2, warmup=150, thin=1, seed=22411517)
    #fit_model = stan_model.sampling(
    #    data=stan_data, iter=300, chains=4, warmup=150, thin=1, seed=22074)
    #fit_model = stan_model.sampling(
    #    data=stan_data, iter=2000, chains=4, warmup=1000, thin=1, seed=22074)
    fit_model = stan_model.sampling(
        data=stan_data, iter=2000, chains=4, warmup=1000, thin=2, seed=22074)


    # tabular summaries
    ##################################################

    print(fit_model)
    # parameter estimates are nearly identical to those in book, page 125

    # all samples for all parameters, predicted values, and diagnostics
    #   number of rows = number of 'iter' in 'StanModel.sampling' call
    fit_df = fit_model.to_dataframe()

    # text summary of means, sd, se, and quantiles for parameters, n_eff, & Rhat
    fit_stansummary = fit_model.stansummary()
    output_filepath = output_path / 'summary_stansummary.txt'
    write_list_to_text_file([fit_stansummary], output_filepath, True)

    # same summary as for 'stansummary', but in matrix/dataframe form instead of text
    fit_summary_df = pd.DataFrame(
        fit_model.summary()['summary'],
        index=fit_model.summary()['summary_rownames'],
        columns=fit_model.summary()['summary_colnames'])
    output_filepath = output_path / 'summary_summary.txt'
    fit_summary_df.to_csv(output_filepath, index=True)

    # same summary as for 'stansummary', but separated by chain
    fit_summary_by_chain_df_list = [pd.DataFrame(
        fit_model.summary()['c_summary'][:, :, i],
        index=fit_model.summary()['c_summary_rownames'],
        columns=fit_model.summary()['c_summary_colnames'])
        for i in range(fit_model.summary()['c_summary'].shape[-1])]


    # plot parameters
    ##################################################

    az_stan_data = az.from_pystan(
        posterior=fit_model,
        #posterior_predictive=['predict_y_given_x', 'predicted_y_constant_x', 'predicted_y_density_x'],
        posterior_predictive='predict_y_given_x',
        #posterior_predictive='predicted_y_constant_x',
        #posterior_predictive='predicted_y_density_x',
        observed_data=['y']
    )


    az.style.use('arviz-darkgrid')
    parameter_names = ['alpha', 'beta', 'sigma']
    show = True


    # plot chain autocorrelation
    ##################################################

    az.plot_autocorr(az_stan_data, var_names=parameter_names, show=show)
    output_filepath = output_path / 'plot_autocorr.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_autocorr(
        az_stan_data, var_names=parameter_names, combined=True, show=show)
    output_filepath = output_path / 'plot_autocorr_combined_chains.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot energy
    ##################################################

    az.plot_energy(az_stan_data, show=show)
    output_filepath = output_path / 'plot_energy.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot parameter density
    ##################################################

    az.plot_density(
        az_stan_data, var_names=parameter_names, outline=False, shade=0.7,
        credible_interval=0.9, point_estimate='mean', show=show)
    output_filepath = output_path / 'plot_density.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot distribution
    ##################################################

    az.plot_dist(
        fit_df[parameter_names[1]+'[1]'], rug=True,
        quantiles=[0.25, 0.5, 0.75], show=show)
    output_filepath = output_path / 'plot_distribution.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_dist(
        fit_df[parameter_names[1]+'[1]'], rug=True, cumulative=True,
        quantiles=[0.25, 0.5, 0.75], show=show)
    output_filepath = output_path / 'plot_distribution_cumulative.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot ESS across local parts of distribution
    ##################################################

    az.plot_ess(
        az_stan_data, var_names=parameter_names, kind='local', n_points=10,
        rug=True, extra_methods=True, show=show)
    output_filepath = output_path / 'plot_ess_local.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_ess(
        az_stan_data, var_names=parameter_names, kind='quantile', n_points=10,
        rug=True, extra_methods=True, show=show)
    output_filepath = output_path / 'plot_ess_quantile.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_ess(
        az_stan_data, var_names=parameter_names, kind='evolution', n_points=10,
        rug=True, extra_methods=True, show=show)
    output_filepath = output_path / 'plot_ess_evolution.png'
    plt.savefig(output_filepath)
    plt.close()


    # forest plots
    ##################################################

    # look at model estimations of parameters, r-hat, and ess
    az.plot_forest(
        az_stan_data, kind='forestplot', var_names=parameter_names,
        linewidth=6, markersize=8,
        credible_interval=0.9, r_hat=True, ess=True, show=show)
    output_filepath = output_path / 'plot_forest.png'
    plt.savefig(output_filepath)
    plt.close()

    # look at model estimations of parameters, r-hat, and ess
    az.plot_forest(
        az_stan_data, kind='ridgeplot', var_names=parameter_names,
        credible_interval=0.9, r_hat=True, ess=True,
        ridgeplot_alpha=0.5, ridgeplot_overlap=2, ridgeplot_kind='auto',
        show=show)
    output_filepath = output_path / 'plot_forest_ridge.png'
    plt.savefig(output_filepath)
    plt.close()


    # HPD plot
    ##################################################

    # look at model estimations of parameters, r-hat, and ess
    predicted_y_colnames = [e for e in fit_df.columns if 'y_given_x' in e]
    predicted_y_df = fit_df[predicted_y_colnames]

    x_col_idx = 0
    plt.scatter(x.iloc[:, x_col_idx], y)
    az.plot_hpd(
        x.iloc[:, x_col_idx], predicted_y_df, credible_interval=0.5, show=show)
    az.plot_hpd(
        x.iloc[:, x_col_idx], predicted_y_df, credible_interval=0.9, show=show)
    output_filepath = output_path / 'plot_hpd_x0.png'
    plt.savefig(output_filepath)

    plt.close()
    x_col_idx = 1
    plt.scatter(x.iloc[:, x_col_idx], y)
    az.plot_hpd(
        x.iloc[:, x_col_idx], predicted_y_df, credible_interval=0.5, show=show)
    az.plot_hpd(
        x.iloc[:, x_col_idx], predicted_y_df, credible_interval=0.9, show=show)
    output_filepath = output_path / 'plot_hpd_x1.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot KDE
    ##################################################

    az.plot_kde(
        fit_df[parameter_names[0]], fit_df[parameter_names[1]+'[1]'],
        contour=True, show=show)
    output_filepath = output_path / 'plot_kde_contour.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_kde(
        fit_df[parameter_names[0]], fit_df[parameter_names[1]+'[1]'],
        contour=False, show=show)
    output_filepath = output_path / 'plot_kde_no_contour.png'
    plt.savefig(output_filepath)
    plt.close()


    # MCSE statistics and plots
    ##################################################

    az.mcse(az_stan_data, var_names=parameter_names, method='mean')
    az.mcse(az_stan_data, var_names=parameter_names, method='sd')
    az.mcse(az_stan_data, var_names=parameter_names, method='quantile', prob=0.1)

    az.plot_mcse(
        az_stan_data, var_names=parameter_names, errorbar=True, n_points=10)
    output_filepath = output_path / 'plot_mcse_errorbar.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_mcse(
        az_stan_data, var_names=parameter_names, extra_methods=True,
        n_points=10)
    output_filepath = output_path / 'plot_mcse_extra_methods.png'
    plt.savefig(output_filepath)
    plt.close()


    '''
    # I haven't figured out how to calculate the MCSE statistics directly from
    #   the 'fit_df'
    
    # STD / sqrt(N), but this doesn't exactly match any of the statistics from 
    #   'az.mcse'
    fit_df.alpha.std() / np.sqrt(len(fit_df))

    # I thought MCSE by quantile could be calculated by using only the samples
    #   in that quantile, but haven't been able to get that to work
    var = 'beta[2]'
    var = 'sigma'
    q = 0.05
    n = (fit_df[var] < fit_df[var].quantile(q=q)).sum()
    m = fit_df.loc[fit_df[var] < fit_df[var].quantile(q=q)].loc[:, var].std()
    m / np.sqrt(n)

    fit_df['beta[1]'].quantile(q=0.05)
    (fit_df['beta[2]'] < fit_df['beta[2]'].quantile(q=0.05)).sum()
    '''


    # plot pair
    ##################################################

    az.plot_pair(
        az_stan_data, var_names=parameter_names, kind='scatter',
        divergences=True, show=show)
    output_filepath = output_path / 'plot_pair_scatter.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_pair(
        az_stan_data, var_names=parameter_names, kind='kde',
        divergences=True, show=show)
    output_filepath = output_path / 'plot_pair_kde.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot parameters in parallel
    ##################################################

    az.plot_parallel(
        az_stan_data, var_names=parameter_names, colornd='blue', show=show)
    output_filepath = output_path / 'plot_parallel.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot parameters in parallel
    ##################################################

    az.plot_posterior(
        az_stan_data, var_names=parameter_names, credible_interval=0.9,
        point_estimate='mean', show=show)
    output_filepath = output_path / 'plot_posterior.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot predictive check
    ##################################################

    az.plot_ppc(
        az_stan_data, kind='kde', data_pairs={'y': 'predict_y_given_x'},
        random_seed=483742, show=show)
    output_filepath = output_path / 'plot_predictive_check_kde.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_ppc(
        az_stan_data, kind='cumulative', data_pairs={'y': 'predict_y_given_x'},
        random_seed=483742, show=show)
    output_filepath = output_path / 'plot_predictive_check_cumulative.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_ppc(
        az_stan_data, kind='scatter', data_pairs={'y': 'predict_y_given_x'},
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
        az_stan_data, var_names=parameter_names, kind='bars', show=show)
    output_filepath = output_path / 'plot_rank_bars.png'
    plt.savefig(output_filepath)
    plt.close()

    az.plot_rank(
        az_stan_data, var_names=parameter_names, kind='vlines', show=show)
    output_filepath = output_path / 'plot_rank_vlines.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot traces
    ##################################################

    az.plot_trace(
        az_stan_data, var_names=parameter_names, legend=False, show=show)
    output_filepath = output_path / 'plot_trace.png'
    plt.savefig(output_filepath)
    plt.close()


    # plot distributions on violin plot
    ##################################################

    az.plot_violin(
        az_stan_data, var_names=parameter_names, rug=True,
        credible_interval=0.9, show=show)
    output_filepath = output_path / 'plot_violin.png'
    plt.savefig(output_filepath)
    plt.close()










    # miscellaneous notes
    ##################################################

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



    ##################################################
    #
    ##################################################

    # https://pystan.readthedocs.io/en/latest/index.html

    import stan

    schools_code = """
    data {
      int<lower=0> J;         // number of schools
      array[J] real y;              // estimated treatment effects
      array[J] real<lower=0> sigma; // standard error of effect estimates
    }
    parameters {
      real mu;                // population treatment effect
      real<lower=0> tau;      // standard deviation in treatment effects
      vector[J] eta;          // unscaled deviation from mu by school
    }
    transformed parameters {
      vector[J] theta = mu + tau * eta;        // school treatment effects
    }
    model {
      target += normal_lpdf(eta | 0, 1);       // prior log-density
      target += normal_lpdf(y | theta, sigma); // log-likelihood
    }
    """

    schools_data = {
        "J": 8,
        "y": [28,  8, -3,  7, -1,  1, 18, 12],
        "sigma": [15, 10, 16, 11,  9, 11, 10, 18]}

    posterior = stan.build(schools_code, data=schools_data)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    eta = fit["eta"]  # array with shape (8, 4000)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas


    ##################################################
    #
    ##################################################

    # https://arviz-devs.github.io/arviz/notebooks/InferenceDataCookbook.html

    schools_code = """
    data {
        int<lower=0> J;
        real y[J];
        real<lower=0> sigma[J];
    }

    parameters {
        real mu;
        real<lower=0> tau;
        real theta_tilde[J];
    }

    transformed parameters {
        real theta[J];
        for (j in 1:J)
            theta[j] = mu + tau * theta_tilde[j];
    }

    model {
        mu ~ normal(0, 5);
        tau ~ cauchy(0, 5);
        theta_tilde ~ normal(0, 1);
        y ~ normal(theta, sigma);
    }

    generated quantities {
        vector[J] log_lik;
        vector[J] y_hat;
        for (j in 1:J) {
            log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
            y_hat[j] = normal_rng(theta[j], sigma[j]);
        }
    }
    """

    eight_school_data = {
        'J': 8,
        'y': np.array([28., 8., -3., 7., -1., 1., 18., 12.]),
        'sigma': np.array([15., 10., 16., 11., 9., 11., 10., 18.])
    }

    stan_model = pystan.StanModel(model_code=schools_code)
    fit_model = stan_model.sampling(data=eight_school_data, control={"adapt_delta" : 0.9})

    az_stan_data = az.from_pystan(
        posterior=fit_model,
        posterior_predictive='y_hat',
        observed_data=['y'],
        log_likelihood={'y': 'log_lik'},
        coords={'school': np.arange(eight_school_data['J'])},
        dims={
            'theta': ['school'],
            'y': ['school'],
            'log_lik': ['school'],
            'y_hat': ['school'],
            'theta_tilde': ['school']
        }
    )

    az.plot_mcse(az_stan_data, var_names=['mu', 'tau'], extra_methods=True)
    az.plot_mcse(az_stan_data, var_names=['theta', 'theta_tilde'])

    az.plot_posterior(az_stan_data, var_names=['mu', 'tau'], credible_interval=0.9)
    az.plot_posterior(az_stan_data, var_names=['theta'], credible_interval=0.9)
    az.plot_posterior(az_stan_data, credible_interval=0.9, group='posterior_predictive')
    az.plot_posterior(az_stan_data, credible_interval=0.9, group='posterior')

    az.plot_ppc(az_stan_data, data_pairs={'y': 'y_hat'})
    az.plot_ppc(az_stan_data, kind='cumulative', data_pairs={'y': 'y_hat'})
    az.plot_ppc(az_stan_data, kind='scatter', data_pairs={'y': 'y_hat'})



    ##################################################
    #
    ##################################################
    """
    np.random.seed(101)

    stan_code = '''
    data {
        int<lower=0> N;
        vector[N] x;
        vector[N] y;
    }
    parameters {
        real alpha;
        real beta;
        real <lower=0> sigma;
    }
    model {
        y ~ normal(alpha + beta * x, sigma);
    }
    '''

    alpha = 4.0
    beta = 0.5
    sigma = 1.0

    x = 10 * np.random.rand(100)
    y = alpha + beta * x
    y = np.random.normal(y, scale=sigma)

    data = {'N': len(x), 'x': x, 'y': y}

    stan_model = pystan.StanModel(model_code=stan_code)

    fit_model = stan_model.sampling(
        data=data, iter=1000, chains=4, warmup=500, thin=1, seed=101)

    summary_dict = fit_model.summary()

    summary_df = pd.DataFrame(
        summary_dict['summary'],
        columns=summary_dict['summary_colnames'],
        index=summary_dict['summary_rownames'])

    summary_c_df = pd.DataFrame(
        summary_dict['c_summary'],
        columns=summary_dict['c_summary_colnames'],
        index=summary_dict['c_summary_rownames'])

    summary_dict.keys()
    print(fit_model)
    dir(fit_model)

    fit_model.to_dataframe().columns
    fit_model.traceplot()

    az.plot_density(fit_model)

    az_data = az.from_pystan(
        posterior=fit_model,
        posterior_predictive='y_hat',
        observed_data=['y'],
        log_likelihood={'y': 'log_lik'},

    )



    alpha = fit_model['alpha']
    beta = fit_model['beta']
    sigma = fit_model['sigma']
    lp = fit_model['lp__']
    """





if __name__ == '__main__':
    main()
