data {
  int<lower=0> N;        // number of cases/data points
  int<lower=0> K;        // number of predictor variables
  matrix[N, K] x;        // matrix of predictor variables
  vector[N] y;           // outcome/response variable

  // predictor/x values at which to predict response/y values
  // predictor/x values are at regular intervals
  // int<lower=0> predict_y_constant_x_n;      // number of x values
  // matrix[predict_y_constant_x_n, K] predict_y_constant_x;

  // predictor/x values at which to predict response/y values
  // predictor/x values are sampled to reflect the density of the original x data
  // int<lower=0> predict_y_density_x_n;       // number of x values
  // matrix[predict_y_density_x_n, K] predict_y_density_x;
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> sigma;
}
model {
  // priors provided in McElreath (2016), p. 125
  alpha ~ normal(10, 10);
  beta ~ normal(0, 1);
  sigma ~ uniform(0, 10);
  y ~ normal(x * beta + alpha, sigma);
}
generated quantities {
  // vector[N] predict_y_given_x;
  // vector[predict_y_constant_x_n] predicted_y_constant_x;
  // vector[predict_y_density_x_n] predicted_y_density_x;
  
  // generate response/y values at the given predictor/x values
  // predict_y_given_x = x * beta + alpha + normal_rng(0, sigma);
  
  // generate response/y values at regular predictor/x intervals, which can be
  //      used to show credible intervals at regularly spaced points across the
  //      predictor/x space
  // predicted_y_constant_x = predict_y_constant_x * beta + alpha + normal_rng(0, sigma);
  
  // generate response/y values at predictor/x values that reflect the density
  //      of the predictor/x values in the original data, which is useful for
  //      generating simulated data from the model
  // predicted_y_density_x = predict_y_density_x * beta + alpha + normal_rng(0, sigma);
}
