data {
  int<lower=1> N;
  int<lower=0> K; // num of predictors
  matrix [N,K] X; // independent variables
  vector[N] p_size; // dependent variable
}
parameters {
  real alpha;
  vector [K] beta;
  real<lower = 0> sigma;
}
model {
  // priors including all constants
  target += normal_lpdf(alpha | 1000, 500);
  target += normal_lpdf(beta | 0, 100);
  target += normal_lpdf(sigma | 0, 1000)
    - normal_lccdf(0 | 0, 1000);

//   target += normal_lpdf(p_size | alpha + X * beta, sigma);
  target += normal_id_glm_lpdf(p_size | X, alpha, beta, sigma);
}
