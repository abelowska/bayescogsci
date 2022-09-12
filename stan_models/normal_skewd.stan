data {
  int<lower = 1> N;  // Total number of trials
  vector[N] y;  // Score in each trial
}
parameters {
  real mu;
  real<lower = 0> sigma;
  real alpha;
}
model {
  // Priors:
  target += normal_lpdf(mu | 0, 20);
  target += lognormal_lpdf(sigma | 3, 1);
  target += normal_lpdf(alpha | 0, 1);
  // Likelihood:

//   target += normal_lpdf(y | mu, sigma);
  // skew normal dist
  target += skew_normal_lpdf(y | mu, sigma, alpha);
}
