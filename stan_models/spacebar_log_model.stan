data {
  int<lower=1> N;
  vector[N] rt;
}
parameters {
  real alpha;
  real<lower = 0> sigma;
}
model {
  // priors:
  target += normal_lpdf(alpha | 6, 1.5);
  target += normal_lpdf(sigma | 0, 1)
    - normal_lccdf(0 | 0, 1);
    
  // likelihood
  target += lognormal_lpdf(rt | alpha, sigma);
}
