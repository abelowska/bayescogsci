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
  target += normal_lpdf(alpha | 160, 50);
  target += normal_lpdf(sigma | 0, 50)
    - normal_lccdf(0 | 0, 50);
    
  // likelihood
  target += normal_lpdf(rt | alpha, sigma);
}
