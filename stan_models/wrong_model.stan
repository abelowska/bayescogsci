data {
  int<lower = 1> N;
  vector[N] rt;
  int correct[N];
}
parameters {
  real<lower = 0> sigma;
  real mu; # no mu declaration
  real <lower = 0, upper = 1> theta; # alpha instead theta for bernoulli dist
}
model {
  target += normal_lpdf(mu | 0, 20);
  target += lognormal_lpdf(sigma | 3, 1); # no semi-colon
  // target += normal_lpdf(alpha | 0, 1.5); # prior for alpha 
  target += beta_lpdf(theta | 2, 2);


  for(n in 1:N) {
    target += lognormal_lpdf(rt[n] | mu, sigma);
    // target += bernoulli_logit_lpmf(correct[n] | alpha); # wrong method
    target += bernoulli_lpmf(correct[n] | theta); # wrong method

  }
}
