data {
  int<lower=1> N;
  vector[N] c_cloze; // independent variable
  vector[N] signal; // dependent variable
}
parameters {
  real alpha;
  real beta;
  real<lower = 0> sigma;
}
model {
  // priors:
  target += normal_lpdf(alpha| 0,10);
  target += normal_lpdf(beta | 0,10);
  target += normal_lpdf(sigma | 0, 50)  -
    normal_lccdf(0 | 0, 50);
    
  // likelihood
  target += normal_lpdf(signal | alpha + c_cloze * beta, sigma);
}

generated quantities{
  array[N] real log_lik;
  for (n in 1:N){
    log_lik[n] = normal_lpdf(signal[n] | alpha + c_cloze[n] * beta, sigma);
  }
}
