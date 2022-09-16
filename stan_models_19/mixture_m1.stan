data {
  int<lower = 1> N;
  vector[N] rt;
  vector[N] patient;
}
parameters {
  real alpha;
  real<lower = 0> delta;
  real<lower = 0> sigma;
  real<lower = 0> sigma2;
  real<lower = 0, upper = 1> alpha_lapses;
  real beta_lapses;
  

}
model {
  // priors for the task component
  target += normal_lpdf(alpha | 5, 2);
  target += normal_lpdf(sigma | 0, 1)
    - normal_lccdf(0 | 0, 1);

  // priors for the lapses component
  target += normal_lpdf(delta | 0, 1) - normal_lccdf(0 | 0, 1);
  target += normal_lpdf(sigma2 | 0, 1)
    - normal_lccdf(0 | 0, 1);

  // change in lapses probability given patients
  target += beta_lpdf(alpha_lapses | 1, 8); // avg if non-patient
  target += normal_lpdf(beta_lapses | 0, 1); // change if patient

  // likelihood
  for(n in 1:N) {
    real lodds_lapses = logit(alpha_lapses) + patient[n] * beta_lapses;

    target += log_sum_exp(log_inv_logit(lodds_lapses)+
                            lognormal_lpdf(rt[n] | alpha + delta, sigma2),
                            log1m_inv_logit(lodds_lapses) +
                            lognormal_lpdf(rt[n] | alpha, sigma));
  }
}
