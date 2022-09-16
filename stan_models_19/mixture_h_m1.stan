data {
  int<lower = 1> N;
  vector[N] rt;
  vector[N] patient;
  int N_subj;
  array[N] int subj;
}
parameters {
  real alpha;
  real<lower = 0> delta;
  real<lower = 0> sigma;
  real<lower = 0> sigma2;
  real<lower = 0, upper = 1> alpha_lapses;
  real beta_lapses;
  matrix[N_subj, 2] z;
  vector[2] tau_u;
}

transformed parameters {
  matrix[N_subj, 2] u;
  u[, 1] = z[, 1] * tau_u[1];
  u[, 2] = z[, 2] * tau_u[2];
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

  target += normal_lpdf(tau_u | 0, 1)
    - 2*normal_lccdf(0 | 0, 1);

  target += std_normal_lpdf(to_vector(z));

  // change in lapses probability given patients
  target += beta_lpdf(alpha_lapses | 2, 8); // avg if non-patient
  target += normal_lpdf(beta_lapses | 0, .5); // change if patient

  // likelihood
  for(n in 1:N) {
    real lodds_lapses = logit(alpha_lapses) + patient[n] * beta_lapses;

    target += log_sum_exp(log_inv_logit(lodds_lapses)+
                            lognormal_lpdf(rt[n] | alpha + delta + u[subj[n], 1], sigma2),
                            log1m_inv_logit(lodds_lapses) +
                            lognormal_lpdf(rt[n] | alpha + u[subj[n], 2], sigma));
  }
}
