data {
  int<lower=1> N;
  vector[N] rt;
  int<lower = 1> N_subj;
  vector[N] c_cond;
  int<lower = 1, upper = N_subj> subj[N]; 
}
parameters {
  real<lower = 0> sigma;
  vector<lower = 0>[2]  tau_u;   
  real alpha;
  real beta;
  matrix[2, N_subj] z_u;
  cholesky_factor_corr[2] L_u;
}

transformed parameters {
  matrix[N_subj, 2] u;
  u = (diag_pre_multiply(tau_u, L_u) * z_u)';
}

model {

  target += normal_lpdf(alpha| 6, 1.5);
  target += normal_lpdf(beta | 0, 0.1);
  target += normal_lpdf(sigma | 0, 1)  -
    normal_lccdf(0 | 0, 1);
  target += normal_lpdf(tau_u[1] | 0, 1)  - 
    normal_lccdf(0 | 0, 1);
  target += normal_lpdf(tau_u[2] | 0, 1)  - 
    normal_lccdf(0 | 0, 1);
  target += lkj_corr_cholesky_lpdf(L_u | 2);
  target += std_normal_lpdf(to_vector(z_u));


  target += lognormal_lpdf(rt | alpha + u[subj, 1] +
                        c_cond .* (beta + u[subj, 2]), sigma);
}
generated quantities {
  corr_matrix[2] rho_u= L_u * L_u';  
  array[N] real log_lik;

  for (n in 1:N){
    log_lik[n] = lognormal_lpdf(rt[n] | alpha + u[subj[n], 1] +
                        c_cond[n] * (beta + u[subj[n], 2]), sigma);
  }
}
