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
  matrix[N_subj, 2] u;
  corr_matrix[2] rho_u;
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
  target += lkj_corr_lpdf(rho_u | 2);
  
  for(i in 1:N_subj)
    target +=  multi_normal_lpdf(u[i,] |
                                 rep_row_vector(0, 2),
                                 quad_form_diag(rho_u, tau_u));
  target += lognormal_lpdf(rt | alpha + u[subj, 1] +
                        c_cond .* (beta + u[subj, 2]), sigma);
}

