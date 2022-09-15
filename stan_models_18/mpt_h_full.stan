data {
  int<lower = 1> N_obs;
  array[N_obs] int<lower = 1, upper = 5> w_ans;
  array[N_obs] real complexity;
  int<lower = 1> N_subj;
  array[N_obs] int<lower = 1, upper = N_subj> subj;
}
parameters {  
  real alpha_a;
  real alpha_c;
  real alpha_t;
  real alpha_f;

  real beta_f;

  vector<lower = 0>[4]  tau;

  matrix[4, N_subj] z;

  cholesky_factor_corr[4] L;
}
transformed parameters {
  array[N_obs] simplex[5] theta;

  matrix[N_subj, 4] u;
  u = (diag_pre_multiply(tau, L) * z)';

  for (n in 1:N_obs){
    real a = inv_logit(alpha_a + u[subj[n], 1]);
    real c = inv_logit(alpha_c + u[subj[n], 2]);
    real t = inv_logit(alpha_t + u[subj[n], 3]);
    real f = inv_logit(alpha_f + u[subj[n], 4] + complexity[n] * beta_f);
    
    //Pr_NR
    theta[n, 1] = 1 - a;
    //Pr_Neologism
    theta[n, 2] = a * (1 - t) * (1 - f) * (1 - c) + a * t * (1 - f) * (1 - c);
    //Pr_Formal
    theta[n, 3] = a * (1 - t) * (1 - f) * c + a * t * (1 - f) * c;
    //Pr_Mixed
    theta[n, 4] = a * (1 - t) * f;
    //Pr_Correct
    theta[n, 5] = a * t * f;
  }
}
model {

  target += normal_lpdf(alpha_a | 0, 2);
  target += normal_lpdf(alpha_t | 0, 2);
  target += normal_lpdf(alpha_c | 0, 2);
  target += normal_lpdf(alpha_f | 0, 2);

  target += normal_lpdf(beta_f | 0, 2);

  target += normal_lpdf(tau | 0, 1)  -
    4 * normal_lccdf(0 | 0, 1);

  target += lkj_corr_cholesky_lpdf(L | 2);
  target += std_normal_lpdf(to_vector(z));

  for(n in 1:N_obs)
    target +=  categorical_lpmf(w_ans[n] | theta[n]);
}
generated quantities{
  corr_matrix[4] rho_u= L * L';
  array[N_obs] int<lower = 1, upper = 5> pred_w_ans;
  for(n in 1:N_obs)
    pred_w_ans[n] = categorical_rng(theta[n]);
}
