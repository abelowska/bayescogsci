data {
  int<lower=1> N;
  vector[N] signal;
  int<lower = 1> N_subj;
  int<lower = 1> N_item;
  vector[N] c_cloze;
  int<lower = 1, upper = N_subj> subj[N]; 
  int<lower = 1, upper = N_item> item[N]; 
}
parameters {
  vector<lower = 0>[2]  tau_u;   
  vector<lower = 0>[2]  tau_w;
  vector<lower = 0>[1]  tau_u_sigma;
  
  real alpha_sigma;   
  real alpha;
  real beta;

  matrix[2, N_subj] z_u;
  matrix[2, N_item] z_w;
  matrix[N_subj, 1] z_u_sigma;

  cholesky_factor_corr[2] L_u;
  cholesky_factor_corr[2] L_w;
}

transformed parameters {
  matrix[N_subj, 2] u;
  matrix[N_item, 2] w;
  vector[N_subj] u_sigma;
  vector[N_subj] sigma;
  u = (diag_pre_multiply(tau_u, L_u) * z_u)';
  w = (diag_pre_multiply(tau_w, L_w) * z_w)';
  u_sigma = z_u_sigma * tau_u_sigma;
  sigma = exp(alpha_sigma + u_sigma);

}

model {
  target += normal_lpdf(alpha| 0,10);
  target += normal_lpdf(beta | 0,10);
  target += normal_lpdf(alpha_sigma | 0, log(50));

//   target += normal_lpdf(sigma | 0, 50)  -
//     normal_lccdf(0 | 0, 50);
  target += normal_lpdf(tau_u | 0, 20) -
    2 * normal_lccdf(0 | 0, 20);
  target += normal_lpdf(tau_w | 0, 20) -
    2* normal_lccdf(0 | 0, 20);

  target += normal_lpdf(tau_u_sigma | 0, 5) - normal_lccdf(0 | 0, 5); 

  target += lkj_corr_cholesky_lpdf(L_u | 2);
  target += lkj_corr_cholesky_lpdf(L_w | 2);

  target += std_normal_lpdf(to_vector(z_u));
  target += std_normal_lpdf(to_vector(z_w));
  target += std_normal_lpdf(to_vector(z_u_sigma));


  target += normal_lpdf(signal | alpha + u[subj, 1] + w[item, 1]+
                        c_cloze .* (beta + u[subj, 2] + w[item, 2]), sigma[subj]);
}
generated quantities {
  corr_matrix[2] rho_u = L_u * L_u';
  corr_matrix[2] rho_w = L_w * L_w';

  array[N] real p_size_pred;
  p_size_pred = normal_rng(alpha + u[subj, 1] + w[item, 1]+
                        c_cloze .* (beta + u[subj, 2] + w[item, 2]), sigma[subj]);
}
