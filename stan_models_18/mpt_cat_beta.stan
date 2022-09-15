data {
  int<lower=1> N_obs;
  array[N_obs] int<lower=1,upper=5> w_ans;
  array[N_obs] real complexity;

}
parameters {
  real<lower=0,upper=1> a;
  real<lower=0,upper=1> t;
  real<lower=0,upper=1> c;
  real alpha_f;
  real beta_f;
}
transformed parameters {
  array[N_obs] simplex[5] theta;

  for(n in 1:N_obs){

    real f = inv_logit(alpha_f + complexity[n] * beta_f);

    //Pr_NR:
    theta[n, 1] = 1 - a;
    //Pr_Neologism:
    theta[n, 2] = a * (1 - t) * (1 - f) * (1 - c) + a * t * (1 - f) * (1 - c);
    //Pr_Formal:
    theta[n, 3] = a * (1 - t) * (1 - f) * c +  a * t * (1 - f) * c;
    //Pr_Mixed:
    theta[n, 4] = a * (1 - t) * f;
    //Pr_Correct:
    theta[n, 5] = a * t * f;
  }
}
model {
  target += beta_lpdf(a | 2, 2);
  target += beta_lpdf(t | 2, 2);
  target += normal_lpdf(alpha_f | 0, .5);
  target += normal_lpdf(beta_f | 0, .5);
  target += beta_lpdf(c | 2, 2);
  for(n in 1:N_obs)
    target += categorical_lpmf(w_ans[n] | theta[n]);
}
generated quantities{
    array[N_obs] int pred_w_ans;
  for(n in 1:N_obs)
    pred_w_ans[n] = categorical_rng(theta[n]);
}
