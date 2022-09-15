data {
  int<lower = 1> N_trials;
  array[5] int<lower = 0, upper = N_trials> ans;
}
parameters {
  real<lower = 0, upper = 1> w;
  real<lower = 0, upper = 1> x;
  real<lower = 0, upper = 1> y;
  real<lower = 0, upper = 1> z;
}
transformed parameters {
  simplex[5] theta;

    theta[1] = w * x * y * z * (1 - y) * z;
    theta[2] = w * x * y * (1 - z) * (1 - y) * (1 -z);
    theta[3] = (1 - w) * x;
    theta[4] = (1 - w) * (1 - x);
    theta[5] =  w * (1-x);

}
model {
  target += beta_lpdf(w | 2, 2);
  target += beta_lpdf(x | 2, 2);
  target += beta_lpdf(y | 2, 2);
  target += beta_lpdf(z | 2, 2);
  target += multinomial_lpmf(ans | theta);
}
generated quantities{
    array[5] int pred_ans;
  pred_ans = multinomial_rng(theta, 5);
}
