data {
  int<lower = 1> N;
  int<lower=0> K;   // number of predictors
  matrix[N, K] X;   // model matrix
  int correct[N];
}
parameters {
  real alpha;
  vector[K] beta;
}
model {
  // priors including all constants
  target += normal_lpdf(alpha | 0, 1.5);
  target += normal_lpdf(beta | 0, .2);
  target += bernoulli_logit_glm_lpmf(correct | X, alpha, beta);
}

generated quantities {
  real average_accuracy = inv_logit(alpha); // for average effect between groups
  vector[K] change = inv_logit(beta); // for change effect
  vector[K] accuracy_group_1 = inv_logit(alpha + beta);
  vector[K] accuracy_group_2 = inv_logit(alpha - beta);
}
