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
  target += normal_lpdf(beta | 0, .1);

  target += bernoulli_lpmf(correct| Phi(alpha + X * beta));
  
}

// generated quantities {
//   real average_accuracy = inv_Phi(alpha);
//   vector[K] change_acc = inv_Phi(alpha) - inv_phi(alpha - beta);
// }
