data {
  int<lower=1> N;
  int<lower = 1> N_subj; // level
  vector[N] c_cloze; // independent variable
  vector[N] signal; // dependent variable
  int<lower = 1, upper = N_subj> subj[N]; //an array of integers;
}
parameters {
  real alpha;
  real beta;
  real<lower = 0> sigma;
  real<lower = 0>  tau_u;
  vector[N_subj] u;
}
model {
  target += normal_lpdf(alpha| 0,10);
  target += normal_lpdf(beta | 0,10);
  target += normal_lpdf(sigma | 0, 50)  -
    normal_lccdf(0 | 0, 50);
  target += normal_lpdf(tau_u | 0, 20)  -
    normal_lccdf(0 | 0, 20);
  
  target += normal_lpdf(u | 0, tau_u);
  target += normal_lpdf(signal | alpha + u[subj] +
                        c_cloze * beta, sigma);
}
