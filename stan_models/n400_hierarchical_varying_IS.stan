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
  vector<lower = 0>[2]  tau_u;
  matrix[N_subj, 2] u;
}
model {
  target += normal_lpdf(alpha| 0,10);
  target += normal_lpdf(beta | 0,10);
  target += normal_lpdf(sigma | 0, 50)  -
    normal_lccdf(0 | 0, 50);
  target += normal_lpdf(tau_u[1] | 0, 20)  -
    normal_lccdf(0 | 0, 20);
  target += normal_lpdf(tau_u[2] | 0, 20)  -
    normal_lccdf(0 | 0, 20);
  
  target += normal_lpdf(u[,1] | 0, tau_u[1]);
  target += normal_lpdf(u[,2] | 0, tau_u[2]);
  
  target += normal_lpdf(signal | alpha + u[subj,1] +
                        c_cloze .* (beta + u[subj,2]), sigma);
}
