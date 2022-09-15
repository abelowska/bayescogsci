data {
  int<lower=1> N;
  vector[N] rt;
  vector[N] c_cond;
}
parameters {
  real<lower = 0> sigma;
  real alpha;
  real beta;
}

model {

  target += normal_lpdf(alpha| 6, 1.5);
  target += normal_lpdf(beta | 0, 0.1);
  target += normal_lpdf(sigma | 0, 1)  -
    normal_lccdf(0 | 0, 1);
  

  target += lognormal_lpdf(rt | alpha +
                        c_cond * beta, sigma);
}
generated quantities {
 array[N] real log_lik;
 for (n in 1:N){
    log_lik[n] = lognormal_lpdf(rt[n] | alpha  + c_cond * beta, sigma);
  }
}
