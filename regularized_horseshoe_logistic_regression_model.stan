// regularized horseshoe logistic regression
data {
  int<lower=0> N_train;
  int<lower=0> P;
  
  matrix[N_train,P] X_train;
  int<lower=0,upper=1> y_train[N_train];
  
  int<lower=0> N_test;
  
  matrix[N_test,P] X_test;
  int<lower=0,upper=1> y_test[N_test];
  
  real<lower=0> scale_global;
  
  real<lower=1> nu_local;
  real<lower=1> nu_global;
  
  real<lower=0> slab_scale;
  real<lower=0> slab_df;
  
}


parameters {
  
  real<lower=0> tau; // global shrinkage parameter
  vector<lower=0> [P] lambda; // local shrinkage parameter
  real alpha;     // intercept
  vector[P] z;
  real<lower=0> caux;
}

transformed parameters {
  
  vector[P] beta; // regression coefficients
  vector[N_train] f; 
  vector<lower=0>[P] lambda_tilde; 
  real<lower=0> c; 
  
  c = slab_scale * sqrt(caux);
  lambda_tilde = sqrt( c^2 * square(lambda) ./ (c^2 + tau^2*square(lambda)) );
  beta = z .* lambda_tilde*tau;
  f = alpha + X_train*beta;
}


model {
  
  alpha ~ normal(0,1000);
  lambda ~ student_t(nu_local, 0, 1);

  for (j in 1:P){
    
       z[j] ~ normal(0,1) ;
  }
  tau ~ student_t(nu_global, 0, scale_global);
  caux ~ inv_gamma(0.5*slab_df, 0.5*slab_df);
  y_train ~ bernoulli_logit(f);
}

generated quantities {
  
  
 int<lower=0,upper=1> y_pred[N_test]; 
 vector[N_test] log_lik;
 
 for (i in 1:N_test){
   
   log_lik[i] = -bernoulli_logit_lpmf(y_test[i] | alpha+X_test[i,]*beta);
   y_pred[i] = bernoulli_logit_rng(alpha+X_test[i,]*beta);
   
   
 }
  
  
  
  
}

  

