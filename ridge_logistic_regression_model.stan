// Ridge logistic regression
data {
  int<lower=0> N_train;
  int<lower=0> P;
  
  matrix[N_train,P] X_train;
  int<lower=0,upper=1> y_train[N_train];
  
  int<lower=0> N_test;
  
  matrix[N_test,P] X_test;
  int<lower=0,upper=1> y_test[N_test];
  
  real<lower=0> scale_alpha;
  
  real<lower=0> mean_hypbeta;
  real<lower=0> scale_hypbeta;
}

parameters {
  
  real<lower=0> sigma;
  real alpha;     // intercept
  vector[P] beta;

}

model {
  
 alpha ~ normal(0,scale_alpha);
  
  for (j in 1:P){
    
     beta[j] ~ normal(0,sigma) ;
    
  }
  
  sigma ~ normal(mean_hypbeta,scale_hypbeta);
  y_train ~ bernoulli_logit(alpha+X_train*beta);
  
}

generated quantities {
  
  
 int<lower=0,upper=1> y_pred[N_test]; 
 vector[N_test] log_lik;
 
 for (i in 1:N_test){
   
   log_lik[i] = -bernoulli_logit_lpmf(y_test[i] | alpha+X_test[i,]*beta);
   y_pred[i] = bernoulli_logit_rng(alpha+X_test[i,]*beta);
   
   
 }
  
  
  
  
}

