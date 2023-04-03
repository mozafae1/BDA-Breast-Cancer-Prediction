data {
  int<lower=0> N_benign; // Number of benign training data samples
  int<lower=0> N_malignant; // Number of malignant training data samples
  int<lower=0> N_test; // Number of test samples
  int<lower=0> J; // Number of features
  vector[J] samples_benign[N_benign]; // Benign samples
  vector[J] samples_malignant[N_malignant]; // Malignant samples
  vector[J] samples_test[N_test]; // Test samples
  vector<lower=0, upper=1>[N_test] y_test ; // Test samples' labels; 1 for Malignant, 0 for benign
  real<lower=0, upper=1> prior_p; // Prior probability for a sample being malignant, equal to N_malign / (N_malign + N_benign)
}

parameters {
  vector[J] mu_benign;
  vector<lower=0>[J] sigma_benign;
  vector[J] mu_malignant;
  vector<lower=0>[J] sigma_malignant;
}

model {
  for (j in 1:J) {
     mu_benign[j] ~ normal(0, 10);
     sigma_benign[j] ~ inv_chi_square(0.1);
     mu_malignant[j] ~ normal(0, 10);
     sigma_malignant[j] ~ inv_chi_square(0.1);
  }
  
  for (j in 1:J) {
    samples_benign[, j] ~ normal(mu_benign[j], sigma_benign[j]);
    samples_malignant[, j] ~ normal(mu_malignant[j], sigma_malignant[j]);
  }
}

generated quantities {
  int<lower=0,upper=1> y_pred[N_test];
  real log_lik[N_test];

  for (i in 1:N_test) {
    real malignant_p = bernoulli_lpmf(1 | prior_p);
    real benign_p = bernoulli_lpmf(0 | prior_p);
    real likelihood_malignant_p = 0;
    real likelihood_benign_p = 0;
    for (j in 1:J) {
      likelihood_malignant_p += normal_lpdf(samples_test[i, j] | mu_malignant[j], sigma_malignant[j]);
      likelihood_benign_p += normal_lpdf(samples_test[i, j] | mu_benign[j], sigma_benign[j]);
    }
    malignant_p += likelihood_malignant_p;
    benign_p += likelihood_benign_p;
    if (malignant_p > benign_p) {
      y_pred[i] = 1;
    } else {
      y_pred[i] = 0;
    }
    if (y_test[i] == 1) {
      log_lik[i] = likelihood_malignant_p / (likelihood_malignant_p + likelihood_benign_p);
    } else {
      log_lik[i] = likelihood_benign_p / (likelihood_malignant_p + likelihood_benign_p);
    }
  }
}
