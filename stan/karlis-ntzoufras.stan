functions {

  real skellam_log(int k, real mu1, real mu2) {
    real total;
    real log_prob;

    total = (- mu1 - mu2) + (log(mu1) - log(mu2)) * k / 2;
    log_prob = total + log(modified_bessel_first_kind(k, 2 * sqrt(mu1*mu2)));

    return log_prob;
  }

  real zero_inflated_skellam_log(int k, real mu1, real mu2, real p) {
    real base_prob;
    real prob;
    real log_prob;

    base_prob = exp(skellam_log(k, mu1, mu2));

    if (k == 0)
      prob = p + (1 - p) * base_prob;
    else
      prob = (1 - p) * base_prob;

    log_prob = log(prob);

    return log_prob;
  }

}

data {
  int<lower=1> n_teams;
  int<lower=1> n_games;
  int<lower=1, upper=n_teams> home_team[n_games];
  int<lower=1, upper=n_teams> away_team[n_games];
  int<lower=0> home_goals[n_games];
  int<lower=0> away_goals[n_games];
}

parameters {
  real<lower=0, upper=1> mixing_proportion;
  real constant_mu;
  real home_advantage;
  real offense_raw[n_teams - 1];
  real defense_raw[n_teams - 1];
}

transformed parameters {
  // Enforce sum-to-zero constraints
  vector[n_teams] offense;
  vector[n_teams] defense;

  for (t in 1:(n_teams-1)) {
    offense[t] = offense_raw[t];
    defense[t] = defense_raw[t];
  }

  offense[n_teams] = -sum(offense_raw);
  defense[n_teams] = -sum(defense_raw);
}

model {
  int goal_difference[n_games];
  vector[n_games] home_expected_goals;
  vector[n_games] away_expected_goals;
  vector[n_games] expected_goal_difference;

  // Priors (from the original paper)
  mixing_proportion ~ uniform(0, 1);
  offense ~ normal(0, 10000);
  defense ~ normal(0, 10000);
  home_advantage ~ normal(0, 10);
  constant_mu ~ normal(0, 100);

  for (g in 1:n_games) {
    goal_difference[g] = home_goals[g] - away_goals[g];

    home_expected_goals[g] = exp(
      constant_mu + home_advantage +
      offense[home_team[g]] + defense[away_team[g]]
    );

    away_expected_goals[g] = exp(
      constant_mu + offense[away_team[g]] + defense[home_team[g]]
    );

    goal_difference[g] ~ zero_inflated_skellam(
      home_expected_goals[g],
      away_expected_goals[g],
      mixing_proportion
    );
  }
}
