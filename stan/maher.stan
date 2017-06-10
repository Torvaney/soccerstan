data {
  int<lower=1> n_teams;
  int<lower=1> n_games;
  int<lower=1, upper=n_teams> home_team[n_games];
  int<lower=1, upper=n_teams> away_team[n_games];
  int<lower=0> home_goals[n_games];
  int<lower=0> away_goals[n_games];
}

parameters {
  real<lower=0> home_advantage;
  real<lower=0> offense_raw[n_teams];
  real<lower=0> defense_raw[n_teams];
}

transformed parameters {
  // Enforce mean-to-one constraint
  real<lower=0> offense[n_teams];
  real<lower=0> defense[n_teams];

  for (t in 1:(n_teams)) {
    offense[t] = offense_raw[t] / mean(offense_raw);
    defense[t] = defense_raw[t] / mean(defense_raw);
  }
}

model {
  vector[n_games] home_expected_goals;
  vector[n_games] away_expected_goals;

  // Priors (uninformative)
  offense ~ normal(1, 100);
  defense ~ normal(1, 100);
  home_advantage ~ normal(1, 10);

  for (g in 1:n_games) {
    home_expected_goals[g] = offense[home_team[g]] * defense[away_team[g]] * home_advantage;
    away_expected_goals[g] = offense[away_team[g]] * defense[home_team[g]];

    home_goals[g] ~ poisson(home_expected_goals[g]);
    away_goals[g] ~ poisson(away_expected_goals[g]);
  }
}
