data {
  int<lower=1> n_teams;
  int<lower=1> n_games;
  int<lower=1, upper=n_teams> home_team[n_games];
  int<lower=1, upper=n_teams> away_team[n_games];
  int<lower=0> home_goals[n_games];
  int<lower=0> away_goals[n_games];
}

parameters {
  real home_advantage;
  real offense_raw[n_teams - 1];
  real defense_raw[n_teams - 1];
}

transformed parameters {
  // Enforce sum-to-zero constraint
  real offense[n_teams];
  real defense[n_teams];

  for (t in 1:(n_teams-1)) {
    offense[t] = offense_raw[t];
    defense[t] = defense_raw[t];
  }

  offense[n_teams] = -sum(offense_raw);
  defense[n_teams] = -sum(defense_raw);
}

model {
  vector[n_games] home_expected_goals;
  vector[n_games] away_expected_goals;

  // Priors (uninformative)
  offense ~ normal(0, 10);
  defense ~ normal(0, 10);
  home_advantage ~ normal(0, 100);

  for (g in 1:n_games) {
    home_expected_goals[g] = exp(offense[home_team[g]] + defense[away_team[g]] + home_advantage);
    away_expected_goals[g] = exp(offense[away_team[g]] + defense[home_team[g]]);

    home_goals[g] ~ poisson(home_expected_goals[g]);
    away_goals[g] ~ poisson(away_expected_goals[g]);
  }
}
