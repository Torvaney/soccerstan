functions {

  real tau(int x, int y, real rho, real mu1, real mu2) {
      if (x == 0 && y == 0)
        return 1 - (mu1 * mu2 * rho);
      else if (x == 0 && y == 1)
        return 1 + (mu1 * rho);
      else if (x == 1 && y == 0)
        return 1 + (mu2 * rho);
      else if (x == 1 && y == 1)
        return 1 - rho;
      else
        return 1;
  }

  real dixon_coles_log(int[] goals, real rho, real mu1, real mu2) {
    int home;
    int away;

    home = goals[1];
    away = goals[2];

    return poisson_lpmf(home | mu1) + poisson_lpmf(away | mu2) + log(tau(home, away, rho, mu1, mu2));
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
  real<lower=0> home_advantage;
  real<lower=0> offense_raw[n_teams];
  real<lower=0> defense_raw[n_teams];
  real rho;
}

transformed parameters {
  // Enforce sum-to-zero constraint
  real<lower=0> offense[n_teams];
  real<lower=0> defense[n_teams];

  for (t in 1:(n_teams-1)) {
    offense[t] = offense_raw[t];
    defense[t] = defense_raw[t];
  }

  offense[n_teams] = -sum(offense_raw);
  defense[n_teams] = -sum(defense_raw);
}

model {
  vector[n_games] mu1;
  vector[n_games] mu2;
  int score[2];

  // Priors (uninformative)
  offense ~ normal(0, 10);
  defense ~ normal(0, 10);
  home_advantage ~ normal(0, 100);

  for (g in 1:n_games) {
    score[1] = home_goals[g];
    score[2] = away_goals[g];

    mu1[g] = exp(offense[home_team[g]] + defense[away_team[g]] + home_advantage);
    mu2[g] = exp(offense[away_team[g]] + defense[home_team[g]]);

    score ~ dixon_coles(rho, mu1[g], mu2[g]);
  }
}
