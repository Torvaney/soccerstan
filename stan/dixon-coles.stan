functions {

  real tau(int x, int y, real rho, real mu1, real mu2) {
    real adj;

    if (x == 0 && y == 0)
      adj = 1 - (mu1 * mu2 * rho);
    else if (x == 0 && y == 1)
      adj = 1 + (mu1 * rho);
    else if (x == 1 && y == 0)
      adj = 1 + (mu2 * rho);
    else if (x == 1 && y == 1)
      adj = 1 - rho;
    else
      adj = 1;

    return adj;
  }

  real dixon_coles_log(int[] goals, real rho, real mu1, real mu2) {
    int home;
    int away;
    real prob;

    home = goals[1];
    away = goals[2];

    prob = poisson_lpmf(home | mu1) + poisson_lpmf(away | mu2) + log(tau(home, away, rho, mu1, mu2));
    return prob;
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
  real home_advantage;
  vector[n_teams - 1] offense_raw;
  vector[n_teams - 1] defense_raw;
  real<lower=0, upper=1> rho_raw;
}

transformed parameters {
  vector[n_teams] offense;
  vector[n_teams] defense;
  real rho;

  // Enforce sum-to-zero constraint
  offense[1:(n_teams - 1)]  = offense_raw;
  defense[1:(n_teams - 1)]  = defense_raw;
  offense[n_teams]          = -sum(offense_raw);
  defense[n_teams]          = -sum(defense_raw);

  // Enforce upper and lower bound on rho
  {
    vector[n_games] mu1 = exp(offense[home_team] - defense[away_team] + home_advantage);
    vector[n_games] mu2 = exp(offense[away_team] - defense[home_team]);
    real upper_bound = min({min(1. ./ (mu1 .* mu2)), 1});
    real lower_bound = max({max(-1. ./ mu1), max(-1. ./ mu2)});
    real range = upper_bound - lower_bound;
    rho = lower_bound + rho_raw * range;
  }

}

model {
  real mu1;
  real mu2;
  int score[2];

  // Priors (weakly informative)
  offense ~ normal(0, 2);
  defense ~ normal(0, 2);
  home_advantage ~ normal(0, 2);
  rho_raw ~ uniform(0, 1);

  for (g in 1:n_games) {
    score[1] = home_goals[g];
    score[2] = away_goals[g];

    mu1 = exp(offense[home_team[g]] + defense[away_team[g]] + home_advantage);
    mu2 = exp(offense[away_team[g]] + defense[home_team[g]]);

    score ~ dixon_coles(rho, mu1, mu2);
  }
}
