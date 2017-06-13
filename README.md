# Soccerstan

Reproductions of models for football (soccer) matches in Stan/PyStan.

## How do I run the models?

It's pretty simple. You can just run `src/soccerstan.py` from the command line.
This requires the file containing match data
(from [football-data.co.uk](football-data.co.uk)) and which model you want to
use as arguments.

For instance:

```bash
> python soccerstan.py 'data/example.csv' 'karlis-ntzoufras'
```

## Models

So far, the following models have been implemented:

 * Maher (1982) - Modelling Association Football Scores - `maher`
 * Dixon and Coles (1997) - Modelling Association Football Scores and Inefficiencies in the Football Betting Market - `dixon-coles`
 * Karlis and Ntzoufras (2008) -  Bayesian modelling of football outcomes (using the Skellam's distribution) - `karlis-ntzoufras`

## Data

The main `soccerstan` script has been written to work with data from [football-data.co.uk](football-data.co.uk). The 2015/16 English Premier League season has been added as an example data file.

However, it will also work with any csv with columns labelled `home_team`, `away_team`, `home_goals`, `away_goals`.
