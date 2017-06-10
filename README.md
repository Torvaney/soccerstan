# Soccerstan

Reproductions of models for football (soccer) matches in Stan/PyStan.

## How do I run the models?

It's pretty simple. You can just run the `src/soccerstan.py` module with the
file containing match data (from [football-data.co.uk](football-data.co.uk)) and which model you want to use as arguments.

For instance:

```
> python soccerstan.py 'data/example.csv' 'maher'
```

## Models

So far, the following models have been implemented:

 * Maher (1982)
 * Dixon and Coles (1997)

## Data

Data comes from [football-data.co.uk](football-data.co.uk).

The 2015/16 English Premier League season has been added as an example data file.
