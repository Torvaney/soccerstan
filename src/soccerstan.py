import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan

import models


def stan_map(vector):
    """ Create a map of vector items : id. """
    unique_items = np.unique(vector)
    number_of_unique_items = len(unique_items)
    return dict(zip(unique_items, range(1, number_of_unique_items + 1)))


def read_data(fname):
    """ Read football-data.co.uk csv """
    data = (
        pd.read_csv(fname)
        .rename(columns={
                'HomeTeam': 'home_team',
                'AwayTeam': 'away_team',
                'FTHG': 'home_goals',
                'FTAG': 'away_goals'
            })
        .loc[lambda df: ~pd.isnull(df['home_goals'])]  # Remove future games
    )

    team_map = stan_map(data['home_team'])
    data['home_team_id'] = data['home_team'].replace(team_map)
    data['away_team_id'] = data['away_team'].replace(team_map)


    for col in ('home_goals', 'away_goals'):
        data[col] = [int(c) for c in data[col]]

    return data, team_map


def fit_model(data, team_map, model, **kwargs):
    stan_model = pystan.StanModel(model.modelfile)

    model_data = {
        'n_teams': len(data['home_team_id'].unique()),
        'n_games': len(data),
        'home_team': data['home_team_id'],
        'away_team': data['away_team_id'],
        'home_goals': data['home_goals'],
        'away_goals': data['away_goals']
    }

    fit = stan_model.sampling(data=model_data, **kwargs)
    output = fit.extract()

    # Tidy the output a little...
    reverse_map = {v: k for k, v in team_map.items()}
    for param in model.team_parameters:
        df = pd.DataFrame(output[param])
        df.columns = [reverse_map[id_ + 1] for id_ in df.columns]
        output[param] = df

    return output


def plot_output(model, output):
    for param in model.parameters:
        fig = plot_parameter(output[param], param, 'dimgray')
        fig.savefig('{}/../figures/{}-{}.png'.format(
            os.path.dirname(os.path.realpath(__file__)), model.name, param))

    for param in model.team_parameters:
        fig = plot_team_parameter(output[param], param, 0.05, 'dimgray')
        fig.savefig('{}/../figures/{}-{}.png'.format(
            os.path.dirname(os.path.realpath(__file__)), model.name, param))


def plot_parameter(data, title, alpha=0.05, axes_colour='dimgray'):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(data, bins=50, normed=True, color='black', edgecolor='None')

    # Add title
    fig.suptitle(title, fontsize=16, color=axes_colour)
    # Add axis labels
    ax.set_xlabel('', fontsize=16, color=axes_colour)
    ax.set_ylabel('', fontsize=16, color=axes_colour)

    # Change axes colour
    ax.spines["bottom"].set_color(axes_colour)
    ax.spines["left"].set_color(axes_colour)
    ax.tick_params(colors=axes_colour)
    # Remove top and bottom spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Remove extra ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    return fig


def plot_team_parameter(data, title, alpha=0.05, axes_colour='dimgray'):
    fig, ax = plt.subplots(figsize=(8, 6))

    upper = 1 - (alpha / 2)
    lower = 0 + (alpha / 2)

    for i, team in enumerate(data.columns):
        x_mean = np.median(data[team])
        x_lower = np.percentile(data[team], lower * 100)
        x_upper = np.percentile(data[team], upper * 100)

        ax.scatter(x_mean, i, alpha=1, color='black', s=25)
        ax.hlines(i, x_lower, x_upper, color='black')

    ax.set_ylim([-1, len(data.columns)])
    ax.set_yticks(list(range(len(data.columns))))
    ax.set_yticklabels(list(data.columns))

    # Add title
    fig.suptitle(title, ha='left', x=0.125, fontsize=18, color='k')

    # Change axes colour
    ax.spines["bottom"].set_color(axes_colour)
    ax.spines["left"].set_color(axes_colour)
    ax.tick_params(colors=axes_colour)

    # Remove top and bottom spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Location of file containing match data')
    parser.add_argument('model', help='Name of the model to be used')
    args = parser.parse_args()

    data, team_map = read_data(args.data)
    model = models.model_map[args.model]

    output = fit_model(data, team_map, model)

    plot_output(model, output)
