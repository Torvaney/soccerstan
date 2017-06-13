import os

from collections import namedtuple


def stanfile(fname):
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..', 'stan', fname
    )


SoccerModel = namedtuple(
    'Model',
    ['name', 'modelfile', 'parameters', 'team_parameters']
)


maher = SoccerModel(
    name='maher',
    modelfile=stanfile('maher.stan'),
    parameters=('home_advantage',),
    team_parameters=('offense', 'defense')
)


dixon_coles = SoccerModel(
    name='dixon-coles',
    modelfile=stanfile('dixon-coles.stan'),
    parameters=('home_advantage', 'rho'),
    team_parameters=('offense', 'defense')
)


karlis_ntzoufras = SoccerModel(
    name='karlis-ntzoufras',
    modelfile=stanfile('karlis-ntzoufras.stan'),
    parameters=('home_advantage', 'constant_mu', 'mixing_proportion'),
    team_parameters=('offense', 'defense')
)


model_map = {
    model.name: model for model in [
        maher, dixon_coles, karlis_ntzoufras
    ]
}
