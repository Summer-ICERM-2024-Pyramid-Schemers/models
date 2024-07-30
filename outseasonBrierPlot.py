import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.miscmodels.ordinal_model import OrderedModel
from time import perf_counter

from oddsModel import BettingOddsModel
from homeAdvModel import HomeAdvModel
from transfermarktModel import TMModelOrderedProbit
from masseyModel import MasseyModel, WeightedMasseyModel
from colleyModel import ColleyModel, WeightedColleyModel



DEFAULT_SEASONS = range(2012,2024)
# This list should be the names of the leagues as they should appear in plotting
# This only works properly when the ids begin at 1 and have a consistent step of 1
ALL_LEAGUES = ["Premier League","Championship","League One","League Two","Bundesliga","2. Bundesliga"]
COUNTRY_TO_LEAGUES = {None:[1,2,3,4,5,6], "england":[1,2,3,4], "germany":[5,6]}
ALL_MODELS = ['Massey', 'Weighted Massey', 'Colley', 'Weighted Colley', 'Null', 'Transfermarkt', 'Betting Odds']



def getBrierScores(season, league):

    massey_brier = MasseyModel.getBrierScores(season, league)
    wtd_massey_brier = WeightedMasseyModel.getBrierScores(season, league)
    colley_brier = ColleyModel.getBrierScores(season, league)
    wtd_colley_brier = WeightedColleyModel.getBrierScores(season, league)
    home_brier = HomeAdvModel.getBrierScores(season, league)
    market_brier = TMModelOrderedProbit.getBrierScores(season, league)
    odds_brier = BettingOddsModel.getBrierScores(season, league)
    return massey_brier, wtd_massey_brier, colley_brier, wtd_colley_brier, home_brier, market_brier, odds_brier



def plotBrierScores(*, seasons=DEFAULT_SEASONS, leagues=None, title=None, filename=None, country=None):
    if leagues is None:
        if isinstance(country,str):
            country = country.casefold()
        leagues = COUNTRY_TO_LEAGUES[country]

    leagues = [ALL_LEAGUES[l-1] if isinstance(l, int) else l for l in leagues]
    briers = {model: {league: [] for league in leagues} for model in ALL_MODELS}

    year_briers = {model: pd.DataFrame(index=seasons, columns=leagues) for model in ALL_MODELS}

    # Compute the Brier scores for all games across seasons and leagues
    for season in seasons:
        for league in leagues:
            # Get the list of DataFrames
            dataframes = getBrierScores(season, ALL_LEAGUES.index(league) + 1)
            
            # Store all Brier scores for each model
            for df, model in zip(dataframes, ALL_MODELS):
                brier_score = df["brier_score"].tolist()
                briers[model][league].extend(brier_score)
                year_briers[model].loc[season, league] = np.mean(brier_score)

    # Compute the mean Brier scores for each model and league
    summary_data = {
        model: [np.mean(briers[model][league]) for league in leagues]
        for model in ALL_MODELS
    }

    summary = pd.DataFrame(summary_data, index=leagues, columns=ALL_MODELS) 

    for model in ALL_MODELS:
        title = f"{model}"
        filename = f"{model}_outseason_brier_scores.png"

        brier_data = year_briers[model]

        brier_data.plot()
        plt.title(title, fontsize=20)
        plt.xlabel("Season", fontsize=16)
        plt.ylabel("Brier Score", fontsize=16)
        plt.grid(True)
        plt.ylim(0.17, 0.225)
        plt.tight_layout()
        plt.savefig(filename)

    return summary, year_briers


if __name__ == "__main__":
    start = perf_counter()
    plotBrierScores(country="england")
    end = perf_counter()
    print(end-start)