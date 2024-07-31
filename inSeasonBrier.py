import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.miscmodels.ordinal_model import OrderedModel
from time import perf_counter

from src.getData import fetch_data_for_in_season_brier
from src.models.baseModel import BaseModel



DEFAULT_SEASONS = range(2010,2024)
# This list should be the names of the leagues as they should appear in plotting
# This only works properly when the ids begin at 1 and have a consistent step of 1
ALL_LEAGUES = ["Premier League","Championship","League One","League Two","Bundesliga","2. Bundesliga", 'scottish-premiership', 'scottish-championship', 'scottish-league-one', 'scottish-league-two']
COUNTRY_TO_LEAGUES = {None:[1,2,3,4,5,6,7,8,9,10], "england":[1,2,3,4], "germany":[5,6], "scotland": [7,8,9,10]}
#ALL_MODELS = ['Massey', 'Weighted Massey', 'Colley', 'Weighted Colley', 'Null', 'Transfermarkt', 'Betting Odds']
ALL_MODELS = ['Massey', 'Colley', 'Weighted Colley', 'Null', 'Betting Odds']



def getPredProb(season, league): 
    rank_games_80, rank_games_20, tmk_games_80, tmk_games_20  = fetch_data_for_in_season_brier(season, league)
    # Masssey
    massey_model = OrderedModel(rank_games_80['result'],rank_games_80['massey_diff'],distr='probit')
    massey_model = massey_model.fit(method='bfgs')
    predictions = [massey_model.predict(i)[0] for i in rank_games_20["massey_diff"]]
    massey_pred = rank_games_20[['match_id', 'result']].copy()
    massey_pred[["pred-loss","pred-draw","pred-win"]] = predictions
    '''
    # Weighted Masssey
    wtd_massey_model = OrderedModel(rank_games_80['result'],rank_games_80['wtd_massey_diff'],distr='probit')
    wtd_massey_model = wtd_massey_model.fit(method='bfgs')
    predictions = [wtd_massey_model.predict(i)[0] for i in rank_games_20["wtd_massey_diff"]]
    wtd_massey_pred = rank_games_20[['match_id', 'result']].copy()
    wtd_massey_pred[["pred-loss","pred-draw","pred-win"]] = predictions
    '''
    # Colley
    colley_model = OrderedModel(rank_games_80['result'],rank_games_80['colley_diff'],distr='probit')
    colley_model = colley_model.fit(method='bfgs')
    predictions = [colley_model.predict(i)[0] for i in rank_games_20["colley_diff"]]
    colley_pred = rank_games_20[['match_id', 'result']].copy()
    colley_pred[["pred-loss","pred-draw","pred-win"]] = predictions

    # Weighted Colley
    wtd_colley_model = OrderedModel(rank_games_80['result'],rank_games_80['wtd_colley_diff'],distr='probit')
    wtd_colley_model = wtd_colley_model.fit(method='bfgs')
    predictions = [wtd_colley_model.predict(i)[0] for i in rank_games_20["wtd_colley_diff"]]
    wtd_colley_pred = rank_games_20[['match_id', 'result']].copy()
    wtd_colley_pred[["pred-loss","pred-draw","pred-win"]] = predictions

    # Home Advantage
    home_model = OrderedModel(tmk_games_80['result'],tmk_games_80['Home'],distr = 'probit')
    home_model = home_model.fit(method='bfgs')
    predictions = [home_model.predict(i)[0] for i in tmk_games_20["Home"]]
    home_pred = tmk_games_20[['result']].copy()
    home_pred[["pred-loss","pred-draw","pred-win"]] = predictions
    '''
    # Transfermarkt model
    market_model = OrderedModel(tmk_games_80['result'],tmk_games_80[['Home', 'Value']],distr = 'probit')
    market_model = market_model.fit(method='bfgs')
    market_pred = tmk_games_20[['result']].copy()
    market_pred[["pred-loss","pred-draw","pred-win"]] = market_model.predict(tmk_games_20[['Home','Value']])
    '''
    odds = tmk_games_20

    #return massey_pred, wtd_massey_pred, colley_pred, wtd_colley_pred, home_pred, market_pred, odds
    return massey_pred, colley_pred, wtd_colley_pred, home_pred, odds



def getBrierScores(season, league):
    #massey_pred, wtd_massey_pred, colley_pred, wtd_colley_pred, home_pred, market_pred, odds = getPredProb(season, league)
    massey_pred, colley_pred, wtd_colley_pred, home_pred, odds = getPredProb(season, league)

    massey_brier = BaseModel._calc_brier_scores(massey_pred)
    #wtd_massey_brier = BaseModel._calc_brier_scores(wtd_massey_pred)
    colley_brier = BaseModel._calc_brier_scores(colley_pred)
    wtd_colley_brier = BaseModel._calc_brier_scores(wtd_colley_pred)
    home_brier = BaseModel._calc_brier_scores(home_pred)
    #market_brier = BaseModel._calc_brier_scores(market_pred)
    odds_brier = BaseModel._calc_brier_scores(odds, ["iOdds","drawOdds","jOdds"])

    #return massey_brier, wtd_massey_brier, colley_brier, wtd_colley_brier, home_brier, market_brier, odds_brier
    return massey_brier, colley_brier, wtd_colley_brier, home_brier, odds_brier




def plotBrierScores(*, seasons=DEFAULT_SEASONS, leagues=None, title=None, filename=None, country=None):
    if leagues is None:
        if isinstance(country,str):
            country = country.casefold()
        leagues = COUNTRY_TO_LEAGUES[country]

    leagues = [ALL_LEAGUES[l-1] if isinstance(l,int) else l for l in leagues]
    briers = {model: pd.DataFrame(index=seasons, columns=leagues) for model in ALL_MODELS}
    

    for season in seasons:
        for league in leagues:
            # Get the list of DataFrames
            dataframes = getBrierScores(season, ALL_LEAGUES.index(league) + 1)
            
            # Compute the mean Brier score for each DataFrame and store in the appropriate place in the dictionary
            for df, model in zip(dataframes, ALL_MODELS):
                mean_score = np.mean(df["brier_score"].to_list())
                briers[model].loc[season, league] = mean_score
    
    summary = []
    for model in ALL_MODELS:
        model_means = []
        for league in leagues:
            model_means.append(np.mean(briers[model][league]))
        summary.append(model_means)
    summary = pd.DataFrame(summary, columns=leagues, index=ALL_MODELS)
    '''
    for model in ALL_MODELS:
        title = f"{model} In Season Brier Scores"
        filename = f"{model}_inseason_brier_scores.png"

        brier_data = briers[model]

        brier_data.plot()
        plt.title(title)
        plt.xlabel("Season")
        plt.ylabel("Brier Score")
        plt.grid(True)
        plt.ylim(0.155, 0.235)
        plt.tight_layout()
        plt.savefig(f"images/{filename}")
    '''

    return summary


if __name__ == "__main__":
    start = perf_counter()
    plotBrierScores(country="england")
    end = perf_counter()
    print(end-start)