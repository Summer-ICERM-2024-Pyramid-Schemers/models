import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from getData import getNonYearData, getYearData

def getHAModel(season, league):
    """
    Returns a fit home advantage model for a given season and league trained on out of season data. 

    season - A year between 2010 and 2023 (inclusive)
    league - An integer between 1 and 4 (inclusive)
        1 - Premier League
        2 - Championship
        3 - League One
        4 - League Two
    """
    finalTable = getNonYearData(season, league)
    model = OrderedModel(finalTable['result'],
                        finalTable['Home'],
                        distr = 'probit')

    # Fit the model
    results = model.fit()

    return results

def getHABrierScores(season, league):
    """
    Returns a list containing Brier scores for each game of a given season in a given league
    
    season - A year between 2010 and 2023 (inclusive)
    league - An integer between 1 and 4 (inclusive)
        1 - Premier League
        2 - Championship
        3 - League One
        4 - League Two
    """
    data = getYearData(season, league)
    model = getHAModel(season, league)
    predictions = []
    for i in data['Home']:
        predictions.append(model.predict(i)[0])
    data['pred-loss'] = [pred[0] for pred in predictions]
    data['pred-draw'] = [pred[1] for pred in predictions]
    data['pred-win'] = [pred[2] for pred in predictions]
    brierScores = []

    for i in range(len(data)):
        match = data.iloc[i]
        result = match['result']
        win = 0
        draw = 0
        loss = 0
        if result == 1:
            win = 1
        elif result == 0:
            draw = 1
        elif result == -1:
            loss = 1
        score = 1/3 * (np.square(match['pred-win'] - win) +
                       np.square(match['pred-draw'] - draw) +
                       np.square(match['pred-loss'] - loss))
        
        brierScores.append(score)

    return brierScores

def plotHABrierScores():
    """
    Plots Brier scores from all leagues and seasons.
    """
    briers = pd.DataFrame(columns=['Year', 'Premier League','Championship','League One','League Two'])

    for season in range(2010, 2024, 1):
        prem = np.mean(getHABrierScores(season, 1))
        Ch = np.mean(getHABrierScores(season, 2))
        l1 = np.mean(getHABrierScores(season, 3))
        l2 = np.mean(getHABrierScores(season, 4))
        briers = pd.concat([pd.DataFrame([[season, prem, Ch, l1, l2]], 
                                        columns=briers.columns), briers], 
                                        ignore_index=True)
        
    briers = briers.set_index('Year')
    briers.plot()

    plt.title('Home Advantage Brier Score by Season and League')
    plt.xlabel('Season')
    plt.ylabel('Brier Score')
    plt.grid(True)

    plt.show()