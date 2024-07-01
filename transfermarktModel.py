import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from getData import getNonYearData, getYearData

def getTMModel(season, league, model):
    
    """
    Returns a fit Peeters (2018) model for a given season and league trained on out of season data. 

    season - A year between 2010 and 2023 (inclusive)
    league - An integer between 1 and 4 (inclusive)
        1 - Premier League
        2 - Championship
        3 - League One
        4 - League Two
    model - Which model to return
        1 - An ordered probit model
        2 - An ordered probit model trained on an OLS goal difference model
    """
    finalTable = getNonYearData(season, league)

    # Peeters Model 1
    if model == 1:
        model = OrderedModel(finalTable['result'],
                            finalTable[['Home', 'Value']],
                            distr = 'probit')

    # Peeters Model 2
    if model == 2:
        OLSmodel = smf.ols(formula = "goaldiff ~ Home + Value", data=finalTable)
        OLSresults = OLSmodel.fit()
        predictedDiff = OLSresults.predict(finalTable[['Home','Value']])
        outcome = finalTable['result']
        X_probit = predictedDiff.values.reshape(-1, 1)
        
        model = OrderedModel(outcome, X_probit, distr = 'probit')

    # Fit the model
    results = model.fit()

    return results

def getTMBrierScores(season, league):
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
    model = getTMModel(season, league, 1)
    y = model.predict(data[['Home','Value']])
    y['result'] = data['result']
    brierScores = []

    for i in range(len(y)):
        match = y.iloc[i]
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
        score = 1/3 * (np.square(match[2] - win) +
                       np.square(match[1] - draw) +
                       np.square(match[0] - loss))
        
        brierScores.append(score)

    return brierScores


def plotTMBrierScores():
    """
    Plots Brier scores from all leagues and seasons.
    """
    briers = pd.DataFrame(columns=['Year', 'Premier League','Championship','League One','League Two'])

    for season in range(2010, 2024, 1):
        prem = np.mean(getTMBrierScores(season, 1))
        Ch = np.mean(getTMBrierScores(season, 2))
        l1 = np.mean(getTMBrierScores(season, 3))
        l2 = np.mean(getTMBrierScores(season, 4))
        briers = pd.concat([pd.DataFrame([[season, prem, Ch, l1, l2]], 
                                        columns=briers.columns), briers], 
                                        ignore_index=True)
        
    briers = briers.set_index('Year')
    briers.plot()

    plt.title('Transfer Market Brier Score by Season and League')
    plt.xlabel('Season')
    plt.ylabel('Brier Score')
    plt.grid(True)

    plt.show()