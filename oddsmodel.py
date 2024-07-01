import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from getData import getYearData

def getOddsBrierScores(season, league):
    """
    Returns a list containing odds Brier scores for each game of a given season in a given league
    
    season - A year between 2010 and 2023 (inclusive)
    league - An integer between 1 and 4 (inclusive)
        1 - Premier League
        2 - Championship
        3 - League One
        4 - League Two
    """
    finalTable = getYearData(season, league)
    brierScores = []
    for i in range(len(finalTable)):
        match = finalTable.iloc[i]
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
        score = 1/3 * (np.square(match['iOdds'] - win) +
                       np.square(match['drawOdds'] - draw) +
                       np.square(match['jOdds'] - loss))
        
        brierScores.append(score)

    return brierScores

def plotOddsBrierScores():
    """
    Plots odds Brier scores from all leagues and seasons.
    """
    briers = pd.DataFrame(columns=['Year', 'Premier League','Championship','League One','League Two'])

    for season in range(2010, 2024, 1):
        prem = np.mean(getOddsBrierScores(season, 1))
        Ch = np.mean(getOddsBrierScores(season, 2))
        l1 = np.mean(getOddsBrierScores(season, 3))
        l2 = np.mean(getOddsBrierScores(season, 4))
        briers = pd.concat([pd.DataFrame([[season, prem, Ch, l1, l2]], 
                                        columns=briers.columns), briers], 
                                        ignore_index=True)
        
    briers = briers.set_index('Year')
    briers.plot()

    plt.title('Betting Odds Brier Score by Season and League')
    plt.xlabel('Season')
    plt.ylabel('Brier Score')
    plt.grid(True)

    plt.show()