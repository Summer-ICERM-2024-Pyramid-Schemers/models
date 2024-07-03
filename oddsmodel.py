from time import perf_counter

import numpy as np

from basemodel import BaseModel, DEFAULT_SEASONS
from getData import getYearData


class BettingOddsModel(BaseModel):
    @classmethod
    def getBrierScores(cls, season, league):
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
    
    @classmethod
    def plotBrierScores(cls, seasons=DEFAULT_SEASONS, *args):
        return super().plotBrierScores(seasons, *args, title='Betting Odds Brier Score by Season and League', filename="BOModel.png")


if __name__ == "__main__":
    start = perf_counter()
    BettingOddsModel.plotBrierScores()
    end = perf_counter()
    print(end-start)
