from time import perf_counter

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
        return super()._calc_brier_scores(finalTable,["iOdds","drawOdds","jOdds"])
        
    @classmethod
    def plotBrierScores(cls, seasons=DEFAULT_SEASONS, *args):
        return super().plotBrierScores(seasons, *args, title='Betting Odds Brier Score by Season and League', filename="BOModel.png")


if __name__ == "__main__":
    start = perf_counter()
    BettingOddsModel.plotBrierScores()
    end = perf_counter()
    print(end-start)
