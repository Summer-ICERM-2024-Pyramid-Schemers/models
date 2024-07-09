from time import perf_counter

from statsmodels.miscmodels.ordinal_model import OrderedModel

from baseModel import BaseModel, DEFAULT_SEASONS
from getData import getNonYearData, getYearData


class HomeAdvModel(BaseModel):
    @classmethod
    def getModel(cls, season, league):
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
        model = OrderedModel(finalTable['result'],finalTable['Home'],distr = 'probit')

        # Fit the model
        return model.fit()

    @classmethod
    def getBrierScores(cls, season, league):
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
        model = cls.getModel(season, league)
        predictions = [model.predict(i)[0] for i in data["Home"]]
        # Careful, the model.predict should return in the order of loss, draw, win...
        data[["pred-loss","pred-draw","pred-win"]] = predictions
        return super()._calc_brier_scores(data)

    
    @classmethod
    def plotBrierScores(cls, seasons=DEFAULT_SEASONS, *args):
        return super().plotBrierScores(seasons, *args, title='Home Advantage Brier Score by Season and League', filename="HAModel.png")


if __name__ == "__main__":
    start = perf_counter()
    HomeAdvModel.plotBrierScores()
    end = perf_counter()
    print(end-start)
