from time import perf_counter

import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel

from basemodel import BaseModel, DEFAULT_SEASONS
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
        #TODO vectorize method
        data = getYearData(season, league)
        model = cls.getModel(season, league)
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

    
    @classmethod
    def plotBrierScores(cls, seasons=DEFAULT_SEASONS, *args):
        return super().plotBrierScores(seasons, *args, title='Home Advantage Brier Score by Season and League', filename="HAModel.png")


if __name__ == "__main__":
    start = perf_counter()
    HomeAdvModel.plotBrierScores()
    end = perf_counter()
    print(end-start)
