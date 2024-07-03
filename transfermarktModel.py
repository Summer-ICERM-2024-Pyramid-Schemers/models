from time import perf_counter

import numpy as np
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

from basemodel import DEFAULT_SEASONS, BaseModel
from getData import getNonYearData, getYearData


# An ordered probit model
class TMModelOrderedProbit(BaseModel):
    @classmethod
    def getModel(cls, season, league):
        """
        Returns a fit Peeters (2018) model for a given season and league trained on out of season data. 

        season - A year between 2010 and 2023 (inclusive)
        league - An integer between 1 and 4 (inclusive)
            1 - Premier League
            2 - Championship
            3 - League One
            4 - League Two
        """
        finalTable = getNonYearData(season, league)
        # Peeters Model 1
        model = OrderedModel(finalTable['result'],finalTable[['Home', 'Value']],distr = 'probit')
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
        data[["pred-loss","pred-draw","pred-win"]] = model.predict(data[['Home','Value']])
        
        brierScores = []
        #TODO vectorize
        for idx,pwin,pdraw,ploss,result in data.loc[:,["pred-win","pred-draw","pred-loss","result"]].itertuples():
            win = 0
            draw = 0
            loss = 0
            if result == 1:
                win = 1
            elif result == 0:
                draw = 1
            elif result == -1:
                loss = 1
            else:
                raise RuntimeError("HELP")
            score = 1/3 * (np.square(pwin - win) +
                        np.square(pdraw - draw) +
                        np.square(ploss - loss))
            
            brierScores.append(score)
        
        return brierScores
    
    @classmethod
    def plotBrierScores(cls, seasons=DEFAULT_SEASONS, *args):
        return super().plotBrierScores(seasons, *args, title="Transfer Market Brier Score by Season and League", filename="TMModel1.png")


# An ordered probit model trained on an OLS goal difference model
class TMModelOrderedProbitOLSGoalDiff(BaseModel):
    @classmethod
    def getModel(cls, season, league):    
        """
        Returns a fit Peeters (2018) model for a given season and league trained on out of season data. 

        season - A year between 2010 and 2023 (inclusive)
        league - An integer between 1 and 4 (inclusive)
            1 - Premier League
            2 - Championship
            3 - League One
            4 - League Two
        """
        finalTable = getNonYearData(season, league)
        # Peeters Model 2
        OLSmodel = smf.ols(formula = "goaldiff ~ Home + Value", data=finalTable)
        OLSresults = OLSmodel.fit()
        predictedDiff = OLSresults.predict(finalTable[['Home','Value']])
        outcome = finalTable['result']
        X_probit = predictedDiff.values.reshape(-1, 1)
        
        model = OrderedModel(outcome, X_probit, distr = 'probit')
        return OLSresults, model.fit()

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

        OLSmodel,probitModel = cls.getModel(season, league)
        intPred = OLSmodel.predict(data[['Home','Value']])
        predictions = np.array([probitModel.predict(i)[0] for i in intPred])

        data[["pred-loss","pred-draw","pred-win"]] = predictions
        
        brierScores = []
        #TODO vectorize
        for idx,pwin,pdraw,ploss,result in data.loc[:,["pred-win","pred-draw","pred-loss","result"]].itertuples():
            win = 0
            draw = 0
            loss = 0
            if result == 1:
                win = 1
            elif result == 0:
                draw = 1
            elif result == -1:
                loss = 1
            else:
                raise RuntimeError("HELP")
            score = 1/3 * (np.square(pwin - win) +
                        np.square(pdraw - draw) +
                        np.square(ploss - loss))
            
            brierScores.append(score)

        return brierScores
    
    @classmethod
    def plotBrierScores(cls, seasons=DEFAULT_SEASONS, *args):
        return super().plotBrierScores(seasons, *args, title="Transfer Market Brier Score by Season and League", filename="TMModel2.png")


if __name__ == "__main__":
    start = perf_counter()
    TMModelOrderedProbit.plotBrierScores()
    TMModelOrderedProbitOLSGoalDiff.plotBrierScores()
    end = perf_counter()
    print(end-start)
