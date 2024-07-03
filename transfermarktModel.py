import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from getData import getNonYearData, getYearData
import time
from basemodel import DEFAULT_SEASONS, BaseModel

class TMModelOrderedProbit(BaseModel):
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
        model = getTMModel(season, league, 1)[0]
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


class TMModelOrderedProbitOLSGoalDiff(BaseModel):
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

        model = getTMModel(season, league, 2)
        OLSmodel = model[0]
        probitModel = model[1]
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


#TODO make more object oriented
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
        results = [model.fit()]

    # Peeters Model 2
    if model == 2:
        OLSmodel = smf.ols(formula = "goaldiff ~ Home + Value", data=finalTable)
        OLSresults = OLSmodel.fit()
        predictedDiff = OLSresults.predict(finalTable[['Home','Value']])
        outcome = finalTable['result']
        X_probit = predictedDiff.values.reshape(-1, 1)
        
        model = OrderedModel(outcome, X_probit, distr = 'probit')
        results = [OLSresults, model.fit()]

    return results

    

if __name__ == "__main__":
    start = time.time()
    TMModelOrderedProbit.plotBrierScores()
    TMModelOrderedProbitOLSGoalDiff.plotBrierScores()
    end = time.time()
    print(end-start)
