from time import perf_counter

import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

from baseModel import BaseModel
from getData import getNonYearData, getYearData


# An ordered probit model
class TMModelOrderedProbit(BaseModel):
    _plot_title = "Transfer Market Brier Score by Season and League"
    _plot_filename = "TMModel1.png"

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
        return super()._calc_brier_scores(data)
    
    @classmethod
    def plotBrierScores(cls, seasons=range(2012,2024), *args, title=None, filename=None):
        return super().plotBrierScores(seasons=seasons, *args, title=title, filename=filename)


# An ordered probit model trained on an OLS goal difference model
class TMModelOrderedProbitOLSGoalDiff(BaseModel):
    _plot_title = "Transfer Market Brier Score by Season and League"
    _plot_filename = "TMModel2.png"

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
        predictions = [probitModel.predict(i)[0] for i in intPred]

        data[["pred-loss","pred-draw","pred-win"]] = predictions
        return super()._calc_brier_scores(data)
    
    @classmethod
    def plotBrierScores(cls, seasons=range(2012,2024), *args, title=None, filename=None):
        return super().plotBrierScores(seasons=seasons, *args, title=title, filename=filename)


if __name__ == "__main__":
    start = perf_counter()
    TMModelOrderedProbit.plotBrierScores()
    TMModelOrderedProbitOLSGoalDiff.plotBrierScores()
    end = perf_counter()
    print(end-start)
