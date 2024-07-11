from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SEASONS = range(2012,2024)

class BaseModel(ABC):
    _plot_title = None
    _plot_filename = None

    @classmethod
    def _calc_brier_scores(cls, data, win_draw_loss=["pred-win","pred-draw","pred-loss"]):
        reality_arr = data["result"].to_numpy()[:,None] == np.arange(1,-2,-1)[None,:]
        # ... while here we are using win, draw, loss
        diff_arr = data[win_draw_loss].to_numpy() - reality_arr
        brierScores = 1/3*np.sum(np.square(diff_arr),axis=1)
        return brierScores
    
    @classmethod
    def _calc_success_ratio(cls, data, win_draw_loss=["pred-win","pred-draw","pred-loss"]):
        reality_arr = data["result"].to_numpy()[:,None] == np.arange(1,-2,-1)[None,:]
        preds = data[win_draw_loss].to_numpy()

        max_indices = np.argmax(preds, axis=1)
        result = np.zeros_like(preds)
        result[np.arange(preds.shape[0]), max_indices] = 1

        equal_rows = np.all(result == reality_arr, axis=1)
        
        return np.mean(equal_rows)


    @classmethod
    @abstractmethod
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
        raise NotImplementedError()

    @classmethod
    def plotBrierScores(cls, country, seasons=DEFAULT_SEASONS, *args, title=None, filename=None):
        """
        Plots Brier scores from all leagues and seasons.
        """

        if title is None:
            title = cls._plot_title or f"{cls.__name__} Brier Score by Season and League"
        if filename is None:
            filename = cls._plot_filename or f"{cls.__name__}_brier_scores.png"
        if country.casefold() == "england":
            briers = pd.DataFrame([[np.mean(cls.getBrierScores(season, league)) for league in range(1,5)] for season in seasons],
                                columns=['Premier League','Championship','League One','League Two'], index=seasons)
        elif country.casefold() == "germany":
            briers = pd.DataFrame([[np.mean(cls.getBrierScores(season, league)) for league in range(5,7)] for season in seasons],
                                columns=['Bundesliga','2. Bundesliga'], index=seasons)
        else:
            raise Exception("Invalid country name")         

        briers.plot()

        plt.title(title)
        plt.xlabel('Season')
        plt.ylabel('Brier Score')
        plt.grid(True)

        plt.savefig(filename)
    