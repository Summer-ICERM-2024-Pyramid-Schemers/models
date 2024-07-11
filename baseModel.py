from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SEASONS = range(2010,2024)
# This list should be the names of the leagues as they should appear in plotting
# This only works properly when the ids begin at 1 and have a consistent step of 1
DEFAULT_LEAGUES = ['Premier League','Championship','League One','League Two']


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
    def plotBrierScores(cls, *, seasons=DEFAULT_SEASONS, leagues=DEFAULT_LEAGUES, title=None, filename=None):
        """
        Plots Brier scores from all leagues and seasons.
        """

        if title is None:
            title = cls._plot_title or f"{cls.__name__} Brier Score by Season and League"
        if filename is None:
            filename = cls._plot_filename or f"{cls.__name__}_brier_scores.png"
        leagues = [DEFAULT_LEAGUES[l-1] if isinstance(l,int) else l for l in leagues]
        briers = pd.DataFrame([[np.mean(cls.getBrierScores(season, league)) for league in range(1,len(leagues)+1)] for season in seasons], columns=leagues, index=seasons)

        briers.plot()
        plt.title(title)
        plt.xlabel('Season')
        plt.ylabel('Brier Score')
        plt.grid(True)

        plt.savefig(filename)
    