from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import ALL_LEAGUES, COUNTRY_TO_LEAGUES, DEFAULT_SEASONS, savefig_to_images_dir


class BaseModel(ABC):
    _plot_seasons = None
    _plot_title = None
    _plot_filename = None

    @classmethod
    def _calc_brier_scores(cls, data, win_draw_loss=["pred-win","pred-draw","pred-loss"]):
        indices = list(data.index.values)
        reality_arr = data["result"].to_numpy()[:,None] == np.arange(1,-2,-1)[None,:]
        # ... while here we are using win, draw, loss
        diff_arr = data[win_draw_loss].to_numpy() - reality_arr
        brierScores = 1/3*np.sum(np.square(diff_arr),axis=1)
        brierScores = pd.concat([pd.Series(indices,name='match_id'),pd.Series(brierScores,name='brier_score')], axis = 1)
        # brierScores is a dataframe of match ids and brier scores for the match
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
        Returns a dataframe containing match ids and Brier scores for each game of a given season in a given league
        
        season - A year between 2010 and 2023 (inclusive)
        league - An integer between 1 and 4 (inclusive)
            1 - Premier League
            2 - Championship
            3 - League One
            4 - League Two
        """
        raise NotImplementedError()

    @classmethod
    def plotBrierScores(cls, *, seasons=None, leagues=None, title=None, filename=None, country=None):
        """
        Plots Brier scores from all leagues and seasons.
        """
        if seasons is None:
            seasons = cls._plot_seasons or DEFAULT_SEASONS
        if leagues is None:
            if isinstance(country,str):
                country = country.casefold()
            leagues = COUNTRY_TO_LEAGUES[country]
        if title is None:
            title = cls._plot_title or f"{cls.__name__} Brier Score by Season and League"
        if filename is None:
            filename = cls._plot_filename or f"{cls.__name__}_brier_scores.png"
        
        leagues = [ALL_LEAGUES[l-1] if isinstance(l,int) else l for l in leagues]
        briers = pd.DataFrame([[np.mean(cls.getBrierScores(season, ALL_LEAGUES.index(league)+1)["brier_score"].to_list()) for league in leagues] for season in seasons], columns=leagues, index=seasons)

        briers.plot()
        plt.title(title)
        plt.xlabel("Season")
        plt.ylabel("Brier Score")
        plt.grid(True)

        savefig_to_images_dir(filename)
        
    @classmethod
    @abstractmethod
    def getSuccessRatio(cls, season, league):
        """
        Returns a list containing Brier scores for each game of a given season in a given league
        
        season - A year between 2010 and 2023 (inclusive)
        league - An integer between 1 and 4 (inclusive)
            1 - Premier League
            2 - Championship
            3 - League One
            4 - League Two
    
        """
        raise NotImplementedError()
    