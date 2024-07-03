from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SEASONS = range(2010,2024)

class BaseModel(ABC):
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
    def plotBrierScores(cls, seasons=DEFAULT_SEASONS, *args, title, filename):
        """
        Plots Brier scores from all leagues and seasons.
        """

        briers = pd.DataFrame([[np.mean(cls.getBrierScores(season, league)) for league in range(1,5)] for season in seasons],
                            columns=['Premier League','Championship','League One','League Two'], index=seasons)

        briers.plot()

        plt.title(title)
        plt.xlabel('Season')
        plt.ylabel('Brier Score')
        plt.grid(True)

        plt.savefig(filename)
        plt.show()
    