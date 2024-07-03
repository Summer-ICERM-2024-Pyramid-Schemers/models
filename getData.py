import sqlite3
import numpy as np
import pandas as pd
from functools import lru_cache
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))


def prepare_data(games_data: pd.DataFrame):
    home_vec = np.random.choice([1,-1],size=len(games_data))

    games_data = games_data.copy()

    games_data.loc[:,"goaldiff"] = games_data.loc[:,'fulltime_home_goals'] - games_data.loc[:,'fulltime_away_goals']
    games_data.loc[:,"Value"] = games_data.loc[:,'homeLogMV'] - games_data.loc[:,'awayLogMV']
    games_data.loc[:,"PurchaseValue"] = games_data.loc[:,'homeLogPV'] - games_data.loc[:,'awayLogPV']
    games_data.loc[:,["goaldiff","Value","PurchaseValue"]] *= home_vec[:,None]
    games_data.loc[:,"Home"] = home_vec
    games_data.loc[:,"result"] = np.sign(games_data.loc[:,"goaldiff"])

    probsData = normOddsVectorized(games_data.loc[:,['home_odds','draw_odds','away_odds']].to_numpy())
    games_data.loc[:,["iOdds","drawOdds","jOdds"]] = np.where(home_vec[:,None]==1,probsData,probsData[:,::-1])

    # TODO test .to_numpy
    return (games_data.loc[:,['result','goaldiff','Home','Value', 'PurchaseValue', 'iOdds', 'drawOdds', 'jOdds']]).apply(pd.to_numeric)
    
def getYearData(season, league):
    """
    Returns a dataframe containing matches from a season of a given league.
    
    season - A year between 2010 and 2023 (inclusive)
    league - An integer between 1 and 4 (inclusive)
        1 - Premier League
        2 - Championship
        3 - League One
        4 - League Two
    """

    data = getData()
    Games = data.query(f'season == {season} and league_id == {league}')
    return prepare_data(Games)

def getNonYearData(season, league):
    """
    Returns a dataframe containing matches from every season of a given league excluding the specified season.
    
    season - A year between 2010 and 2023 (inclusive)
    league - An integer between 1 and 4 (inclusive)
        1 - Premier League
        2 - Championship
        3 - League One
        4 - League Two
    """

    data = getData()
    Games = data.query(f'season != {season} and league_id == {league}')
    return prepare_data(Games)

@lru_cache(1)
def getData():
    """
    Returns raw data for all matches of all leagues
    """

    con = sqlite3.connect("english_football_data.sqlite")

    query = f"""
    SELECT id AS match_id, season, league_id, home_team_id, away_team_id, fulltime_home_goals, fulltime_away_goals, fulltime_result, 
    market_average_home_win_odds AS home_odds, market_average_draw_odds AS draw_odds, market_average_away_win_odds AS away_odds,
    home.starters_purchase_val + home.bench_purchase_val AS homePurchaseVal,
    home.starters_total_market_val + home.bench_total_market_val AS homeMarketVal,
    away.starters_purchase_val + away.bench_purchase_val AS awayPurchaseVal,
    away.starters_total_market_val + away.bench_total_market_val AS awayMarketVal
    FROM Matches
    JOIN LineupMarketvalues AS home
    ON Matches.id = home.match_id 
    AND Matches.home_team_id = home.team_id
    JOIN LineupMarketvalues AS away
    ON Matches.id = away.match_id
    AND Matches.away_team_id = away.team_id
    """

    Games = pd.read_sql_query(query, con)
    Games['homeLogMV'] = np.log1p(Games['homeMarketVal'])
    Games['awayLogMV'] = np.log1p(Games['awayMarketVal'])
    Games['homeLogPV'] = np.log1p(Games['homePurchaseVal'])
    Games['awayLogPV'] = np.log1p(Games['awayPurchaseVal'])

    return Games


def normOdds(iOdds, drawOdds, jOdds):
    juice = 1/iOdds + 1/drawOdds + 1/jOdds
    iProb = 1/(iOdds*juice)
    drawProb = 1/(drawOdds*juice)
    jProb = 1/(jOdds*juice)
    
    return iProb, drawProb, jProb

def normOddsVectorized(data):
    temp = 1/data
    juice = np.sum(temp,axis=1,keepdims=True)
    return temp/juice
