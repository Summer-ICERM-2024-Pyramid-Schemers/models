from functools import lru_cache
import sqlite3

import numpy as np
import pandas as pd

DATABASE_FILEPATH = "english_football_data.sqlite"

def _prepare_data_for_transfermarkt_model(games_data: pd.DataFrame):
    home_vec = np.random.choice([1,-1],size=len(games_data))

    games_data = games_data.copy()

    games_data["goaldiff"] = games_data['fulltime_home_goals'] - games_data['fulltime_away_goals']
    games_data["Value"] = games_data['homeLogMV'] - games_data['awayLogMV']
    games_data["PurchaseValue"] = games_data['homeLogPV'] - games_data['awayLogPV']
    games_data[["goaldiff","Value","PurchaseValue"]] *= home_vec[:,None]
    games_data["Home"] = home_vec
    games_data["result"] = np.sign(games_data["goaldiff"])

    probsData = norm_odds_vectorized(games_data[['home_odds','draw_odds','away_odds']].to_numpy())
    games_data.loc[:,["iOdds","drawOdds","jOdds"]] = np.where(home_vec[:,None]==1,probsData,probsData[:,::-1])

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
    data = _fetch_data_for_transfermarkt_model()
    Games = data.query(f'season == {season} and league_id == {league}')
    return _prepare_data_for_transfermarkt_model(Games)

def getNonYearData(season, league):
    """
    Returns a dataframe containing matches from every season of a given league prior to the specified season.
    
    season - A year between 2012 and 2023 (inclusive)
    league - An integer between 1 and 4 (inclusive)
        1 - Premier League
        2 - Championship
        3 - League One
        4 - League Two
    """
    data = _fetch_data_for_transfermarkt_model()
    Games = data.query(f'season < {season} and league_id == {league}')
    return _prepare_data_for_transfermarkt_model(Games)

@lru_cache(1)
def _fetch_data_for_transfermarkt_model():
    """
    Returns raw data for all matches of all leagues
    """

    con = sqlite3.connect(DATABASE_FILEPATH)

    query = f"""
    SELECT id AS match_id, season, league_id, home_team_id, away_team_id, fulltime_home_goals, fulltime_away_goals, fulltime_result, 
    market_average_home_win_odds AS home_odds, market_average_draw_odds AS draw_odds, market_average_away_win_odds AS away_odds,
    home.starters_purchase_val + home.bench_purchase_val AS homePurchaseVal,
    home.starters_total_market_val + home.bench_total_market_val AS homeMarketVal,
    away.starters_purchase_val + away.bench_purchase_val AS awayPurchaseVal,
    away.starters_total_market_val + away.bench_total_market_val AS awayMarketVal,
    ((home.starters_avg_age * 11) + (home.bench_avg_age * home.bench_size)) / (11 + home.bench_size) AS homeAge,
    ((away.starters_avg_age * 11) + (away.bench_avg_age * away.bench_size)) / (11 + away.bench_size) AS awayAge
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

def norm_odds_vectorized(data):
    temp = 1/data
    juice = np.sum(temp,axis=1,keepdims=True)
    return temp/juice

#TODO remove and replace
@lru_cache(1)
def fetch_data_for_massey_model():
    con = sqlite3.connect(DATABASE_FILEPATH)

    gamesQuery = f"""
    SELECT season, league_id, date, home_team_id, away_team_id, fulltime_home_goals, fulltime_away_goals
    FROM Matches
    """

    mvQuery = f"""
    SELECT season, league_id, team_id, avg_market_val
    FROM TeamMarketvalues
    """

    Games = pd.read_sql_query(gamesQuery, con)
    marketValues = pd.read_sql_query(mvQuery, con)
    con.close()

    Games['result'] = np.sign(Games["fulltime_home_goals"] - Games["fulltime_away_goals"]).astype(int)

    return Games, marketValues

@lru_cache(1)
def fetch_data_for_massey_eos_eval():
    con = sqlite3.connect(DATABASE_FILEPATH)

    gamesQuery = f"""
    SELECT season, league_id, date, home_team_id, away_team_id, fulltime_home_goals, fulltime_away_goals, fulltime_result
    FROM Matches
    """

    rankQuery = f"""
    SELECT season, league_id, team_id, ranking	
    FROM EOSStandings
    """

    mvQuery = f"""
    SELECT season, league_id, team_id, avg_market_val
    FROM TeamMarketvalues
    """

    Games = pd.read_sql_query(gamesQuery, con)
    ranking = pd.read_sql_query(rankQuery, con)
    marketValues = pd.read_sql_query(mvQuery, con)
    con.close()

    return Games, ranking, marketValues
