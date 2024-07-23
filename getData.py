from functools import lru_cache
import sqlite3

import numpy as np
import pandas as pd

from weighted_colley_engine import WeightedColleyEngine
from weighted_massey_engine import WeightedMasseyEngine

DATABASE_FILEPATH = "football_data.sqlite"


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

    query = """
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

@lru_cache(1)
def fetch_data_for_massey_eos_eval():

    con = sqlite3.connect(DATABASE_FILEPATH)

    gamesQuery = """
    SELECT season, id AS match_id, league_id, date, home_team_id, away_team_id, fulltime_home_goals, fulltime_away_goals
    FROM Matches
    WHERE league_id IN (1, 2, 3, 4)
    """

    rankQuery = """
    SELECT season, league_id, team_id, ranking	
    FROM EOSStandings
    WHERE league_id IN (1, 2, 3, 4)
    """

    mvQuery = """
    SELECT season, league_id, team_id, avg_market_val
    FROM TeamMarketvalues
    WHERE league_id IN (1, 2, 3, 4)
    """

    Games = pd.read_sql_query(gamesQuery, con)
    ranking = pd.read_sql_query(rankQuery, con)
    marketValues = pd.read_sql_query(mvQuery, con)
    con.close()

    Games['result'] = np.sign(Games["fulltime_home_goals"] - Games["fulltime_away_goals"]).astype(int)

    return Games, ranking, marketValues

def fetch_data_for_colley_accuracy(season,league):
    con = sqlite3.connect(DATABASE_FILEPATH)

    gamesQuery = """
    SELECT id AS match_id, home_team_id, away_team_id, fulltime_home_goals, fulltime_away_goals, date
    FROM Matches
    WHERE season = @season
    AND league_id = @league
    """
    
    Games = pd.read_sql_query(gamesQuery, con, params={"season":season,"league":league})

    home_goals_list = Games.loc[:,"fulltime_home_goals"]
    away_goals_list = Games.loc[:,"fulltime_away_goals"]
    home_teams_list = Games.loc[:,'home_team_id']
    away_teams_list = Games.loc[:,'away_team_id']
    date_list = Games.loc[:,'date']

    N = 8*len(home_goals_list)//10
    colley_ratings = WeightedColleyEngine.get_ratings(home_goals_list[:N],away_goals_list[:N],home_teams_list[:N],away_teams_list[:N])

    wtd_colley_ratings = WeightedColleyEngine.get_ratings(home_goals_list[:N],away_goals_list[:N],home_teams_list[:N],away_teams_list[:N], date_list[:N])
    
    c_dictionary = colley_ratings.set_index('team')['rating'].to_dict()
    wc_dictionary = wtd_colley_ratings.set_index('team')['rating'].to_dict()

    Games['home_team_colley'] = Games['home_team_id'].map(c_dictionary)
    Games['away_team_colley'] = Games['away_team_id'].map(c_dictionary)

    Games['home_team_wtd_colley'] = Games['home_team_id'].map(wc_dictionary)
    Games['away_team_wtd_colley'] = Games['away_team_id'].map(wc_dictionary)
    
    Games = Games.iloc[N:]
    Games.reset_index(drop=True, inplace=True)
    Games['result'] = np.sign(Games["fulltime_home_goals"] - Games["fulltime_away_goals"]).astype(int)

    return Games

def fetch_data_for_massey_accuracy(season,league):
    con = sqlite3.connect(DATABASE_FILEPATH)

    gamesQuery = """
    SELECT id AS match_id, date, home_team_id, away_team_id, fulltime_home_goals, fulltime_away_goals, fulltime_result
    FROM Matches
    WHERE season = @season
    AND league_id = @league
    """

    mvQuery = """
    SELECT season, league_id, team_id, avg_market_val
    FROM TeamMarketvalues
    WHERE season = @season
    AND league_id = @league
    """
    
    Games = pd.read_sql_query(gamesQuery, con, params={"season":season,"league":league})
    avg_mv = pd.read_sql_query(mvQuery, con, params={"season":season,"league":league})

    home_goals_list = Games.loc[:,"fulltime_home_goals"]
    away_goals_list = Games.loc[:,"fulltime_away_goals"]
    home_teams_list = Games.loc[:,'home_team_id']
    away_teams_list = Games.loc[:,'away_team_id']
    match_date_list = Games.loc[:,'date']
    N = 8*len(home_goals_list)//10
    m_ratings, _ = WeightedMasseyEngine.get_ratings(home_goals_list[:N], away_goals_list[:N],
                                                                  home_teams_list[:N], away_teams_list[:N])
    wm_ratings, home_advantage = WeightedMasseyEngine.get_ratings(home_goals_list[:N], away_goals_list[:N],
                                                                  home_teams_list[:N], away_teams_list[:N],
                                                                  match_date=match_date_list[:N], avg_mv=avg_mv)

    m_dictionary = m_ratings.set_index('team')['rating'].to_dict()
    wm_dictionary = wm_ratings.set_index('team')['mv_rating'].to_dict()

    Games['home_team_massey'] = Games['home_team_id'].map(m_dictionary)
    Games['away_team_massey'] = Games['away_team_id'].map(m_dictionary)

    Games['home_team_wtd_massey'] = Games['home_team_id'].map(wm_dictionary)
    Games['away_team_wtd_massey'] = Games['away_team_id'].map(wm_dictionary)

    Games['home_team_wtd_massey'] += home_advantage
    
    Games = Games.iloc[N:]
    Games.reset_index(drop=True, inplace=True)
    Games['result'] = np.sign(Games["fulltime_home_goals"] - Games["fulltime_away_goals"]).astype(int)

    return Games


def fetch_data_for_in_season_brier(season, league):
    '''
    return 1) first 80% data in each season for transfermarkt models & rank models
            2) last 20% for evaluation
    '''
    con = sqlite3.connect(DATABASE_FILEPATH)

    rankQuery = """
    SELECT id AS match_id, date, home_team_id, away_team_id, fulltime_home_goals, fulltime_away_goals, fulltime_result
    FROM Matches
    WHERE season = @season
    AND league_id = @league
    """

    mvQuery = """
    SELECT season, league_id, team_id, avg_market_val
    FROM TeamMarketvalues
    WHERE season = @season
    AND league_id = @league
    """

    tmkQuery = """
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


    rank_games = pd.read_sql_query(rankQuery, con, params={"season":season,"league":league})
    avg_mv = pd.read_sql_query(mvQuery, con, params={"season":season,"league":league})
    tmk_games = pd.read_sql_query(tmkQuery, con)
    tmk_games = tmk_games.query(f'season == {season} and league_id == {league}').copy()

    # exclud top teams 
    top_teams=[129, 130]
    #top_teams = [93, 94, 95]


    rank_mask = ~((rank_games['home_team_id'].isin(top_teams)) | (rank_games['away_team_id'].isin(top_teams)))
    rank_games = rank_games.loc[rank_mask, :].copy()
    tmk_mask = ~((tmk_games['home_team_id'].isin(top_teams)) | (tmk_games['away_team_id'].isin(top_teams)))
    tmk_games = tmk_games.loc[tmk_mask, :].copy()
    '''
    rank_games = rank_games[~((rank_games['home_team_id'].isin(top_teams)) | (rank_games['away_team_id'].isin(top_teams)))]
    tmk_games = tmk_games[~((rank_games['home_team_id'].isin(top_teams)) | (tmk_games['away_team_id'].isin(top_teams)))]
    '''

    rank_games['result'] = np.sign(rank_games["fulltime_home_goals"] - rank_games["fulltime_away_goals"]).astype(int)


    home_goals_list = rank_games.loc[:,"fulltime_home_goals"]
    away_goals_list = rank_games.loc[:,"fulltime_away_goals"]
    home_teams_list = rank_games.loc[:,'home_team_id']
    away_teams_list = rank_games.loc[:,'away_team_id']
    match_date_list = rank_games.loc[:,'date']
    N = 8*len(home_goals_list)//10

    # dataset used for Massey
    m_ratings, _ = WeightedMasseyEngine.get_ratings(home_goals_list[:N], away_goals_list[:N],home_teams_list[:N], away_teams_list[:N])
    #wm_ratings, home_advantage = WeightedMasseyEngine.get_ratings(home_goals_list[:N], away_goals_list[:N],home_teams_list[:N], away_teams_list[:N],match_date=match_date_list[:N], avg_mv=avg_mv)

    m_dictionary = m_ratings.set_index('team')['rating'].to_dict()
    #wm_dictionary = wm_ratings.set_index('team')['mv_rating'].to_dict()

    rank_games['home_team_massey'] = rank_games['home_team_id'].map(m_dictionary)
    rank_games['away_team_massey'] = rank_games['away_team_id'].map(m_dictionary)
    rank_games['massey_diff'] = rank_games['home_team_massey'] - rank_games['away_team_massey']
    '''
    rank_games['home_team_wtd_massey'] = rank_games['home_team_id'].map(wm_dictionary)
    rank_games['away_team_wtd_massey'] = rank_games['away_team_id'].map(wm_dictionary)
    rank_games['home_team_wtd_massey'] += home_advantage
    rank_games['wtd_massey_diff'] = rank_games['home_team_wtd_massey'] - rank_games['away_team_wtd_massey']
    '''

    # Colley (on the same dataset)
    colley_ratings = WeightedColleyEngine.get_ratings(home_goals_list[:N],away_goals_list[:N],home_teams_list[:N],away_teams_list[:N])

    wtd_colley_ratings = WeightedColleyEngine.get_ratings(home_goals_list[:N],away_goals_list[:N],home_teams_list[:N],away_teams_list[:N], match_date_list[:N])
    
    c_dictionary = colley_ratings.set_index('team')['rating'].to_dict()
    wc_dictionary = wtd_colley_ratings.set_index('team')['rating'].to_dict()

    rank_games['home_team_colley'] = rank_games['home_team_id'].map(c_dictionary)
    rank_games['away_team_colley'] = rank_games['away_team_id'].map(c_dictionary)
    rank_games['colley_diff'] = rank_games['home_team_colley'] - rank_games['away_team_colley']

    rank_games['home_team_wtd_colley'] = rank_games['home_team_id'].map(wc_dictionary)
    rank_games['away_team_wtd_colley'] = rank_games['away_team_id'].map(wc_dictionary)
    rank_games['wtd_colley_diff'] = rank_games['home_team_wtd_colley'] - rank_games['away_team_wtd_colley']
    
    rank_games_80 = rank_games.iloc[:N]
    rank_games_80.reset_index(drop=True, inplace=True)
    rank_games_20 = rank_games.iloc[N:]
    rank_games_20.reset_index(drop=True, inplace=True)



    # dataset used for peeters models
    tmk_games['homeLogMV'] = np.log1p(tmk_games['homeMarketVal'])
    tmk_games['awayLogMV'] = np.log1p(tmk_games['awayMarketVal'])
    tmk_games['homeLogPV'] = np.log1p(tmk_games['homePurchaseVal'])
    tmk_games['awayLogPV'] = np.log1p(tmk_games['awayPurchaseVal'])
    tmk_games = _prepare_data_for_transfermarkt_model(tmk_games)

    tmk_games_80 = tmk_games.iloc[:N]
    tmk_games_80.reset_index(drop=True, inplace=True)
    tmk_games_20 = tmk_games.iloc[N:]
    tmk_games_20.reset_index(drop=True, inplace=True)

    return rank_games_80, rank_games_20, tmk_games_80, tmk_games_20 