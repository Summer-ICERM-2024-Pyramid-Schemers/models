import sqlite3
import numpy as np
import pandas as pd

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
    con = sqlite3.connect("english_football_data.sqlite")

    gamesQuery = f"""
    SELECT home_team_id, away_team_id, fulltime_home_goals, fulltime_away_goals, fulltime_result, market_average_home_win_odds AS home_odds, market_average_draw_odds AS draw_odds, market_average_away_win_odds AS away_odds 
    FROM Matches
    WHERE season = {season}
    AND league_id = {league}
    """

    mvQuery = f"""
    SELECT team_id, avg_market_val, total_market_val
    FROM TeamMarketvalues
    WHERE season = {season}
    AND league_id = {league}
    """

    Games = pd.read_sql_query(gamesQuery, con)
    marketValues = pd.read_sql_query(mvQuery, con)

    Games = Games.merge(
        marketValues.rename(columns={'total_market_val': 'home_team_total_value'}),
        left_on='home_team_id',
        right_on='team_id',
        how='left'
    ).drop('team_id', axis=1)

    Games = Games.merge(
        marketValues.rename(columns={'total_market_val': 'away_team_total_value'}),
        left_on='away_team_id',
        right_on='team_id',
        how='left'
    ).drop('team_id', axis=1)

    Games['homeLogMV'] = np.log(Games['home_team_total_value'])
    Games['awayLogMV'] = np.log(Games['away_team_total_value'])


    intTable = pd.DataFrame(columns=['result','iHome','jHome','iGoals','jGoals','iValue', 'jValue', 'iOdds', 'jOdds', 'drawOdds'])

    for i in range(len(Games)):
        match = Games.iloc[i]
        oddsProbs = normOdds(match['home_odds'],
                                match['draw_odds'],
                                match['away_odds'])
        teami = np.random.choice([0,1])
        if teami == 0:
            # Home team is team i, away team is team j
            iOdds = oddsProbs[0]
            drawOdds = oddsProbs[1]
            jOdds = oddsProbs[2]
            iHome = 1
            jHome = 0
            iGoals = match['fulltime_home_goals']
            jGoals = match['fulltime_away_goals']
            iValue = match['homeLogMV']
            jValue = match['awayLogMV']
            if match['fulltime_result'] == 'H':
                result = 1
            elif match['fulltime_result'] == 'A':
                result = -1
            elif match['fulltime_result'] == 'D':
                result = 0
        elif teami == 1:
            # Away team is team i, home team is team j
            iOdds = oddsProbs[2]
            drawOdds = oddsProbs[1]
            jOdds = oddsProbs[0]
            jHome = 1
            iHome = 0
            jGoals = match['fulltime_home_goals']
            iGoals = match['fulltime_away_goals']
            jValue = match['homeLogMV']
            iValue = match['awayLogMV']
            if match['fulltime_result'] == 'H':
                result = -1
            elif match['fulltime_result'] == 'A':
                result = 1
            elif match['fulltime_result'] == 'D':
                result = 0

        intTable = pd.concat([pd.DataFrame([[result, iHome, jHome, iGoals, jGoals, iValue, jValue, iOdds, jOdds, drawOdds]], 
                                            columns=intTable.columns), intTable], 
                                            ignore_index=True)
        

    finalTable = pd.DataFrame(columns=['result','goaldiff','Home','Value', 'iOdds', 'drawOdds', 'jOdds'])
    for i in range(len(intTable)):
        match = intTable.iloc[i]
        result = match['result']
        goaldiff = match['iGoals'] - match['jGoals']
        home = match['iHome'] - match['jHome']
        value = match['iValue'] - match['jValue']
        finalTable = pd.concat([pd.DataFrame([[result, goaldiff, home, value, match['iOdds'], match['drawOdds'], match['jOdds']]], 
                                            columns=finalTable.columns), finalTable], 
                                            ignore_index=True)

    finalTable = finalTable.apply(pd.to_numeric)

    return 

def normOdds(iOdds, drawOdds, jOdds):
    juice = 1/iOdds + 1/drawOdds + 1/jOdds
    iProb = 1/(iOdds*juice)
    drawProb = 1/(drawOdds*juice)
    jProb = 1/(jOdds*juice)
    
    return [iProb, drawProb, jProb]