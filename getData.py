import sqlite3
import numpy as np
import os
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

    data = getData()
    Games = data.query(f'season == {season} and league_id == {league}')


    intTable = pd.DataFrame(columns=['result','iHome','jHome','iGoals','jGoals','iValue', 'jValue', 'iPurchase', 'jPurchase', 'iOdds', 'jOdds', 'drawOdds'])

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
            iPurchase = match['homeLogPV']
            jPurchase = match['awayLogPV']
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
            jPurchase = match['homeLogPV']
            iPurchase = match['awayLogPV']
            if match['fulltime_result'] == 'H':
                result = -1
            elif match['fulltime_result'] == 'A':
                result = 1
            elif match['fulltime_result'] == 'D':
                result = 0

        intTable = pd.concat([pd.DataFrame([[result, iHome, jHome, iGoals, jGoals, iValue, jValue, iPurchase, jPurchase, iOdds, jOdds, drawOdds]], 
                                            columns=intTable.columns), intTable], 
                                            ignore_index=True)
        
    finalTable = pd.DataFrame(columns=['result','goaldiff','Home','Value', 'PurchaseValue', 'iOdds', 'drawOdds', 'jOdds'])

    for i in range(len(intTable)):
        match = intTable.iloc[i]
        result = match['result']
        goaldiff = match['iGoals'] - match['jGoals']
        home = match['iHome'] - match['jHome']
        value = match['iValue'] - match['jValue']
        purchaseValue = match['iPurchase'] - match['jPurchase']
        finalTable.append([pd.DataFrame(result, goaldiff, home, value, purchaseValue, match['iOdds'], match['drawOdds'], match['jOdds'])], 
                                            columns=finalTable.columns, ignore_index = True)

       #finalTable = pd.concat([pd.DataFrame([[result, goaldiff, home, value, purchaseValue, match['iOdds'], match['drawOdds'], match['jOdds']]], 
       #                                     columns=finalTable.columns), finalTable], 
        #                                    ignore_index=True)'''

    finalTable = finalTable.apply(pd.to_numeric)

    return finalTable


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

    intTable = pd.DataFrame(columns=['result','iHome','jHome','iGoals','jGoals','iValue', 'jValue', 'iPurchase', 'jPurchase', 'iOdds', 'jOdds', 'drawOdds'])

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
            iPurchase = match['homeLogPV']
            jPurchase = match['awayLogPV']
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
            jPurchase = match['homeLogPV']
            iPurchase = match['awayLogPV']
            if match['fulltime_result'] == 'H':
                result = -1
            elif match['fulltime_result'] == 'A':
                result = 1
            elif match['fulltime_result'] == 'D':
                result = 0

        intTable = pd.concat([pd.DataFrame([[result, iHome, jHome, iGoals, jGoals, iValue, jValue, iPurchase, jPurchase, iOdds, jOdds, drawOdds]], 
                                            columns=intTable.columns), intTable], 
                                            ignore_index=True)
        
    finalTable = pd.DataFrame(columns=['result','goaldiff','Home','Value', 'PurchaseValue', 'iOdds', 'drawOdds', 'jOdds'])

    for i in range(len(intTable)):
        match = intTable.iloc[i]
        result = match['result']
        goaldiff = match['iGoals'] - match['jGoals']
        home = match['iHome'] - match['jHome']
        value = match['iValue'] - match['jValue']
        purchaseValue = match['iPurchase'] - match['jPurchase']
        finalTable = pd.concat([pd.DataFrame([[result, goaldiff, home, value, purchaseValue, match['iOdds'], match['drawOdds'], match['jOdds']]], 
                                            columns=finalTable.columns), finalTable], 
                                            ignore_index=True)

    finalTable = finalTable.apply(pd.to_numeric)

    return finalTable


def getData():
    """
    Returns raw data for all matches of all leagues
    """

    if os.path.exists("rawData.pkl"):
        print("File read successfully.")
        return pd.read_pickle("rawData.pkl")
    else:
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

        pd.to_pickle(Games, "rawData.pkl")
        print("File pickled")

        return Games


def normOdds(iOdds, drawOdds, jOdds):
    juice = 1/iOdds + 1/drawOdds + 1/jOdds
    iProb = 1/(iOdds*juice)
    drawProb = 1/(drawOdds*juice)
    jProb = 1/(jOdds*juice)
    
    return [iProb, drawProb, jProb]