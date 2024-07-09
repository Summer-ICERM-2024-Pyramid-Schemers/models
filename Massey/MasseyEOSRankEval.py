from weightedMassey import WeightedMassey
from massey import Massey
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import sqlite3


""" 
This is used to compare the end of season ranking prediction by league made with Massey and Weighted Massey, using metrics
    1) MAE (mean absolute error)
    2) Spearman's rank correlation coefficient

TO USE THIS:

from MasseyEOSRankEval import ranking_eval

ranking_eval(2013)
"""


def prepareData():
    # substitute with directory of your dataset
    con = sqlite3.connect('/Users/yutong.bu/Desktop/ICERM/sports analytics/data/english_football_data.sqlite')

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

def ranking_eval(season):
    '''
    Compute the Massey and Weighted Massey ratings based on matches data before predict season, then compare it with true ranking using MAE and Spearman's rank correlation coefficient as metrics

    Input
    --------
        season: year of which the end of season ranking is to predict and compare

    Output
    --------
        m_eval: a dataframe containing 1)MAE 2)Spearman evaluation result for the end of season ranking predictions by league using Massey method  
        wm_eval: a dataframe containing 1)MAE 2)Spearman evaluation result for the end of season ranking predictions by league using Weighted Massey method  

    '''

    Games, ranking, marketValues = prepareData()

    EEOSR = ranking.loc[ranking['season']==season, ]
    pre_Matches = Games.loc[Games['season'] < season,]

    avg_mv = marketValues.loc[(marketValues['season'] == season-1), ['season', 'team_id', 'avg_market_val']] # TBD: season or season-1

    #avg_mv = marketValues.loc[(marketValues['season'] == season) | (marketValues['season'] == season - 1), ['season', 'team_id', 'avg_market_val']]

    massey = Massey(goals_home=pre_Matches['fulltime_home_goals'], 
                    goals_away=pre_Matches['fulltime_away_goals'], 
                    teams_home=pre_Matches['home_team_id'],
                    teams_away=pre_Matches['away_team_id'])
    m_ratings = massey.get_ratings()

    m_eval = evaluation(m_ratings, EEOSR)

    wtd_massey = WeightedMassey(goals_home=pre_Matches['fulltime_home_goals'], 
                                goals_away=pre_Matches['fulltime_away_goals'], 
                                teams_home=pre_Matches['home_team_id'],
                                teams_away=pre_Matches['away_team_id'],
                                match_date=pre_Matches['date'],
                                avg_mv=avg_mv)
    wm_ratings, wm_home_advantage = wtd_massey.get_ratings()

    # if uses massey rating in combination with market value
    wm_ratings = wm_ratings.drop(columns=['rating'])
    wm_ratings = wm_ratings.rename(columns={'mv_rating':'rating'})
    

    wm_eval = evaluation(wm_ratings, EEOSR)
    

    
    return m_eval, wm_eval



def evaluation (ratings, EEOSR):
    '''
    Input
    ---------
    ranings: 
        Massey ratings (calculated based on data before target season)
    EEOSR:
        end of season ranking for target (predicting) season

    Output
    ---------
    eval:
        a dataframe shows evaluation metrics (MAE and Spearman) for each league
    '''
    eval = pd.DataFrame()

    # evaluate the rankings by league
    for league_id in np.unique(EEOSR['league_id']):

        league_EEOSR = EEOSR.loc[EEOSR['league_id']==league_id, ]
        teams = np.unique(league_EEOSR['team_id'])
        league_massey = pd.DataFrame(teams)
        league_massey.columns = ['team_id']

        # get massey rantings for all teams in this league
        rank_mapping = dict(zip(ratings['team'], ratings['rating']))
        league_massey['rating'] = league_massey['team_id'].map(rank_mapping)
        # Remove rows where 'rating' is NaN (i.e., no match)
        league_massey = league_massey.dropna(subset=['rating'])
        
        league_massey['massey_ranking'] = league_massey['rating'].rank(ascending=False, method='min').astype(int)

        compare = league_massey.merge(league_EEOSR, on='team_id')
        compare = compare.rename(columns={'ranking': 'true_ranking'})

        mae = np.mean(np.abs(compare['massey_ranking'] - compare['true_ranking']))

        spearman_corr, _ = spearmanr(compare['massey_ranking'], compare['true_ranking'])

        eval.loc[league_id-1,'MAE'] = mae
        eval.loc[league_id-1,'Spearman'] = spearman_corr
    return eval