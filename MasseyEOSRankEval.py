from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from getData import fetch_data_for_massey_eos_eval
from massey_engine import MasseyEngine
from weighted_massey_engine import WeightedMasseyEngine


""" 
This is used to compare the end of season ranking prediction by league made with Massey and Weighted Massey, using metrics
    1) MAE (mean absolute error)
    2) Spearman's rank correlation coefficient
"""


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
    Games, ranking, marketValues = fetch_data_for_massey_eos_eval()

    EEOSR = ranking.loc[ranking['season']==season, :]
    pre_Matches = Games.loc[Games['season'] < season, :]

    avg_mv = marketValues.loc[(marketValues['season'] == season-1), ['season', 'team_id', 'avg_market_val']] # TBD: season or season-1

    #avg_mv = marketValues.loc[(marketValues['season'] == season) | (marketValues['season'] == season - 1), ['season', 'team_id', 'avg_market_val']]
    avg_mv = marketValues.loc[(marketValues['season'] == season-1), ['season', 'team_id', 'avg_market_val']] # TBD: season or season-1

    #avg_mv = marketValues.loc[(marketValues['season'] == season) | (marketValues['season'] == season - 1), ['season', 'team_id', 'avg_market_val']]

    m_ratings = MasseyEngine.get_ratings(goals_home=pre_Matches['fulltime_home_goals'], 
                    goals_away=pre_Matches['fulltime_away_goals'], 
                    teams_home=pre_Matches['home_team_id'],
                    teams_away=pre_Matches['away_team_id'])

    m_eval = evaluation(m_ratings, EEOSR)

    wm_ratings, wm_home_advantage = WeightedMasseyEngine.get_ratings(goals_home=pre_Matches['fulltime_home_goals'], 
                                goals_away=pre_Matches['fulltime_away_goals'], 
                                teams_home=pre_Matches['home_team_id'],
                                teams_away=pre_Matches['away_team_id'],
                                match_date=pre_Matches['date'],
                                avg_mv=avg_mv)

    # if uses massey rating in combination with market value
    wm_ratings = wm_ratings.drop(columns=['rating'])
    wm_ratings = wm_ratings.rename(columns={'mv_rating':'rating'})

    wm_eval = evaluation(wm_ratings, EEOSR)
    
    return m_eval, wm_eval

def evaluation(ratings, EEOSR):
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


if __name__ == "__main__":
    start = perf_counter()
    print(*ranking_eval(2013),sep='\n')
    print(perf_counter()-start)
