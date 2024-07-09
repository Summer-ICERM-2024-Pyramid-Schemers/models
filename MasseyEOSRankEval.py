from weightedMassey import WeightedMassey
from massey import Massey
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ranking_eval (eval_year, match_data, supp_data):
    '''
    Compute the Massey and Weighted Massey ratings based on matches data before `eval_year`,
    then compare it with true ranking using MAE and Spearman's rank correlation coefficient as metrics
    '''
    EEOSR = supp_data.loc[supp_data['season']==eval_year,['league_id', 'team_id', 'ranking']]
    pre_Matches = match_data.loc[match_data['season'] < eval_year,:]

    avg_mv = supp_data.loc[(supp_data['season'] == eval_year),['team_id', 'avg_market_val']] # TBD: eval_year or eval_year-1

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