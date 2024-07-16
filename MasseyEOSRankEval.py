from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

from getData import fetch_data_for_massey_eos_eval
from weighted_massey_engine import WeightedMasseyEngine


""" 
This is used to compare the end of season ranking prediction by league made with Massey and Weighted Massey, using metrics
    1) MAE (mean absolute error)
    2) Spearman's rank correlation coefficient
    3) Kendall's tau rank correlation coefficient
"""


def ranking_eval(season):
    '''
    Compute the Massey and Weighted Massey ratings based on matches data before predict season, then compare it with true ranking using MAE and Spearman's rank correlation coefficient as metrics

    Input
    --------
        season: year of which the end of season ranking is to predict and compare

    Output
    --------
        m_eval: a dataframe containing 1)MAE 2)Spearman 3)Kandall's tau evaluation result for the end of season ranking predictions by league using Massey method  
        wm_eval: a dataframe containing 1)MAE 2)Spearman 3)Kandall's tau evaluation result for the end of season ranking predictions by league using Weighted Massey method  

    '''
    Games, ranking, marketValues = fetch_data_for_massey_eos_eval()

    EEOSR = ranking.loc[ranking['season']==season, :]
    pre_Matches = Games.loc[Games['season'] < season, :]

    avg_mv = marketValues.loc[(marketValues['season'] == season-1), ['season', 'team_id', 'avg_market_val']] 

    #avg_mv = marketValues.loc[(marketValues['season'] == season) | (marketValues['season'] == season - 1), ['season', 'team_id', 'avg_market_val']]
    avg_mv = marketValues.loc[(marketValues['season'] == season-1), ['season', 'team_id', 'avg_market_val']]

    m_ratings, m_home_advantage = WeightedMasseyEngine.get_ratings(goals_home=pre_Matches['fulltime_home_goals'], 
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
        # remove NA teams from true ranking as well
        compare['ranking'] = compare['ranking'].rank(ascending=True, method='min').astype(int)
        compare = compare.rename(columns={'ranking': 'true_ranking'})

        mae = np.mean(np.abs(compare['massey_ranking'] - compare['true_ranking']))

        spearman_corr, _ = spearmanr(compare['massey_ranking'], compare['true_ranking'])

        tau, _ = kendalltau(compare['massey_ranking'], compare['true_ranking'])

        eval.loc[league_id-1,'MAE'] = mae
        eval.loc[league_id-1,'Spearman'] = spearman_corr
        eval.loc[league_id-1,'Kendall’s tau'] = tau
    return eval

def plot_EOS():
    league_names = ['Premier League', 'Championship', 'League 1', 'League 2']
    league_colors = {
        'Premier League': 'blue',
        'Championship': 'green',
        'League 1': 'red',
        'League 2': 'purple'
    }

    massey_data = pd.DataFrame()
    wm_data = pd.DataFrame()
    for season in range(2011,2024):
        m_eval, wm_eval = ranking_eval(season)
        m_eval['League'] = league_names
        m_eval['Year'] = season
        wm_eval['League'] = league_names
        wm_eval['Year'] = season
        
        massey_data = pd.concat([massey_data, m_eval], axis=0)
        wm_data = pd.concat([wm_data, wm_eval], axis=0)
        
    
    y_min = min(massey_data['Kendall’s tau'].min(), wm_data['Kendall’s tau'].min()) - 0.05
    y_max = max(wm_data['Kendall’s tau'].max(), massey_data['Kendall’s tau'].max()) + 0.05

    # Plot the line graph
    plt.figure(figsize=(10, 6))
    for league in league_names:
        league_data = massey_data[massey_data['League'] == league]
        plt.plot(league_data['Year'], league_data['Kendall’s tau'], marker='o', label=league, color=league_colors[league])
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Kendall’s tau')
    plt.title('Kendall tau rank correlation of Massey Ranking from 2011-2023')
    plt.legend()
    #plt.ylim(y_min, y_max)
    plt.ylim(-0.35, 0.75)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("massey_EOS_line.png")

    plt.figure(figsize=(10, 6))
    for league in league_names:
        league_data = wm_data[wm_data['League'] == league]
        plt.plot(league_data['Year'], league_data['Kendall’s tau'], marker='o', label=league, color=league_colors[league])
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Kendall’s tau')
    plt.title('Kendall tau rank correlation of Weighted Massey Ranking from 2011-2023')
    plt.legend()
    #plt.ylim(y_min, y_max)
    plt.ylim(-0.35, 0.75)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("weighted_massey_EOS_line.png")


    # Plot bar graph
    # Calculate the average accuracy for each league
    custom_order = ['Premier League', 'Championship', 'League 1', 'League 2']
    m_average_tau= massey_data.groupby('League')['Kendall’s tau'].mean().reset_index()
    m_average_tau['League'] = pd.Categorical(m_average_tau['League'], categories=custom_order, ordered=True)
    m_average_tau = m_average_tau.sort_values(by='League')

    wm_average_tau = wm_data.groupby('League')['Kendall’s tau'].mean().reset_index()
    wm_average_tau['League'] = pd.Categorical(wm_average_tau['League'], categories=custom_order, ordered=True)
    wm_average_tau = wm_average_tau.sort_values(by='League')

    y_max = max(m_average_tau['Kendall’s tau'].max(), wm_average_tau['Kendall’s tau'].max()) * 1.05

    plt.figure(figsize=(10, 6))
    plt.bar(m_average_tau['League'], m_average_tau['Kendall’s tau'], color=[league_colors[league] for league in m_average_tau['League']])
    plt.xlabel('League')
    plt.ylabel('Kendall\'s tau')
    plt.title('Average Kendall\'s tau of Massey Model from 2011-2023')
    plt.ylim(0, y_max)
    plt.grid(True, axis='y')
    plt.savefig('massey_EOS_bar.png')
    #plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(wm_average_tau['League'], wm_average_tau['Kendall’s tau'], color=[league_colors[league] for league in wm_average_tau['League']])
    plt.xlabel('League')
    plt.ylabel('Kendall\'s tau')
    plt.title('Average Kendall\'s tau of Weighted Massey Model from 2011-2023')
    plt.ylim(0, y_max)
    plt.grid(True, axis='y')
    plt.savefig('weighted_massey_EOS_bar.png')

    print(m_average_tau)
    print(wm_average_tau)





if __name__ == "__main__":
    start = perf_counter()
    plot_EOS()
    print(perf_counter()-start)