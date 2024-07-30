from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

from baseModel import ALL_LEAGUES, COUNTRY_TO_ADJECTIVES, COUNTRY_TO_LEAGUES
from getData import fetch_data_for_massey_eos_eval
from weighted_colley_engine import WeightedColleyEngine


""" 
This is used to compare the end of season ranking prediction by league made with Massey and Weighted Massey, using metrics
    1) MAE (mean absolute error)
    2) Spearman's rank correlation coefficient
    3) Kendall's tau rank correlation coefficient
"""


def ranking_eval(season):
    '''
    Compute the Colley and Weighted Massey Colley based on matches data before predict season, then compare it with true ranking using MAE and Spearman's rank correlation coefficient as metrics

    Input
    --------
        season: year of which the end of season ranking is to predict and compare

    Output
    --------
        c_eval: a dataframe containing 1)MAE 2)Spearman 3)Kandall's tau evaluation result for the end of season ranking predictions by league using Colley method  
        wc_eval: a dataframe containing 1)MAE 2)Spearman 3)Kandall's tau evaluation result for the end of season ranking predictions by league using Weighted Colley method  

    '''
    Games, ranking, marketValues = fetch_data_for_massey_eos_eval()

    EEOSR = ranking.loc[ranking['season']==season, :]
    pre_Matches = Games.loc[Games['season'] < season, :]

    home_goals_list = pre_Matches.loc[:,"fulltime_home_goals"]
    away_goals_list = pre_Matches.loc[:,"fulltime_away_goals"]
    home_teams_list = pre_Matches.loc[:,'home_team_id']
    away_teams_list = pre_Matches.loc[:,'away_team_id']
    date_list = pre_Matches.loc[:,'date']

    c_ratings = WeightedColleyEngine.get_ratings(home_goals_list,away_goals_list,home_teams_list,away_teams_list)

    c_eval = evaluation(c_ratings, EEOSR)

    wc_ratings = WeightedColleyEngine.get_ratings(home_goals_list,away_goals_list,home_teams_list,away_teams_list, date_list)

    wc_eval = evaluation(wc_ratings, EEOSR)
    
    return c_eval, wc_eval


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


def plot_EOS(country):
    country = country.casefold() if isinstance(country,str) else country
    league_ids = COUNTRY_TO_LEAGUES[country]
    league_names = [ALL_LEAGUES[_id-1] for _id in league_ids]

    c_data = []
    wc_data = []
    for season in range(2011,2024):
        c_eval, wc_eval = ranking_eval(season)
        c_eval['League'] = ALL_LEAGUES
        c_eval['Year'] = season
        wc_eval['League'] = ALL_LEAGUES
        wc_eval['Year'] = season
        c_data.append(c_eval)
        wc_data.append(wc_eval)

    c_data = pd.concat(c_data, axis=0)
    wc_data = pd.concat(wc_data, axis=0)
    
    y_min = min(c_data['Kendall’s tau'].min(), wc_data['Kendall’s tau'].min()) - 0.05
    y_max = max(wc_data['Kendall’s tau'].max(), c_data['Kendall’s tau'].max()) + 0.05

    # Plot the line graph
    plt.figure(figsize=(10, 6))
    for league in league_names:
        league_data = c_data[c_data['League'] == league]
        #plt.plot(league_data['Year'], league_data['Kendall’s tau'], marker='o', label=league, color=league_colors[league])
        plt.plot(league_data['Year'], league_data['Kendall’s tau'],label=league)
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Kendall’s tau')
    plt.title(f'Kendall tau rank correlation of {COUNTRY_TO_ADJECTIVES[country]} leagues Colley Ranking from 2011-2023')
    plt.legend()
    #plt.ylim(y_min, y_max)
    plt.ylim(-0.35, 0.75)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"colley_{country}_EOS_line.png")
    plt.figure(figsize=(10, 6))
    for league in league_names:
        league_data = wc_data[wc_data['League'] == league]
        #plt.plot(league_data['Year'], league_data['Kendall’s tau'], marker='o', label=league, color=league_colors[league])
        plt.plot(league_data['Year'], league_data['Kendall’s tau'],label=league)
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Kendall’s tau')
    plt.title(f'Kendall tau rank correlation of {COUNTRY_TO_ADJECTIVES[country]} leagues Weighted Colley Ranking from 2011-2023')
    plt.legend()
    #plt.ylim(y_min, y_max)
    plt.ylim(-0.35, 0.75)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"weighted_colley_{country}_EOS_line.png")
    
    
    # Plot bar graph
    # Calculate the average accuracy for each league
    custom_order = league_names
    c_average_tau = c_data.loc[[l in league_names for l in c_data['League']],:].groupby('League')['Kendall’s tau'].mean().reset_index()
    c_average_tau['League'] = pd.Categorical(c_average_tau['League'], categories=custom_order, ordered=True)
    c_average_tau = c_average_tau.sort_values(by='League')

    wc_average_tau = wc_data.loc[[l in league_names for l in wc_data['League']],:].groupby('League')['Kendall’s tau'].mean().reset_index()
    wc_average_tau['League'] = pd.Categorical(wc_average_tau['League'], categories=custom_order, ordered=True)
    wc_average_tau = wc_average_tau.sort_values(by='League')
    
    y_max = max(c_average_tau['Kendall’s tau'].max(), wc_average_tau['Kendall’s tau'].max()) * 1.05

    plt.figure(figsize=(10, 6))
    plt.bar(c_average_tau['League'], c_average_tau['Kendall’s tau'])
    plt.xlabel('League')
    plt.ylabel('Kendall\'s tau')
    plt.title(f'Average Kendall\'s tau of {COUNTRY_TO_ADJECTIVES[country]} leagues Colley Model from 2011-2023')
    plt.ylim(0, y_max)
    plt.grid(True, axis='y')
    plt.savefig(f'colley_{country}_EOS_bar.png')

    plt.figure(figsize=(10, 6))
    plt.bar(wc_average_tau['League'], wc_average_tau['Kendall’s tau'])
    plt.xlabel('League')
    plt.ylabel('Kendall\'s tau')
    plt.title(f'Average Kendall\'s tau of {COUNTRY_TO_ADJECTIVES[country]} leagues Weighted Colley Model from 2011-2023')
    plt.ylim(0, y_max)
    plt.grid(True, axis='y')
    plt.savefig(f'weighted_colley_{country}_EOS_bar.png')

    return c_average_tau, wc_average_tau


if __name__ == "__main__":
    start = perf_counter()
    plot_EOS(country="england")
    print(perf_counter()-start)
