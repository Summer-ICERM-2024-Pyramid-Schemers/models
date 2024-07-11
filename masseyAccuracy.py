from weightedMassey import WeightedMassey
from massey import Massey
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

def get_massey(season,league):
        
    con = sqlite3.connect('/Users/yutong.bu/Desktop/ICERM/sports analytics/data/english_football_data.sqlite')

    gamesQuery = f"""
    SELECT home_team_id, away_team_id, fulltime_home_goals, fulltime_away_goals, fulltime_result
    FROM Matches
    WHERE season = {season}
    AND league_id = {league}
    """
    
    Games = pd.read_sql_query(gamesQuery, con)


    home_goals_list = Games["fulltime_home_goals"].tolist()
    away_goals_list = Games["fulltime_away_goals"].tolist()
    home_teams_list = Games['home_team_id'].tolist()
    away_teams_list = Games['away_team_id'].tolist()
    if(league==1):
        massey = Massey(home_goals_list[:305],away_goals_list[:305],home_teams_list[:305], away_teams_list[:305])
    elif((season==2019)and league != 2):
        if(league==3):
           massey = Massey(home_goals_list[:321],away_goals_list[:321],home_teams_list[:321], away_teams_list[:321])
        if(league==4):
            massey = Massey(home_goals_list[:353],away_goals_list[:353],home_teams_list[:353], away_teams_list[:353])
    else:
        massey = Massey(home_goals_list[:443],away_goals_list[:443],home_teams_list[:443], away_teams_list[:443])
    massey_ratings = massey.get_ratings()

    dictionary = massey_ratings.set_index('team')['rating'].to_dict()

    Games['home_team_massey'] = Games['home_team_id'].map(dictionary)
    Games['away_team_massey'] = Games['away_team_id'].map(dictionary)
    
    if(league==1):
        Games = Games.iloc[305:]
    elif(season==2019):
        if(league==3):
            Games = Games.iloc[321:]
        if(league==4):
            Games = Games.iloc[353:]
    else:
        Games = Games.iloc[443:]
        
    Games.reset_index(drop=True, inplace=True)

    return Games   



def get_wtd_massey(season,league):
        
    con = sqlite3.connect('/Users/yutong.bu/Desktop/ICERM/sports analytics/data/english_football_data.sqlite')

    gamesQuery = f"""
    SELECT date, home_team_id, away_team_id, fulltime_home_goals, fulltime_away_goals, fulltime_result
    FROM Matches
    WHERE season = {season}
    AND league_id = {league}
    """

    mvQuery = f"""
    SELECT season, league_id, team_id, avg_market_val
    FROM TeamMarketvalues
    WHERE season = {season}
    AND league_id = {league}
    """
    
    Games = pd.read_sql_query(gamesQuery, con)
    avg_mv = pd.read_sql_query(mvQuery, con)


    home_goals_list = Games["fulltime_home_goals"].tolist()
    away_goals_list = Games["fulltime_away_goals"].tolist()
    home_teams_list = Games['home_team_id'].tolist()
    away_teams_list = Games['away_team_id'].tolist()
    match_date_list = Games['date'].tolist()
    if(league==1):
        wtd_massey = WeightedMassey(home_goals_list[:305],away_goals_list[:305],home_teams_list[:305], away_teams_list[:305], match_date=match_date_list[:305],avg_mv=avg_mv)
    elif((season==2019)and league != 2):
        if(league==3):
           wtd_massey = WeightedMassey(home_goals_list[:321],away_goals_list[:321],home_teams_list[:321], away_teams_list[:321], match_date=match_date_list[:321],avg_mv=avg_mv)
        if(league==4):
            wtd_massey = WeightedMassey(home_goals_list[:353],away_goals_list[:353],home_teams_list[:353], away_teams_list[:353], match_date=match_date_list[:353],avg_mv=avg_mv)
    else:
        wtd_massey = WeightedMassey(home_goals_list[:443],away_goals_list[:443],home_teams_list[:443], away_teams_list[:443], match_date=match_date_list[:443],avg_mv=avg_mv)
    wm_ratings, home_advantage = wtd_massey.get_ratings()
    # if uses massey rating in combination with market value
    wm_ratings = wm_ratings.drop(columns=['rating'])
    wm_ratings = wm_ratings.rename(columns={'mv_rating':'rating'})

    dictionary = wm_ratings.set_index('team')['rating'].to_dict()

    Games['home_team_wtd_massey'] = Games['home_team_id'].map(dictionary)
    Games['away_team_wtd_massey'] = Games['away_team_id'].map(dictionary)

    Games['home_team_wtd_massey'] = Games['home_team_wtd_massey'] + home_advantage
    
    if(league==1):
        Games = Games.iloc[305:]
    elif(season==2019):
        if(league==3):
            Games = Games.iloc[321:]
        if(league==4):
            Games = Games.iloc[353:]
    else:
        Games = Games.iloc[443:]
        
    Games.reset_index(drop=True, inplace=True)

    return Games, home_advantage

def get_accuracy(season,league):
    massey = get_massey(season,league)
    wtd_massey, home_advantage = get_wtd_massey(season, league)
    massey_counter=0
    total_counter=0
    for index, row in massey.iterrows():
        if (row['home_team_massey'] > row['away_team_massey']) and (row['fulltime_result'] == 'H') or ((row['home_team_massey'] < row['away_team_massey']) and (row['fulltime_result'] == 'A')):
            massey_counter +=1
            total_counter +=1
        elif(row['fulltime_result'] != 'D'):
            total_counter += 1

    wm_counter = 0
    for index, row in wtd_massey.iterrows():
        if (row['home_team_wtd_massey'] > row['away_team_wtd_massey']) and (row['fulltime_result'] == 'H') or ((row['home_team_wtd_massey'] < row['away_team_wtd_massey']) and (row['fulltime_result'] == 'A')):
            wm_counter +=1

    
    massey_accuracy = massey_counter / total_counter
    wm_accuracy = wm_counter / total_counter
    return massey_accuracy, wm_accuracy


def plot_accuracy():
    years = list(range(2010, 2024))
    leagues = list(range(1,5))

    league_names = {
        1: 'Premier League',
        2: 'Championship',
        3: 'League 1',
        4: 'League 2'
    }

    league_colors = {
        'Premier League': 'blue',
        'Championship': 'green',
        'League 1': 'red',
        'League 2': 'purple'
    }

    # Collect accuracy data
    massey_data = []
    wm_data = []
    for year in years:
        for league in leagues:
            massey_accuracy, wm_accuracy = get_accuracy(year, league)
            massey_data.append({'Year': year, 'League': league_names[league], 'Accuracy': massey_accuracy})
            wm_data.append({'Year': year, 'League': league_names[league], 'Accuracy': wm_accuracy})

    # Create DataFrame
    massey_df = pd.DataFrame(massey_data)
    wm_df = pd.DataFrame(wm_data)


    # Calculate the average accuracy for each league
    average_massey_accuracies = massey_df.groupby('League')['Accuracy'].mean().reset_index()
    average_wm_accuracies = wm_df.groupby('League')['Accuracy'].mean().reset_index()


    # Plot the line graph
    plt.figure(figsize=(10, 6))
    for league in league_names.values():
        league_data = massey_df[massey_df['League'] == league]
        plt.plot(league_data['Year'], league_data['Accuracy'], marker='o', label=league, color=league_colors[league])
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of Massey Model from 2010-2023')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    for league in league_names.values():
        league_data = wm_df[wm_df['League'] == league]
        plt.plot(league_data['Year'], league_data['Accuracy'], marker='o', label=league, color=league_colors[league])
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of Weighted Massey Model from 2010-2023')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(average_massey_accuracies['League'], average_massey_accuracies['Accuracy'], color=[league_colors[league] for league in average_massey_accuracies['League']])
    # Set the y-axis range
    plt.ylim(0.5, 0.7)
    # Add labels and title
    plt.xlabel('League')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy of Massey Model from 2010-2023')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(average_wm_accuracies['League'], average_wm_accuracies['Accuracy'], color=[league_colors[league] for league in average_wm_accuracies['League']])
    # Set the y-axis range
    plt.ylim(0.5, 0.7)
    # Add labels and title
    plt.xlabel('League')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy of Weighted Massey Model from 2010-2023')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.show()
