from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from getData import fetch_data_for_massey_accuracy


def get_accuracy(season,league):
    Games = fetch_data_for_massey_accuracy(season,league)
    non_draws = Games.loc[Games["result"].to_numpy() != 0,:]
    
    massey_counter = np.count_nonzero(np.sign(non_draws['home_team_massey']-non_draws['away_team_massey'])==non_draws["result"])
    wm_counter = np.count_nonzero(np.sign(non_draws['home_team_wtd_massey']-non_draws['away_team_wtd_massey'])==non_draws["result"])
    total_counter = len(non_draws)

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
    plt.ylim(0.5, average_wm_accuracies['Accuracy'].max()*1.05)
    # Add labels and title
    plt.xlabel('League')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy of Weighted Massey Model from 2010-2023')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.show()

if __name__ == "__main__":
    start = perf_counter()
    plot_accuracy()
    print(perf_counter()-start)
