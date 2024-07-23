import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from getData import fetch_data_for_colley_accuracy


def get_accuracy(season,league):
    games_table = fetch_data_for_colley_accuracy(season,league)
    non_draws = games_table.loc[games_table["result"].to_numpy() != 0,:]
    
    colley_counter = np.count_nonzero(np.sign(non_draws['home_team_colley']-non_draws['away_team_colley'])==non_draws["result"])
    wc_counter = np.count_nonzero(np.sign(non_draws['home_team_wtd_colley']-non_draws['away_team_wtd_colley'])==non_draws["result"])
    total_counter = len(non_draws)

    colley_accuracy = colley_counter / total_counter
    wc_accuracy = wc_counter / total_counter
    return colley_accuracy, wc_accuracy


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

def plot_accuracy():
    # Collect accuracy data
    colley_data = []
    wc_data = []
    for year in years:
        for league in leagues:
            colley_accuracy, wc_accuracy = get_accuracy(year, league)
            colley_data.append({'Year': year, 'League': league_names[league], 'Accuracy': colley_accuracy})
            wc_data.append({'Year': year, 'League': league_names[league], 'Accuracy': wc_accuracy})

    # Create DataFrame
    colley_df = pd.DataFrame(colley_data)
    wc_df = pd.DataFrame(wc_data)

    # Calculate the average accuracy for each league
    average_colley_accuracies = colley_df.groupby('League')['Accuracy'].mean().reset_index()
    average_wc_accuracies = wc_df.groupby('League')['Accuracy'].mean().reset_index()
    # Sort by league
    custom_order = ['Premier League', 'Championship', 'League 1', 'League 2']
    average_colley_accuracies['League'] = pd.Categorical(average_colley_accuracies['League'], categories=custom_order, ordered=True)
    average_colley_accuracies = average_colley_accuracies.sort_values(by='League')
    average_wc_accuracies['League'] = pd.Categorical(average_wc_accuracies['League'], categories=custom_order, ordered=True)
    average_wc_accuracies = average_wc_accuracies.sort_values(by='League')

    # Plot the line graph
    plt.figure(figsize=(10, 6))
    for league in league_names.values():
        league_data = colley_df[colley_df['League'] == league]
        plt.plot(league_data['Year'], league_data['Accuracy'], marker='o', label=league, color=league_colors[league])
    plt.xlabel('Year')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of Colley Model from 2010-2023')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.43, 0.82)
    plt.tight_layout()
    plt.savefig('colley_accuracy_line.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    for league in league_names.values():
        league_data = wc_df[wc_df['League'] == league]
        plt.plot(league_data['Year'], league_data['Accuracy'], marker='o', label=league, color=league_colors[league])
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of Weighted Colley Model from 2010-2023')
    plt.legend()
    plt.ylim(0.43, 0.82)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("weighted_colley_accuracy_line.png")
    plt.show()

    # Plot the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(average_colley_accuracies['League'], average_colley_accuracies['Accuracy'], color=[league_colors[league] for league in average_colley_accuracies['League']])
    plt.ylim(0.5, 0.73)
    plt.xlabel('League')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy of Colley Model from 2010-2023')
    #plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.savefig("colley_accuracy_bar.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(average_wc_accuracies['League'], average_wc_accuracies['Accuracy'], color=[league_colors[league] for league in average_wc_accuracies['League']])
    plt.ylim(0.5, 0.73)
    plt.xlabel('League')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy of Colley Model from 2010-2023')
    #plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.savefig("weighted_colley_accuracy_bar.png")
    plt.show()

    print(average_colley_accuracies)
    print(average_wc_accuracies)


if __name__ == "__main__":
    plot_accuracy()