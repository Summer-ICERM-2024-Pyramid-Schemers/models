import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from getData import fetch_data_for_colley_accuracy


def get_accuracy(season,league):
    games_table = fetch_data_for_colley_accuracy(season,league)
    non_draws = games_table.loc[games_table["result"].to_numpy() != 0,:]
    true_counter = np.sum(np.sign(non_draws['home_team_colley']-non_draws['away_team_colley'])==non_draws["result"])
    return true_counter / len(non_draws)


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

if __name__ == "__main__":
    # Collect accuracy data
    data = []
    for year in years:
        for league in leagues:
            accuracy = get_accuracy(year, league)
            data.append({'Year': year, 'League': league_names[league], 'Accuracy': accuracy})

    # Create DataFrame
    df = pd.DataFrame(data)

    # Calculate the average accuracy for each league
    average_accuracies = df.groupby('League')['Accuracy'].mean().reset_index()

    # Plot the line graph
    plt.figure(figsize=(10, 6))
    for league in league_names.values():
        league_data = df[df['League'] == league]
        plt.plot(league_data['Year'], league_data['Accuracy'], marker='o', label=league, color=league_colors[league])

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of Colley Model from 2010-2023')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(average_accuracies['League'], average_accuracies['Accuracy'], color=[league_colors[league] for league in average_accuracies['League']])

    # Set the y-axis range
    plt.ylim(0.5, 0.7)

    # Add labels and title
    plt.xlabel('League')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy of Colley Model from 2010-2023')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

    # Show the plot
    plt.show()
