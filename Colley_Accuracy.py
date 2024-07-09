import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3

class Colley:
    """
    Calculates each team's Colley ratings

    Parameters
    ----------
    goals_home : list
        List of goals scored by the home teams

    goals_away : list
        List of goals scored by the away teams

    teams_home : list
        List of names of the home teams

    teams_away : list
        List of names of the away teams

    include_draws : bool
        Should tied results be included in the ratings?

    draw_weight : float
        if include_draws is `True` then this sets the weighting applied to tied scores.
        For example `0.5` means a draw is worth half a win, `0.333` means a draw
        is a third of a win etc
    """

    def __init__(
        self,
        goals_home,
        goals_away,
        teams_home,
        teams_away,
        include_draws=True,
        draw_weight=0.5,
    ):
        self.goals_home = goals_home
        self.goals_away = goals_away
        self.teams_home = teams_home
        self.teams_away = teams_away
        self.include_draws = include_draws
        self.draw_weight = draw_weight

    def get_ratings(self) -> pd.DataFrame:
        """
        Gets the Colley ratings

        Returns
        -------
            Returns a dataframe containing colley ratings per team
        """
        teams = np.sort(np.unique(np.concatenate([self.teams_home, self.teams_away])))

        fixtures = _build_fixtures(
            self.goals_home, self.goals_away, self.teams_home, self.teams_away
        )

        C, b = _build_C_b(fixtures, teams, self.include_draws, self.draw_weight)

        r = _solve_r(C, b)
        r = pd.DataFrame([teams, r]).T
        r.columns = ["team", "rating"]
        r = r.sort_values("rating", ascending=False)
        r = r.reset_index(drop=True)

        return r


def _build_fixtures(goals_home, goals_away, teams_home, teams_away):
    fixtures = pd.DataFrame([goals_home, goals_away, teams_home, teams_away]).T
    fixtures.columns = ["goals_home", "goals_away", "team_home", "team_away"]
    fixtures["goals_home"] = fixtures["goals_home"].astype(int)
    fixtures["goals_away"] = fixtures["goals_away"].astype(int)
    return fixtures


def _solve_r(C, b):
    r = np.linalg.solve(C, b)
    return r


def _build_C_b(fixtures, teams, include_draws, draw_weight):
    n_teams = len(teams)
    C = np.zeros([n_teams, n_teams])
    b = np.zeros([n_teams])

    for _, row in fixtures.iterrows():
        h = np.where(teams == row["team_home"])[0][0]
        a = np.where(teams == row["team_away"])[0][0]

        C[h, a] = C[h, a] - 1
        C[a, h] = C[a, h] - 1

        if row["goals_home"] > row["goals_away"]:
            b[h] += 1
            b[a] -= 1

        elif row["goals_home"] < row["goals_away"]:
            b[h] -= 1
            b[a] += 1

        else:
            if include_draws:
                b[h] += draw_weight
                b[a] += draw_weight

    np.fill_diagonal(C, np.abs(C.sum(axis=1)) + 2)
    b = 1 + b * 0.5

    return C, b


def get_colley(season,league):
        
    con = sqlite3.connect("english_football_data.sqlite")

    gamesQuery = f"""
    SELECT home_team_id, away_team_id, fulltime_home_goals, fulltime_away_goals, fulltime_result
    FROM Matches
    WHERE season = {season}
    AND league_id = {league}
    """
    teamsQuery = f"""
    SELECT id, name
    FROM Teams
    """


    Games = pd.read_sql_query(gamesQuery, con)
    Names = pd.read_sql_query(teamsQuery, con)
    merged_df = pd.merge(Games, Names, left_on='home_team_id', right_on='id', how='left')
    merged_df = merged_df.rename(columns={'name': 'home_team_name'})
    new_df = merged_df.drop(columns=['id'])
    final_df = pd.merge(new_df, Names, left_on='away_team_id', right_on='id', how='left')
    final_df = final_df.rename(columns={'name': 'away_team_name'})
    final_df = final_df.rename(columns={'name': 'away_team_name'})
    Games = final_df.drop(columns=['id'])

    home_goals_list = Games["fulltime_home_goals"].tolist()
    away_goals_list = Games["fulltime_away_goals"].tolist()
    home_teams_list = Games['home_team_id'].tolist()
    away_teams_list = Games['away_team_id'].tolist()
    if(league==1):
        colley = Colley(home_goals_list[:305],away_goals_list[:305],home_teams_list[:305], away_teams_list[:305])
    elif((season==2019)and league != 2):
        if(league==3):
           colley = Colley(home_goals_list[:321],away_goals_list[:321],home_teams_list[:321], away_teams_list[:321])
        if(league==4):
            colley = Colley(home_goals_list[:353],away_goals_list[:353],home_teams_list[:353], away_teams_list[:353])
    else:
        colley = Colley(home_goals_list[:443],away_goals_list[:443],home_teams_list[:443], away_teams_list[:443])
    colley_ratings = colley.get_ratings()

    dictionary = colley_ratings.set_index('team')['rating'].to_dict()

    Games['home_team_colley'] = Games['home_team_id'].map(dictionary)
    Games['away_team_colley'] = Games['away_team_id'].map(dictionary)
    
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


def get_accuracy(season,league):
    games_table = get_colley(season,league)
    true_counter=0
    total_counter=0
    for index, row in games_table.iterrows():
        if (row['home_team_colley'] > row['away_team_colley']) and (row['fulltime_result'] == 'H') or ((row['home_team_colley'] < row['away_team_colley']) and (row['fulltime_result'] == 'A')):
            true_counter +=1
            total_counter +=1
        elif(row['fulltime_result'] != 'D'):
            total_counter += 1
    return true_counter / total_counter


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
