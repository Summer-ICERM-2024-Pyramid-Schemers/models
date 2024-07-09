import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from statsmodels.miscmodels.ordinal_model import OrderedModel

# load functions
from Weighted_Colley import Colley

# seasons to plot (starting season has to be >= (earliest season+2))
DEFAULT_SEASONS = range(2012,2024)


def prepareData():
    con = sqlite3.connect('english_football_data.sqlite')

    gamesQuery = f"""
    SELECT season, league_id, date, home_team_id, away_team_id, fulltime_home_goals, fulltime_away_goals, fulltime_result
    FROM Matches
    """

    Games = pd.read_sql_query(gamesQuery, con)
    con.close()

    # turn game result to numerical value
    conditions = [
        Games['fulltime_result'] == 'H',
        Games['fulltime_result'] == 'A',
        Games['fulltime_result'] == 'D'
    ]
    results = [1, -1, 0]
    Games['result'] = np.select(conditions, results, default=np.nan)
    Games['result'] = Games['result'].astype(int)


    return Games



def getData(season, league):
    """ 
    Function returns a dataframe which contains the game statistics of given season and league (if input includes a value league id, otherwise it returns stats of all games in that season)
        - one row of dataframe represents 1 game
        - each row also contains 
            1) massey ratings of home and away team, which are calculated using data from all previous seasons
            2) rating differentials = home team rating - away team rating
    """
    Games = prepareData()

    pre_Games = Games.loc[Games['season'] < season, ]
   
    # get WEIGHTED colley ratings using data before predict season
    colley = Colley(goals_home=pre_Games['fulltime_home_goals'], 
                                goals_away=pre_Games['fulltime_away_goals'], 
                                teams_home=pre_Games['home_team_id'],
                                teams_away=pre_Games['away_team_id'],
                                match_date=pre_Games['date'],
                                )
    colley_ratings = colley.get_ratings()

    # merge ratings with game data before predict season
    ratings = colley_ratings[['team', 'rating'] ]
    ratings.loc[:, 'season'] = season


    home_rating = ratings.rename(columns={'team': 'home_team_id', 
                                        'rating': 'home_rating'})
    away_rating = ratings.rename(columns={'team': 'away_team_id', 
                                        'rating': 'away_rating'})
    Year_Games = Games.loc[Games['season']==season, ]
    Year_Games = pd.merge(Year_Games, home_rating, on=['home_team_id', 'season'], how='left')
    Year_Games = pd.merge(Year_Games, away_rating, on=['away_team_id', 'season'], how='left')
    Year_Games = Year_Games.dropna()

    Year_Games['rating_diff'] = Year_Games['home_rating'] - Year_Games['away_rating']

    if league is not None:
        Year_Games = Year_Games.loc[Year_Games['league_id'] == league, ]

    return Year_Games


def getPredProb(season, league):
    """
    Get ordered probit model using game data before predict season
        y(win/draw/loss) = beta * (r_h - r_a)

    Return a dataframe containing 
        - game statistics of predict season
        - predicted probability of win, draw, loss of each game in predict season 
    """

    frames = []
    # change 2011 to (earliest season + 1) if dataset is updated
    for Year in range(2011, season):
        Year_Games = getData(Year, None)
        frames.append(Year_Games)
    # Concatenate all collected DataFrames into a single DataFrame
    pre_data = pd.concat(frames, ignore_index=True)

    model = OrderedModel(pre_data['result'],pre_data['rating_diff'],distr = 'probit')
    model = model.fit(method='bfgs')

    # get games data and massey ratings for predict season and league
    pred_data = getData(season, league)

    predictions = [model.predict(i)[0] for i in pred_data["rating_diff"]]
    # Careful, the model.predict should return in the order of loss, draw, win...
    pred_data[["pred-loss","pred-draw","pred-win"]] = predictions
    
    return pred_data


def getBrierScores(season, league):
    """
    Compute Brier score of all games in a given season and given league using
        BS = 1/3 * ( (prob_win - win)^2 +
                     (prob_draw - draw)^2 +
                     (prob_loss - loss)^2   )
    """

    pred_data = getPredProb(season, league)    
    brierScores = []
    for i in range(len(pred_data)):
        match = pred_data.iloc[i]
        result = match['result']
        win = 0
        draw = 0
        loss = 0
        if result == 1:
            win = 1
        elif result == 0:
            draw = 1
        elif result == -1:
            loss = 1

        score = 1/3 * (np.square(match['pred-win'] - win) + 
                        np.square(match['pred-draw'] - draw) +
                        np.square(match['pred-loss'] - loss))
        
        brierScores.append(score)

    return brierScores


def plotBrierScores(seasons=DEFAULT_SEASONS, *args, title=None, filename=None):
    """
    Plot the average Brier scores of all leagues and seasons.
    """

    if title is None:
        title = "Colley Brier Score by Season and League"
    if filename is None:
        filename = "wtd_colley_brier_scores.png"
    
    briers = pd.DataFrame([[np.mean(getBrierScores(season, league)) 
                            for league in range(1,5)] for season in seasons],
                            columns=['Premier League', 'Championship','League One','League Two'], 
                            index=seasons)

    briers.plot()

    plt.title(title)
    plt.xlabel('Season')
    plt.ylabel('Brier Score')
    plt.grid(True)
    plt.show()

plotBrierScores()