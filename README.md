# Models

## Data

The data for this project is compiled by [football-data-scraping](https://github.com/Summer-ICERM-2024-Pyramid-Schemers/football-data-scraping) into a SQLite database which resides in the "data/" folder.

## Installation

1. Ensure a recent version of Python3 is installed.
2. Clone this repository and cd into the folder.
3. Install the modules using `pip`. Depending on your python installation, the command might be `pip install -r requirements.txt` or `python3 -m pip install -r requirements.txt`.

## Usage

To run a script (which resides in the top level of the repository), use the command `python {SCRIPT_FILE}`. For example: `python colleyAccuracy.py` will run the "colleyAccuracy.py" file.

### Running Models
To run a model (which resides in "src/models/"), use the script "eval_model.py". The syntax is `python eval_model.py [--country {all,england,germany,scotland}] {MODEL_NAME}`. For example, `python eval_model.py --country germany bettingodds` will run the betting odds model from "src/models/oddsModel.py".
The list of model names is:
- Homeadv
- Colley
- Wtdcolley
- Massey
- Wtdmassey
- TM
- TMOLS
- Bettingodds

Model names are case-insensitive and have multiple aliases.

# Model Overview

## General Model Structure

All models inherit from the abstract base class `BaseModel`. It contains some helper methods like `BaseModel._calc_brier_scores` and `BaseModel._calc_success_ratio`.

It also has the method `BaseModel.plotBrierScores` which will plot the brier scores for the specified seasons and leagues. Subclasses can set the class variables `_plot_seasons`, `_plot_title`, and `_plot_filename` to customize the behavior of the method without needing to override the method, though it can be done if necessary.

Subclasses should override `BaseModel.getBrierScores` and `BaseModel.getSuccessRatio`.

## HomeAdvantage (Null)

The HomeAdvantage model only considers home advantage in its predictions and nothing else. As such, it is the most naive model and is our baseline to compare against.

The model class is stored in "src/models/homeAdvModel.py"

## Colley

The Colley model takes into account a team's win percentage and strength of schedule.

Weighted Colley weights the games based on the date each game was played on; meanwhile, unweighted Colley weights all games the same regardless of time.

The model classes are stored in "src/models/colleyModel.py" and the process of setting up and solving the Colley system is done in "src/weighted_colley_engine.py"

## Massey

The Massey model is similar to Colley but takes into account goal differentials. 

Weighted Massey weights games based on the date a game was played and the average market values of team lineups.

The model classes are stored in "src/models/masseyModel.py" and the process of setting up and solving the Massey system is done in "src/weighted_massey_engine.py"

## Transfermarkt Regression

The Transfermarkt regression model uses the market value of both teams and home advantage in its predictions. Technically, the model uses the difference between the log1p of the lineups' average market value.

We also have a regression model that factors in goal differentials, but we found that the model showed little to no improvement from the change.

The model classes are stored in "src/models/transfermarktModel.py"

## Betting Odds

The betting odds model is the best possible model for us because we assume the betting odds factor in all available data. We simply take the betting odds of home team winning, away team winning, and the teams tie and convert them into implied probabilities.

The model class is stored in "src/models/oddsModel.py"

# Comparing and Evaluating Models

The top level scripts evaluate models in different ways:
* "colleyAccuracy.py" calculates the accuracy of the unweighted and weighted Colley models across the 4 English leagues
* "ColleyEOSRankEval.py" calculates the kendall tau rank correlation between the Colley rankings and EOS rankings
* "compareModels.py" contains a few different methods for comparing the models. Note that this file is incredibly hardcoded
* "inSeasonBrier.py" trains models on the first 80% of a season, then evaluates their performance on the remaining 20%
* "masseyAccuracy.py" calculates the accuracy of the unweighted and weighted Massey models across the 4 English leagues
* "MasseyEOSRankEval.py" calculates the kendall tau rank correlation between the Massey rankings and EOS rankings
* "outseasonBrierPlot.py" calculates the brier scores of the models for the 4 English leagues using data from prior seasons
