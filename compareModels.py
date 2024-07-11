import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from oddsModel import BettingOddsModel
from homeAdvModel import HomeAdvModel
from transfermarktModel import TMModelOrderedProbit, TMModelOrderedProbitOLSGoalDiff
from masseyModel import MasseyModel, WeightedMasseyModel

def compareModels(getM1BrierScores, getM2BrierScores, country, excludel2 = False):
    """
    Returns a dataframe containing a list for each league and season.

    Inputs should be functions that return a list of Brier scores for a given season and league
    
    Elements of list:
    [0] - The difference in mean Brier scores of M1 - M2
    [1] - The 2-sided p-value of a paired t-test for difference in means
    [2] - A lower 95% confidence interval bound for Brier score difference
    [3] - An upper 95% confidence interval bound for Brier score difference
    """
    if country.casefold() == "england":
        comparison = pd.DataFrame(columns=['Year', 'Premier League','Championship','League One','League Two'])
    elif country.casefold() == "germany":
        comparison = pd.DataFrame(columns=['Year', 'Bundesliga', '2. Bundesliga'])
    else:
        raise Exception("Invalid country")
    for season in range(2012, 2024, 1):
        # For each league, calculate raw difference M1-M2 Brier score
        # Then calculate p value from t test
        # Then add lower and upper CIs
        # Add list to slot in table
        if country.casefold() == "england":
            b1 = getM1BrierScores(season, 1)
            b2 = getM2BrierScores(season, 1)
            ttest = stats.ttest_rel(b1,b2)
            ci = ttest.confidence_interval()
            prem = [np.mean(b1) - np.mean(b2), ttest[1], ci[0], ci[1]]
            b1 = getM1BrierScores(season, 2)
            b2 = getM2BrierScores(season, 2)
            ttest = stats.ttest_rel(b1,b2)
            ci = ttest.confidence_interval()
            Ch = [np.mean(b1) - np.mean(b2), ttest[1], ci[0], ci[1]]
            b1 = getM1BrierScores(season, 3)
            b2 = getM2BrierScores(season, 3)
            ttest = stats.ttest_rel(b1,b2)
            ci = ttest.confidence_interval()
            l1 = [np.mean(b1) - np.mean(b2), ttest[1], ci[0], ci[1]]
            if excludel2 == False:
                b1 = getM1BrierScores(season, 4)
                b2 = getM2BrierScores(season, 4)
                ttest = stats.ttest_rel(b1,b2)
                ci = ttest.confidence_interval()
                l2 = [np.mean(b1) - np.mean(b2), ttest[1], ci[0], ci[1]]
            elif excludel2 == True:
                l2 = [0,0,0,0]
            comparison = pd.concat([pd.DataFrame([[season, prem, Ch, l1, l2]], 
                                            columns=comparison.columns), comparison], 
                                            ignore_index=True)
        elif country.casefold() == "germany":
            b1 = getM1BrierScores(season, 5)
            b2 = getM2BrierScores(season, 5)
            ttest = stats.ttest_rel(b1,b2)
            ci = ttest.confidence_interval()
            bund1 = [np.mean(b1) - np.mean(b2), ttest[1], ci[0], ci[1]]
            b1 = getM1BrierScores(season, 6)
            b2 = getM2BrierScores(season, 6)
            ttest = stats.ttest_rel(b1,b2)
            ci = ttest.confidence_interval()
            bund2 = [np.mean(b1) - np.mean(b2), ttest[1], ci[0], ci[1]]
            comparison = pd.concat([pd.DataFrame([[season, bund1, bund2]], 
                                            columns=comparison.columns), comparison], 
                                            ignore_index=True)
    return comparison

def plotComparison(getM1BrierScores, getM2BrierScores, M1Title, M2Title, country, excludel2 = False):
    comparison = compareModels(getM1BrierScores, getM2BrierScores, country, excludel2)
    comparison = comparison.set_index("Year")
    diffs = comparison.applymap(lambda x: x[0])
    pvalues = comparison.applymap(lambda x: x[1])
    lowerCI = comparison.applymap(lambda x: x[2])
    upperCI = comparison.applymap(lambda x: x[3])
    
    df_combined = diffs.stack().reset_index()
    df_combined.columns = ['Season', 'League', 'Brier_Diff']
    df_combined['pvalue'] = pvalues.stack().values
    
    # Calculate error bounds
    df_combined['lower_error'] = np.abs(df_combined['Brier_Diff'] - lowerCI.stack().values)
    df_combined['upper_error'] = np.abs(upperCI.stack().values - df_combined['Brier_Diff'])
    # Plot points
    sns.lineplot(data=df_combined, x='Season', y='Brier_Diff', hue='League', style='League')
    # Add error bars
    for league in df_combined['League'].unique():
        league_data = df_combined[df_combined['League'] == league]
        plt.errorbar(league_data['Season'], league_data['Brier_Diff'],
                     yerr=[league_data['lower_error'], league_data['upper_error']],
                     fmt='none', capsize=5, alpha=0.5, linestyle='--')

    # Customize the plot
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Comparison of Brier Scores: {M1Title} - {M2Title}')
    plt.xlabel('Season')
    plt.ylabel('Difference in Brier Score')
    plt.legend(title='League')
    plt.savefig(M1Title + " - " + M2Title)

    plt.show()

def compareModelsAggregate(getM1BrierScores, getM2BrierScores, country, excludel2 = False):
    """
    Returns a data frame with a column for each league, with a list containing aggregate data

    Inputs should be functions that return a list of Brier scores for a given season and league
    
    Elements of list:
    [0] - The difference in mean Brier scores of M1 - M2
    [1] - The 2-sided p-value of a paired t-test for difference in means
    [2] - A lower 95% confidence interval bound for Brier score difference
    [3] - An upper 95% confidence interval bound for Brier score difference
    """
    
    if country.casefold() == "england":
        comparison = pd.DataFrame(columns=['Premier League','Championship','League One','League Two'])
        l1m1 = []
        l2m1 = []
        l3m1 = []
        l4m1 = []
        l1m2 = []
        l2m2 = []
        l3m2 = []
        l4m2 = []
    elif country.casefold() == "germany":
        comparison = pd.DataFrame(columns=['Bundesliga', '2. Bundesliga'])
        l5m1 = []
        l5m2 = []
        l6m1 = []
        l6m2 = []
    else:
        raise Exception("Invalid country")
    
    if country.casefold() == 'england':
        for season in range(2012, 2024, 1):
            l1m1.extend(getM1BrierScores(season, 1))
            l1m2.extend(getM2BrierScores(season, 1))

            l2m1.extend(getM1BrierScores(season, 2))
            l2m2.extend(getM2BrierScores(season, 2))

            l3m1.extend(getM1BrierScores(season, 3))
            l3m2.extend(getM2BrierScores(season, 3))

            if excludel2 == False:
                l4m1.extend(getM1BrierScores(season, 4))
                l4m2.extend(getM2BrierScores(season, 4))

        ttest = stats.ttest_rel(l1m1, l1m2)
        ci = ttest.confidence_interval()
        prem = [np.mean(l1m1) - np.mean(l1m2), ttest[1], ci[0], ci[1]]

        ttest = stats.ttest_rel(l2m1, l2m2)
        ci = ttest.confidence_interval()
        Ch = [np.mean(l2m1) - np.mean(l2m2), ttest[1], ci[0], ci[1]]

        ttest = stats.ttest_rel(l3m1, l3m2)
        ci = ttest.confidence_interval()
        l1 = [np.mean(l3m1) - np.mean(l3m2), ttest[1], ci[0], ci[1]]

        if excludel2 == False:
            ttest = stats.ttest_rel(l4m1, l4m2)
            ci = ttest.confidence_interval()
            l2 = [np.mean(l4m1) - np.mean(l4m2), ttest[1], ci[0], ci[1]]
        elif excludel2 == True:
            l2 = [0,0,0,0]

        comparison = pd.concat([pd.DataFrame([[prem, Ch, l1, l2]], 
                                            columns=comparison.columns), comparison], 
                                            ignore_index=True)
    elif country.casefold() == 'germany':
        for season in range(2012, 2024, 1):
            l5m1.extend(getM1BrierScores(season, 5))
            l5m2.extend(getM2BrierScores(season, 5))

            l6m1.extend(getM1BrierScores(season, 6))
            l6m2.extend(getM2BrierScores(season, 6))

        ttest = stats.ttest_rel(l5m1, l5m2)
        ci = ttest.confidence_interval()
        bund1 = [np.mean(l5m1) - np.mean(l5m2), ttest[1], ci[0], ci[1]]

        ttest = stats.ttest_rel(l6m1, l6m2)
        ci = ttest.confidence_interval()
        bund2 = [np.mean(l6m1) - np.mean(l6m2), ttest[1], ci[0], ci[1]]

        comparison = pd.concat([pd.DataFrame([[bund1, bund2]],
                                            columns=comparison.columns), comparison], 
                                            ignore_index=True)
    return comparison


def plotAggregateComparison(getM1BrierScores, getM2BrierScores, M1Title, M2Title, country, excludel2 = False):
    comparison = compareModelsAggregate(getM1BrierScores, getM2BrierScores, country, excludel2)

    diffs = comparison.applymap(lambda x: x[0])
    pvalues = comparison.applymap(lambda x: x[1])
    lowerCI = comparison.applymap(lambda x: x[2])
    upperCI = comparison.applymap(lambda x: x[3])
    
    df_combined = diffs.stack().reset_index()
    df_combined = df_combined.drop('level_0', axis=1)
    df_combined.columns = ['League', 'Brier_Diff']
    df_combined['pvalue'] = pvalues.stack().values
    
    # Calculate error bounds
    df_combined['lower_error'] = np.abs(df_combined['Brier_Diff'] - lowerCI.stack().values)
    df_combined['upper_error'] = np.abs(upperCI.stack().values - df_combined['Brier_Diff'])
    # Plot points
    sns.barplot(data=df_combined, x='League', y='Brier_Diff', color = 'blue')
    # Add error bars
    for league in df_combined['League'].unique():
        league_data = df_combined[df_combined['League'] == league]
        plt.errorbar(league_data['League'], league_data['Brier_Diff'],
                     yerr=[league_data['lower_error'], league_data['upper_error']],
                     fmt='none', capsize=5, alpha=1, linestyle='--', ecolor='red', elinewidth=3)

    # Customize the plot
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Comparison of Brier Scores: {M1Title} - {M2Title}')
    plt.xlabel('League')
    plt.ylabel('Difference in Brier Score')
    plt.savefig(M1Title + " - " + M2Title + " AGG")

    plt.show()

def compareSRAggregate(getM1SuccessRatio, getM2SuccessRatio, country, excludel2 = False):
    """
    Returns a data frame with a column for each league, with a list containing aggregate data

    Inputs should be functions that return a list of Brier scores for a given season and league
    
    Elements of list:
    [0] - The difference in mean Brier scores of M1 - M2
    [1] - The 2-sided p-value of a paired t-test for difference in means
    [2] - A lower 95% confidence interval bound for Brier score difference
    [3] - An upper 95% confidence interval bound for Brier score difference
    """

    if country.casefold() == "england":
        comparison = pd.DataFrame(columns=['Premier League','Championship','League One','League Two'])
        l1m1, l1m2, l2m1, l2m2, l3m1, l3m2, l4m1, l4m2 = 0,0,0,0,0,0,0,0
    elif country.casefold() == "germany":
        comparison = pd.DataFrame(columns=['Bundesliga', '2. Bundesliga'])
        l5m1, l5m2, l6m1, l6m2 = 0,0,0,0
    else:
        raise Exception("Invalid country")

    if country.casefold() == "england":
        for season in range(2012, 2024, 1):
            div = season - 2011
            mult = season - 2012
            # Each proportion is of the same number of games...
            l1m1 = (getM1SuccessRatio(season, 1) + (l1m1 * mult)) / div
            l1m2 = (getM2SuccessRatio(season, 1) + (l1m2 * mult)) / div
            l2m1 = (getM1SuccessRatio(season, 2) + (l2m1 * mult)) / div
            l2m2 = (getM2SuccessRatio(season, 2) + (l2m2 * mult)) / div
            l3m1 = (getM1SuccessRatio(season, 3) + (l3m1 * mult)) / div
            l3m2 = (getM2SuccessRatio(season, 3) + (l3m2 * mult)) / div
            l4m1 = (getM1SuccessRatio(season, 4) + (l4m1 * mult)) / div
            l4m2 = (getM2SuccessRatio(season, 4) + (l4m2 * mult)) / div
        
        prem = l1m1 - l1m2
        Ch = l2m1 - l2m2
        l1 = l3m1 - l3m2
        if excludel2 == False:
            l2 = l4m1 - l4m2
        elif excludel2 == True:
            l2 = 0

        comparison = pd.concat([pd.DataFrame([[prem, Ch, l1, l2]], 
                                            columns=comparison.columns), comparison], 
                                            ignore_index=True)
        
    elif country.casefold() == "germany":
        for season in range(2012, 2024, 1):
            div = season - 2011
            mult = season - 2012
            # Each proportion is of the same number of games...
            l5m1 = (getM1SuccessRatio(season, 5) + (l5m1 * mult)) / div
            l5m2 = (getM2SuccessRatio(season, 5) + (l5m2 * mult)) / div
            l6m1 = (getM1SuccessRatio(season, 6) + (l6m1 * mult)) / div
            l6m2 = (getM2SuccessRatio(season, 6) + (l6m2 * mult)) / div
        
        bund1 = l5m1 - l5m2
        bund2 = l6m1 - l6m2

        comparison = pd.concat([pd.DataFrame([[bund1, bund2]], 
                                            columns=comparison.columns), comparison], 
                                            ignore_index=True)

    return comparison

def plotSRAggregateComparison(getM1SuccessRatio, getM2SuccessRatio, M1Title, M2Title, country, excludel2 = False):
    diffs = compareSRAggregate(getM1SuccessRatio, getM2SuccessRatio, country, excludel2)
    print(diffs)
    
    df_combined = diffs.stack().reset_index()
    df_combined = df_combined.drop('level_0', axis=1)
    df_combined.columns = ['League', 'SR_Diff']

    # Plot points
    sns.barplot(data=df_combined, x='League', y='SR_Diff', color = 'blue')

    # Customize the plot
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Comparison of Success Ratios: {M1Title} - {M2Title}')
    plt.xlabel('League')
    plt.ylabel('Difference in Success Ratios')
    plt.savefig(M1Title + " - " + M2Title + " AGG")

    plt.show()

# Odds - HA
#plotComparison(BettingOddsModel.getBrierScores, HomeAdvModel.getBrierScores, "Betting Odds", "Home Advantage", "Germany")
plotAggregateComparison(BettingOddsModel.getBrierScores, HomeAdvModel.getBrierScores, "Betting Odds", "Home Advantage", "Germany")

# Odds - TM1
#plotComparison(BettingOddsModel.getBrierScores, TMModelOrderedProbit.getBrierScores, "Betting Odds", "Transfermarkt Model 1")
#plotAggregateComparison(BettingOddsModel.getBrierScores, TMModelOrderedProbit.getBrierScores, "Betting Odds", "Transfermarkt Model 1")

# Odds - TM2
#plotComparison(BettingOddsModel.getBrierScores, TMModelOrderedProbitOLSGoalDiff.getBrierScores, "Betting Odds", "Transfermarkt Model 2")
#plotAggregateComparison(BettingOddsModel.getBrierScores, TMModelOrderedProbitOLSGoalDiff.getBrierScores, "Betting Odds", "Transfermarkt Model 2")

# TM1 - TM2
#plotComparison(TMModelOrderedProbit.getBrierScores, TMModelOrderedProbitOLSGoalDiff.getBrierScores, "Transfermarkt Model 1", "Transfermarkt Model 2")
#plotAggregateComparison(TMModelOrderedProbit.getBrierScores, TMModelOrderedProbitOLSGoalDiff.getBrierScores, "Transfermarkt Model 1", "Transfermarkt Model 2")

# TM2 - HA
#plotComparison(TMModelOrderedProbitOLSGoalDiff.getBrierScores, HomeAdvModel.getBrierScores, "Transfermarkt Model 2", "Home Advantage")
#plotAggregateComparison(TMModelOrderedProbitOLSGoalDiff.getBrierScores, HomeAdvModel.getBrierScores, "Transfermarkt Model 2", "Home Advantage")