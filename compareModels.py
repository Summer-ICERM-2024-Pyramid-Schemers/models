import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from oddsModel import getOddsBrierScores
from homeAdvModel import getHABrierScores
from transfermarktModel import TMModelOrderedProbit, TMModelOrderedProbitOLSGoalDiff

def compareModels(getM1BrierScores, getM2BrierScores):
    """
    Returns a dataframe containing a list for each league and season.

    Inputs should be functions that return a list of Brier scores for a given season and league
    
    Elements of list:
    [0] - The difference in mean Brier scores of M1 - M2
    [1] - The 2-sided p-value of a paired t-test for difference in means
    [2] - A lower 95% confidence interval bound for Brier score difference
    [3] - An upper 95% confidence interval bound for Brier score difference
    """
    comparison = pd.DataFrame(columns=['Year', 'Premier League','Championship','League One','League Two'])
    for season in range(2010, 2024, 1):
        # For each league, calculate raw difference M1-M2 Brier score
        # Then calculate p value from t test
        # Then add lower and upper CIs
        # Add list to slot in table
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
        b1 = getM1BrierScores(season, 4)
        b2 = getM2BrierScores(season, 4)
        ttest = stats.ttest_rel(b1,b2)
        ci = ttest.confidence_interval()
        l2 = [np.mean(b1) - np.mean(b2), ttest[1], ci[0], ci[1]]
        comparison = pd.concat([pd.DataFrame([[season, prem, Ch, l1, l2]], 
                                        columns=comparison.columns), comparison], 
                                        ignore_index=True)
    return comparison

def plotComparison(getM1BrierScores, getM2BrierScores, M1Title, M2Title):
    comparison = compareModels(getM1BrierScores, getM2BrierScores)
    comparison = comparison.set_index("Year")
    diffs = comparison.applymap(lambda x: x[0])
    pvalues = comparison.applymap(lambda x: x[1])
    lowerCI = comparison.applymap(lambda x: x[2])
    upperCI = comparison.applymap(lambda x: x[3])
    print(lowerCI)
    print(upperCI)

    
    df_combined = diffs.stack().reset_index()
    df_combined.columns = ['Season', 'League', 'Brier_Diff']
    df_combined['pvalue'] = pvalues.stack().values
    
    # Calculate error bounds
    df_combined['lower_error'] = np.abs(df_combined['Brier_Diff'] - lowerCI.stack().values)
    df_combined['upper_error'] = np.abs(upperCI.stack().values - df_combined['Brier_Diff'])
    print(df_combined)
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
    plt.savefig("OddsvsTMmodel1")
    
    # Display the plot
    plt.show()


plotComparison(getOddsBrierScores, TMModelOrderedProbit.getBrierScores, "Odds", "Transfermarkt Model 1")