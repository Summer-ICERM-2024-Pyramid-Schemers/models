import numpy as np
import pandas as pd
import scipy.stats as stats

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
        Ch = [np.mean(b1) - np.mean(b2), ttest[1], ci[0], ci[1]]
        b1 = getM1BrierScores(season, 3)
        b2 = getM2BrierScores(season, 3)
        ttest = stats.ttest_rel(b1,b2)
        l1 = [np.mean(b1) - np.mean(b2), ttest[1], ci[0], ci[1]]
        b1 = getM1BrierScores(season, 4)
        b2 = getM2BrierScores(season, 4)
        ttest = stats.ttest_rel(b1,b2)
        l2 = [np.mean(b1) - np.mean(b2), ttest[1], ci[0], ci[1]]
        comparison = pd.concat([pd.DataFrame([[season, prem, Ch, l1, l2]], 
                                        columns=comparison.columns), comparison], 
                                        ignore_index=True)
    return comparison