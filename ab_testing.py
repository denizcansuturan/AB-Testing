######################################################
# Basic Statistics
######################################################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


############################
# Sampling 
############################

populasyon = np.random.randint(0, 80, 10000)
# random population is created with numpy
# let's say it is age info of 10000 people
populasyon.mean()
# why do we work with the sample and not the population.
#


np.random.seed(115)

orneklem = np.random.choice(a=populasyon, size=100)
# sample is chosen from population

orneklem.mean()


np.random.seed(10)
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

x = (orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10
# closer to population mean than first sample's mean

############################
# Descriptive Statistics
############################

df = sns.load_dataset("tips")
df.describe().T

############################
# Confidence Intervals 
############################

# Confidence Interval Calculation for Numeric Variables in the Tips Dataset
df = sns.load_dataset("tips")
df.describe().T
# restaurant owner needs the total bill on a bad day
# average does not give a general lower bound

df.head()

sms.DescrStatsW(df["total_bill"]).tconfint_mean()
# confidence interval is calculated above
# it means that in any sample that is from this total bill population
# total bill will be in this interval with 95% probability.
# this way, owner knows the best and worst scenario with 5% mistake probability in a day

sms.DescrStatsW(df["tip"]).tconfint_mean()
# confidence interval is calculated above
# it means that in any sample that is from this total bill population
# total bill will be in this interval with 95% probability.
# this way, owner knows the best and worst scenario with 5% mistake probability in a day

# Confidence Interval Calculation for Numeric Variables in the Titanic Dataset
df = sns.load_dataset("titanic")
df.describe().T

sms.DescrStatsW(df["age"].dropna()).tconfint_mean()
# null values must be excluded

sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()


######################################################
# Correlation 
######################################################

df = sns.load_dataset('tips')
df.head()

df["total_bill"] = df["total_bill"] - df["tip"]

df.plot.scatter("tip", "total_bill")
plt.show()

df["tip"].corr(df["total_bill"])

######################################################
# AB Testing (Independent Two Sample T Test)
######################################################
# 1. Formulate Hypothesis
# 2. Assumption Checks
#   - Normality Assumption
#   - Homogeneity of Variance
# 3. Apply the Hypothesis Test
#   - If assumptions are met, use independent two-sample t-test (parametric test).
#   - If assumptions are not met, use Mann-Whitney U test (non-parametric test).
# 4. Interpret the results based on the p-value.
# Note:
# - If normality assumption is not met, go directly to step 2. If homogeneity of variance assumption is not met, enter the argument to step 1.
# - It may be useful to examine and correct outliers before examining normality.

############################
# Application 1: Is there a statistically significant difference between the average bills of smokers and non-smokers?
############################


df = sns.load_dataset("tips")
df.head()

df.groupby("smoker").agg({"total_bill": "mean"})

############################
# 1. Formulate Hypothesis
############################

# H0: M1 = M2
# H1: M1 != M2

############################
# 2. Assumption Checks
############################

# Normality Assumption
# Homogeneity of Variance

############################
# Normality Assumption
############################

# H0: Normality Assumption is being met.
# H1: Normality Assumption is not being met.


test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < 0.05 HO is rejected.
# p-value < 0.05 HO is not rejected.


test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


############################
# Homogeneity of Variance
############################

# H0: Variances are homogenous.
# H1: Variances are not homogenous.

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < 0.05 HO is rejected.
# p-value < 0.05 HO is not rejected.

############################
# 3 and 4. Apply the Hypothesis Test and Interpret the results based on the p-value.
############################

#  - If assumptions are met, use independent two-sample t-test (parametric test).
#  - If assumptions are not met, use Mann-Whitney U test (non-parametric test).

############################
# 1.If assumptions are met, use independent two-sample t-test (parametric test).
############################

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

############################
# 2.If assumptions are not met, use Mann-Whitney U test (non-parametric test).
############################

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

############################
# Application 2: Is there a statistically significant difference between the average ages of male and female passengers on the Titanic?
############################

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})


# 1. Formulate Hypothesis:
# H0: M1  = M2 
# H1: M1! = M2 

# 2. Assumption Checks

# Normality Assumption:
# H0: Normality Assumption is being met.
# H1: Normality Assumption is not being met.


test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Homogeneity of Variance:
# H0: Variances are homogenous.
# H1: Variances are not homogenous.

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Assumptions are not met, use Mann-Whitney U test (non-parametric test).

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


############################
# Application 3: Is there a statistically significant difference between the average ages of individuals with and without diabetes?
############################

df = pd.read_csv("datasets/diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"})

# 1. Formulate Hypothesis:
# H0: M1 = M2
# there is not a statistically significant difference between the average ages
# H1: M1 != M2
# there is a statistically significant difference between the average ages

# 2. Assumption Checks

# Normality Assumption:

# H0: Normality Assumption is being met.
# H1: Normality Assumption is not being met.

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# Since Normality Assumption is being met nonparametic test will be used.

# Hypothesis (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


###################################################
# Business Problem: Are the scores of those who attend the course and those who do not attend significantly different from each other?"
###################################################

# H0: M1 = M2 (There is not statistically important difference between scores)
# H1: M1 != M2 (There is statistically important difference between scores)

df = pd.read_csv("course_reviews.csv")
df.head()

df[(df["Progress"] > 75)]["Rating"].mean()

df[(df["Progress"] < 25)]["Rating"].mean()


test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


######################################################
# AB Testing (Two-Sample Proportion Test)
######################################################

# H0: p1 = p2
# There is not statistically significant difference between the conversion rate of the new design and the proportion of the old design
# H1: p1 != p2
# There is

successful = np.array([300, 250])
observation = np.array([1000, 1100])

proportions_ztest(count=successful, nobs=observation)

# proportions_ztest method compares and returns p value
# (3.7857863233209255, 0.0001532232957772221)
# p value = 0.00015 < 0.05
# H0 is rejected so there is a statistical correlation between
# two arrays

successful / observation


############################
# Application 1: Is there a statistically significant difference between the survival rates of women and men?
#############################

# H0: p1 = p2 (p1 - p2 = 0)
# there is not a statistically significant difference between the survival rates of women and men
# H1: p1 != p2
# there is

df = sns.load_dataset("titanic")
df.head()

df.loc[df["sex"] == "female", "survived"].mean()

df.loc[df["sex"] == "male", "survived"].mean()

female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


######################################################
# ANOVA (Analysis of Variance)
######################################################

# ANOVA is used to determine whether
# there are significant differences between the means of three or more groups in a dataset.

df = sns.load_dataset("tips")
df.head()

df.groupby("day")["total_bill"].mean()

#  1. Formulate Hypothesis:

# HO: m1 = m2 = m3 = m4
# Means are equal

# H1: at least one is different

# 2. Assumption Check

#  - Normality Assumption
#  - Homogeneity of Variance

# If assumptions are met: one way anova
# If assumptions are not met: kruskal

# H0: Normality Assumption is met

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, 'p-value: %.4f' % pvalue)



test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# 3. Apply hypothesis and  p-value check


df.groupby("day").agg({"total_bill": ["mean", "median"]})


# parametrik one way anova test:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

# Nonparametrik kruskal anova test:
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])

from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.10)
print(tukey.summary())
