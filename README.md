# Prediction of Match Excitement Based on In-Game Statistics
## Introduction
A football match takes at least 90 minutes to watch and whether it is worth spending that much time is an important question in the minds of football lovers. Therefore, football statisticians developed a metric called “Excitement Rating”. Different companies use different methods to calculate that, for example, SmartRatings by Thuuz which was acquired by sports statistics giant Stats Perform in 2020 combines in-game statistics with social media interactions, context, novelty, and 4 different factors (Wikipedia, 2021). The aim of this project is with Machine Learning methods, predicting Excitement Ratings of matches purely based on in-game statistics. In accordance with this purpose, polynomial regression with degree 3 has been used with mean squared error.
## Problem Formulation
In the context of this project, the focus will be on the pre-pandemic football matches as the world is gradually going back to normal. The matches, in other words, data points, that belong to the interval of 2015-2020 will be used. Matches played in 5 major leagues of Europe - England, Germany, France, Spain, Italy- have been selected. In parallel with the interests of this study, the “Football Data : Top 5 Leagues” dataset by Sanjeet Singh Naik will be used (Naik, 2022)
The data points are matches since the football statistics are kept on a match basis. For almost every football match played, there is an unlimited number of different statistics available on the internet. Some of the widely used statistic types are half-time score, match score, home goals, away goals, home team possession, away team possession, to name a few.
In this project feature values are total goal number, total shot number, mean pass success rate, total foul number, possession difference, where all these numbers are integer values. All of those variables are solely based on in-game statistics and they are compatible with the aims of this paper.
As already stated, the label value is the excitement rating which is an integer value between 0 – 10.
## Methods
### Dataset
In the dataset, 12062 data points with 41 different variables are available to use, and there is not any missing data in any fields. One of the reasons for the high number of variables is the same type of variables were recorded for both teams separately. For example, about the total goal number, there are three different variables available: 1. Score 2. Home Team Goals Scored 3. Away Team Goals Scored. One another reason is there are variable types with basic logical relations. An example of this is shots. There are three different statistics available for shots of a team: 1. Shots 2. On Target Shots 3. Off Target Shots. One of the On Target and Off Target shots can be eliminated from the dataset because as long as we know the total shot number and one of the two above-mentioned variables we can calculate another one by using simple logic, a shot is either on target or off target.
It has been known that too many features may lead to computational complexity, which was a probable case for this project. Hence, the need for feature selection was obvious. For that reason, feature values were decided to represent the game in the following aspects: the number of critical points of the match, the balance between sides, and the fluency of the match. In this sense, critical points were mainly represented by Total Goal Number and Total Shot Number, balance between sides was mainly represented by Possession Difference, and the fluency of the match was mainly represented by Total Foul Number and Mean Pass Success. However, those representations are not strict and there are cross-relationships between different features.
None of the feature values mentioned was originally available, therefore some operations were done by using pandas library (Pandas, 2022) Home Team Goals Scored and Away Team Goals Scored summed to obtain Total Goal Number, Home Team Total Shots and Away Team Total Shots summed to obtain Total Shot Number, average of Home Team Pass Success and Away Team Pass Success was taken to obtain Mean Pass Success Rate, Home Team Fouls and Away Team Fouls summed to obtain Total Foul Number and the absolute difference between Home Team Possession and Away Team Possession was taken to obtain Possession Difference.
Although it can be used in all Machine Learning applications, K-fold cross-validation is especially useful when there are comparatively small datasets where the risk of the unlucky single split is relatively high. (Jung, 2022:p.156) In this project, there are 12062 data points, and this number is high enough to use the single split method. As a general rule, the training set is should be bigger than validation and test sets, however, all sets should be large enough to represent reality. For that reason, the data set was split into three as the training set, validation set, and test set with ratios 0.6, 0.2, and 0.2 respectively.
### Polynomial regression model
Analyzing the dataset shows that there isn’t a simple linear relationship between the feature values and the label value. Therefore, it was reasonable to use polynomial regression to make an accurate prediction about the label value. The mathematical ground of polynomial regression with degree k is as follows:
$$𝑦̂ = β_0 + β_2 * 𝑥_1^1 + β_2 𝑥_2^2 + ... + β_n ∗ 𝑥_n^k$$ 
For this method, Mean Squared Error was used, as it is customary to use it with polynomial regression problems, which is a combination of polynomial features with linear mapping. (Jung, 2022:p. 81) The most important characteristic of Mean Squared Error is its rate of punishment increases exponentially. This might be a problem in the cases where the dataset contains outliers, however, that is not the case for the dataset used in this project. Therefore, Mean Squared Error was a reasonable choice. The formula of Mean Square Error is as follows with ŷ representing prediction and y representing actual value:
$$\frac{1}{n} \sum_{i=1}^n (𝑦̂_i - y_i)^2 $$

One of the most critical parts of using polynomial regression is deciding on degree. Since it directly affects the predicted relationship between features and label value, correct degree choice is important. To avoid both underfitting and overfitting, a common model validation method was used. By comparing validation errors of polynomial models with

different degrees from 1 to 10, it was found that degree 3 is the best option for this problem since it yielded the smallest validation error with comparatively small training error.
To implement this problem, scikit-learn’s LinearRegression (scikit-learn, (n.d)a) PolynomialFeatures (scikit-learn, (n.d)b), and mean_squared_error (scikit-learn, (n.d)b) methods were used in addition to basic Python commands.
## Sources
Wikimedia Foundation. (2021, December 15). Thuuz. Wikipedia. Retrieved February 10, 2022, from https://en.wikipedia.org/wiki/Thuuz

Naik, S. S. (2022, January 18). Football data : Top 5 leagues. Kaggle. Retrieved February 10, 2022, from https://www.kaggle.com/sanjeetsinghnai/football-data-top-5-leagues

Pandas. pandas. (March 10, 2022). from https://pandas.pydata.org/

Jung, A. "Machine Learning: The Basics," Springer, Singapore, 2022. Access date March 10, 2022.

Abhigyan. (2020, August 2). Understanding polynomial regression. Medium. Retrieved March 10, 2022, from https://medium.com/analytics-vidhy/understanding-polynomial- regression-5ac25b970e18

Sklearn.linear_model.linearregression. scikit-learn. (n.d)a. Retrieved March 10, 2022, from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

Sklearn.preprocessing.PolynomialFeatures. scikit-learn. (n.d)b. Retrieved March 10, 2022, from
https://scikit- learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

Sklearn.metrics.mean_squared_error. scikit-learn. (n.d.)c. Retrieved March 10, 2022, from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
