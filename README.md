# UFC-Fight-Duration-Analysis

(Titus Bridgwood - TB and Ioana Preoteasa - IP)

Project goals:
The goal of this project was to use Ultimate Fighting Championships (UFC) data from 1993 to 2019 to build a linear regression model on this data to predict a set of useful metrics. 
Our primary target audience is bookkeepers for betting agencies that run bets based on duration of a fight (normally given as over 1.5 or under 1.5). 

Project outcomes:
1. 3 linear regression models with LASSO implementation, trained on polynomial data. 
2. A predictive model  of fight duration that is somewhat better than guessing by average (R2 ~ 0.056)
3. A predictive model of successful significant strike rate that is significantly better than guessing by average (R2 ~ 0.11)
4. A predictive model of successful takedown rate that is somewhat better than guessing by average (R2 ~ 0.095)
5. Two datasets that can be reused by anyone in the future to enhance or extend our results and analysis. There were several more variables we would have liked to try predicting.

We are grateful for the data provided by Rajeev Warrier (https://www.kaggle.com/rajeevw/ufcdata) and scraped from the UFC Fight Stats website (http://ufcstats.com/statistics/events/completed). 

In order to successfully build a linear regression model, we had choose a continuous target variable. We chose 3:
- durations of fight (in seconds)
- percentage of successful significant strikes
- percentage of successful takedowns 

The github repository contains the two datasets used for training and testing the model. time_predictFINAL.pkl is used for predicting duration and post_2001_norm_str_pct.pkl is to be used for predicting the other stats. The reason for this was that there far more non-null values for the duration of the fight than for the other fight metrics, so they were separated to maximize sample size. The second dataset includes several other fight statistic variables (e.g. average number of reversals; average guard passes) so we welcome anyone wanting to build a model on any of those!

Our modelling function is inside the library.py file and uses Python's ScikitLearn library.

Conclusion: although our R squared values for quite low, especially for our first target variable, we are enthusiastic for the potential any exogenous data could bring to this (such as fighter's training regime, home gym, etc... ). 

Labour division: 
IP - dataset cleaning and initial exploratory data analysis; function testing
TB - refactoring modelling code; writing technical notebook and presentation
