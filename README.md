# UFC-Fight-Duration-Analysis

(Titus Bridgwood - TB and Ioana Preoteasa - IP)

Project goals:
How long will a UFC fight last? Will it go over- or under- 1.5 rounds? What will be the rate of successful strikes? What about takedowns? Can these questions be answered using only data available before the fight?

The goal of this project was to use Ultimate Fighting Championships (UFC) data from 1993 to 2019 to build a linear regression model on this data to predict a set of useful metrics. 
Our primary target audience is bookkeepers for betting agencies that run bets based on duration of a fight (normally given as over 1.5 or under 1.5). We hoped to be able to provide robust predictions. 

Project outcomes: Three linear regression models with LASSO implementation, trained on polynomial data. 
2. A model  of fight duration that is somewhat better than guessing by average (R2 ~ 0.056)
3. A model of successful significant strike rate that is significantly better than guessing by average (R2 ~ 0.11)
4. A model of successful takedown rate that is somewhat better than guessing by average (R2 ~ 0.095)
5. Two datasets that can be reused by anyone in the future to enhance or extend our results and analysis. There were several more variables we would have liked to try predicting.

In order to successfully build a linear regression model, we had to choose continuous target variables. We chose 3:
- durations of fight (in seconds);
- percentage of successful significant strikes;
- percentage of successful takedowns ;

Which pre-match fighter stats are most significant in predicting our metrics?
   1. Duration (please note the models are not very strongly predictive): 
    - Fighter's longest win streak and their total time fought so far (seconds) in their career;
    - Total time fought so far and number of wins by submission;
    - total time fought so far and their height;

   2. Percentage significant strikes:
    - the square of the height (cm);
    - reach (cm)
    
   3. Percentage successful takedowns:
    - Longest win streak and total time fought so far;
    - Total time fought so far and number of wins by submission;
    - Total time fought so far and height (cm);
    
    
We are grateful for the data provided by Rajeev Warrier (https://www.kaggle.com/rajeevw/ufcdata) and scraped from the UFC Fight Stats website (http://ufcstats.com/statistics/events/completed). 



Files in repository:
library.py - contains our modelling function and uses Python's ScikitLearn library;
time_predictFINAL.pkl - dataset used for predicting fight duration;
post_2001_norm_str_pct.pkl - dataset used for predicting the other stats, such as percentage successful significant strikes and takedowns*; 


Conclusion: although our R squared values are very low, especially for our first target variable, we are enthusiastic for the potential any exogenous data could bring to this (such as fighter's training regime, home gym, etc... ). 

Labour division: 
IP - dataset cleaning and initial exploratory data analysis; function testing; creating visualizations.
TB - refactoring modelling code; writing technical notebook and presentation
(both) - iterative building of our LASSO model, analysis of final results.


*(different dataframes were used because there far more non-null values for the duration of the fight than for the other fight metrics, so they were separated to maximize sample size). The second dataset includes several other fight statistic variables (e.g. average number of reversals; average guard passes) so we welcome anyone wanting to build a model on any of those!
