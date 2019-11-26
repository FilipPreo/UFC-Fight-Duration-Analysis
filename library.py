import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LassoLarsIC
    


### SCALING ###


def scale_features(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled

### Modelling ###



def modeling(x_train, x_test, y_train, y_test, poly_order=1, criterion='aic', iterations=1000, lars_ic=False, lasso_alpha=None, kfold=True, k_n_splits=2, k_scoring ='r2', var_name = None):
    """
    
    
    Function takes in 2 pandas DataFrames and 2 Series. 
    """
    if var_name==None:
        var_name=f'{y_test.name[0:4]}_polyO{str(poly_order)}_{k_n_splits}ksplits'
        if iterations!=1000:
            var_name=f'{y_test.name[0:4]}_polyO{str(poly_order)}_{k_n_splits}ksplits_iter{iterations}'
    
    # Using scaling function to scale features
    x_train_scaled, x_test_scaled = scale_features(x_train, x_test)
    
    # Producing Polynomial Features (1 being linear regression)
    poly = PolynomialFeatures(poly_order)
    x_poly_train = poly.fit_transform(x_train_scaled)
    x_poly_test = poly.transform(x_test_scaled)
    
    if lars_ic:
        
        lars_poly = LassoLarsIC(
            criterion=criterion, 
            fit_intercept=True, 
            normalize=False,
            max_iter=iterations,
            
        )
        fit = lars_poly.fit(x_poly_train, y_train)
        score = lars_poly.score(x_poly_test, y_test)
        aic_score = np.mean(lars_poly.criterion_)
        optimal_alpha = lars_poly.alpha_
        
        if kfold:
            
            crossval = KFold(n_splits=k_n_splits, shuffle=True, random_state=42)
            cvs = cross_val_score(lars_poly, x_poly_train, y_train, scoring=k_scoring, cv=crossval)
            cvs_mean_score = np.mean(cvs)

            print(f'''The R-2 for a LASSO Least Angle Regression model with with a Polynomial Order of {poly_order} is {score}.\n The model with the lowest AIC of {aic_score} has a LASSO alpha of {optimal_alpha} \n Function returns a tuple indexed as follows: \n 0 - Sklearn lasso-regression object  \n  1 - training X data (np array) \n 2 - testing X data (np array)  \n 3  -  Model results table (pandas DataFrame obj \n  4  -  training Y data (np array)  \n  5  -  testing Y data (np array)''')
            
            return lars_poly, x_poly_train, x_poly_test, pd.DataFrame(data=[[score, aic_score, optimal_alpha, cvs_mean_score]], columns=['R2','AIC','Optimal_alpha', 'Mean_cvs'], index=[var_name]),  y_train, y_test
            
        else:    
            print(f'''The R-2 for a LASSO Least Angle Regression model with with a Polynomial Order of {poly_order} is {score}.\n The model with the lowest AIC of {aic_score} has a LASSO alpha of {optimal_alpha}\n Function returns a tuple indexed as follows: \n 0 - Sklearn lasso-regression object  \n  1 - training X data (np array) \n 2 - testing X data (np array)  \n 3  -  Model results table (pandas DataFrame obj \n  4  -  training Y data (np array)  \n  5  -  testing Y data (np array) ''')
           
            return lars_poly, x_poly_train, x_poly_test, pd.DataFrame(data=[[score, aic_score, optimal_alpha]], columns=['R2','AIC','Optimal_alpha'], index=[var_name]), y_train, y_test
        
         
                
    elif not lars_ic:
        
        lasso_reg = Lasso(
            alpha=lasso_alpha, 
            normalize=False, 
            max_iter=iterations, 
            random_state=42
        )
        fit = lasso_reg.fit(x_poly_train, y_train)
        score = lasso_reg.score(x_poly_test, y_test)
        
        if kfold:
            
            crossval = KFold(n_splits=k_n_splits, shuffle=True, random_state=42)
            cvs = cross_val_score(lasso_reg, x_poly_train, y_train, scoring=k_scoring, cv=crossval)
            cvs_mean_score = np.mean(cvs)
            
            print(f'''The R-2 for a model with with a Polynomial Order of {poly_order} and a Lasso Alpha of {lasso_alpha} is {np.round(score,4)}.\n  Function returns a tuple indexed as follows:  \n  0 - Sklearn lasso-regression object  \n  1 - training X data (np array) \n 2 - testing X data (np array) \n   3  -  Model results table (pandas DataFrame obj  \n  4  -  training Y data (np array)  \n  5  -  testing Y data (np array) ''')
            
            return lasso_reg, score, x_poly_train, x_poly_test, pd.DataFrame(data=[[score, cvs_mean_score]], columns=['R2','Mean_cvs'], index=[var_name]),  y_train, y_test
            
        else:
        
            print(f'''The R-2 for a model with with a Polynomial Order of {poly_order} and a Lasso Alpha of {lasso_alpha} is {np.round(score,4)}.\n  Function returns a tuple indexed as follows:  \n  0 - Sklearn lasso-regression object  \n  1 - training X data (np array) \n 2 - testing X data (np array) \n  3  -  training Y data (np array)  \n  4  -  testing Y data (np array)''')
            
            return lasso_reg, score, x_poly_train, x_poly_test, y_train, y_test


### Cross Validation ###
class CrossValidation:
    def __init__(self, n_splits, model, x_train, y_train):
        self.crossval = KFold(n_splits, shuffle=True, random_state=42)
        self.scores = cross_val_score(reg_poly, x_poly_train, y_train, scoring='r2', cv=crossval)
        self.mean_score = np.mean(scores)
        
    def print_nice(self):
        print(f'A cross validation with {n_splits} splits' )
        
        
# Make DF like Joe #


def LassoCV_find_best_target(x_df, y_df, n_splits=10, poly_order=1, lasso_tol=0.0001, max_iterations=1000, regression_type='ols'):
    """Lasso """
    scaler= StandardScaler()
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=.2, random_state=42)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    poly = PolynomialFeatures(poly_order)
    x_train = poly.fit_transform(x_train)
    x_test = poly.transform(x_test)
    
    if regression_type == 'lasso':
        if isinstance(y_df, pd.DataFrame):
            reg_poly = LassoCV(cv = 5, tol=lasso_tol,max_iter=max_iterations)
            fit = reg_poly.fit(x_train, y_train)
            score_train = reg_poly.score(x_train, y_train)
            score_test = reg_poly.score(x_test, y_test)
            crossval = KFold(n_splits, shuffle=True, random_state=42)
            cvs = cross_val_score(reg_poly, x_train, y_train, scoring='r2', cv=crossval)
            cvs_mean_score = np.mean(cvs)
            print(f'''R2 for training data fitting to variable: {y_df.name} is {score_train}. \n
            CVS = {cvs_mean_score} . \n
            R2 for testing data fitting to variable: {y_df.name} is {score_test} \n.''')
    return reg_poly


def find_best_target(x_df, y_df, lasso_lambda, n_splits=10, poly_order=2, lasso_tol=0.0001, max_iterations=10000):
    scaler= StandardScaler()
    for col in y_df.columns:
        x_train, x_test, y_train, y_test = train_test_split(x_df, y_df[col], test_size=.2, random_state=42)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        poly = PolynomialFeatures(poly_order)
        x_train = poly.fit_transform(x_train)
        x_test = poly.transform(x_test)
        reg_poly = Lasso(alpha=lasso_lambda,tol=lasso_tol,max_iter=max_iterations)
        fit = reg_poly.fit(x_train, y_train)
        score_train = reg_poly.score(x_train, y_train)
        score_test = reg_poly.score(x_test, y_test)
        crossval = KFold(n_splits, shuffle=True, random_state=42)
        cvs = cross_val_score(reg_poly, x_train, y_train, scoring='r2', cv=crossval)
        cvs_mean_score = np.mean(cvs)
        print(f'''R2 for training data fitting to variable: {col} is {score_train}. \n
              CVS = {cvs_mean_score} . \n
              R2 for testing data fitting to variable: {col} is {score_test} \n.''')
        return reg_poly