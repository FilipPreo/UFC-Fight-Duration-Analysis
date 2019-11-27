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

def modeling(x_train, x_test, 
             y_train, y_test, 
             poly_order=1, criterion='bic', 
             iterations=1000, lars_ic=False, 
             lasso_alpha=None, kfold=True, 
             k_n_splits=2, k_scoring ='r2', 
             var_name = None, scale=True):
    """
    Function that produces a tuple of linear regression results plus train and test data. Assumes that
    data has been split already. Default arguments will return a model trained on scaled data using 
    LASSO linear regression with K fold cross validation. 
    x_train - input variables for model training (expects pandas Series/DataFrame)
    x_test -  input variables for model testing (expects pandas Series/DataFrame)
    y_train - target variables for model training (expects pandas Series)
    y_train - target variables for model training (expects pandas Series)
    poly_order - order of polynomial transform to be applied to x_train and x_test
    criterion - which information criterion will be used to compute best model; default is BIC
    iterations - number of iterations for minimizing cost function
    lars_ic - (bool) whether the Sklearn LassoLars return an information criterion and uses it to 
    determine the optimal alpha
    kfold - (bool) whether to use the Sklearn KFold object to cross validate training data
    k_n_splits - how many k splits to use when doing KFold validation
    k_scoring - what metric of model fit the KFold object should return, default is R2
    var_name - what name the variable being tested will have in the pandas DataFrame produced. If None, 
    will default to the name of the y_test series
    scaling - (bool) whether to scale the data or not; uses StandardScaler. 
    
    Function returns a tuple of objects. Printout for every option indicates what the respective 
    indices are for accessing different items. 
    """
    
    if var_name==None:
        var_name=f'{y_test.name[0:4]}_polyO{str(poly_order)}_{k_n_splits}ksplits'
        if iterations!=1000:
            var_name=f'{y_test.name[0:4]}_polyO{str(poly_order)}_{k_n_splits}ksplits_iter{iterations}'
    
    # Using scaling function to scale features
    if scale:
        x_train_scaled, x_test_scaled = scale_features(x_train, x_test)
    else:
        x_train_scaled, x_test_scaled = x_train, x_test
        
        
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
        ic_score = np.mean(lars_poly.criterion_)
        optimal_alpha = lars_poly.alpha_
        
        if kfold:
            
            crossval = KFold(n_splits=k_n_splits, shuffle=True, random_state=42)
            cvs = cross_val_score(lars_poly, x_poly_train, y_train, scoring=k_scoring, cv=crossval)
            cvs_mean_score = np.mean(cvs)

            print(f'''The R-2 for a LASSO Least Angle Regression model with with a Polynomial Order of {poly_order} is {score}.\n The model with the lowest {criterion} of {ic_score} has a LASSO alpha of {optimal_alpha} \n Function returns a tuple indexed as follows: \n 0 - Sklearn lasso-regression object  \n  1 - training X data (np array) \n 2 - testing X data (np array)  \n 3  -  Model results table (pandas DataFrame obj) \n  4  -  training Y data (np array)  \n  5  -  testing Y data (np array)''')
            
            return lars_poly, x_poly_train, x_poly_test, pd.DataFrame(data=[[score, ic_score, optimal_alpha, cvs_mean_score]], columns=['R2',f'{criterion}','Optimal_alpha', 'Mean_cvs'], index=[var_name]),  y_train, y_test
            
        else:    
            print(f'''The R-2 for a LASSO Least Angle Regression model with with a Polynomial Order of {poly_order} is {score}.\n The model with the lowest {criterion} of {ic_score} has a LASSO alpha of {optimal_alpha}\n Function returns a tuple indexed as follows: \n 0 - Sklearn lasso-regression object  \n  1 - training X data (np array) \n 2 - testing X data (np array)  \n 3  -  Model results table (pandas DataFrame obj) \n  4  -  training Y data (np array)  \n  5  -  testing Y data (np array) ''')
           
            return lars_poly, x_poly_train, x_poly_test, pd.DataFrame(data=[[score, ic_score, optimal_alpha]], columns=['R2', f'{criterion}','Optimal_alpha'], index=[var_name]), y_train, y_test
        
         
                
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
            
            print(f'''The R-2 for a model with with a Polynomial Order of {poly_order} and a Lasso Alpha of {lasso_alpha} is {np.round(score,4)}.\n  Function returns a tuple indexed as follows:  \n  0 - Sklearn lasso-regression object  \n  1 - training X data (np array) \n 2 - testing X data (np array) \n   3  -  Model results table (pandas DataFrame obj)  \n  4  -  training Y data (np array)  \n  5  -  testing Y data (np array) ''')
            
            return lasso_reg, x_poly_train, x_poly_test, pd.DataFrame(data=[[score, None, None, cvs_mean_score]], columns=['R2',f'{criterion}','Optimal_alpha', 'Mean_cvs'], index=[var_name]),  y_train, y_test
            
        else:
        
            print(f'''The R-2 for a model with with a Polynomial Order of {poly_order} and a Lasso Alpha of {lasso_alpha} is {np.round(score,4)}.\n  Function returns a tuple indexed as follows:  \n  0 - Sklearn lasso-regression object  \n  1 - training X data (np array) \n 2 - testing X data (np array) \n  3  -  Model results table (pandas DataFrame obj)  \n  4  -  training Y data (np array)  \n  5  -  testing Y data (np array) ''')
            
            return lasso_reg, x_poly_train, x_poly_test, pd.DataFrame(data=[[score, None, None, None]], columns=['R2',f'{criterion}','Optimal_alpha', 'Mean_cvs'], index=[var_name]), y_train, y_test

