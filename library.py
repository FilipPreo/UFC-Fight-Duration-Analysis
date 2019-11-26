### SCALING ###

from sklearn.preprocessing import StandardScaler

def scale_features(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled

### Modelling ###


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsIC
    

def modeling(x_train, x_test, y_train, y_test, poly_order=1, criterion='aic', iterations=1000, lars_ic=False, lasso_alpha=None):
    
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
        
        print(f'The R-2 for a LASSO Least Angle Regression model with with a Polynomial Order of {poly_order} is {score}.\n
        The model with the lowest AIC of {aic_score} has a LASSO alpha of {optimal_alpha}')
        
        return lars_poly
    
    elif not lars_ic:
        
        lasso_reg = Lasso(
            alpha=alpha, 
            normalize=False, 
            max_iter=iterations, 
            random_sate=42
        )
        fit = lasso_reg.fit(x_poly_train, y_train)
        score = lasso_reg.score()
        
        print(f'The R-2 for a model with with a Polynomial Order of {poly_order} and a Lasso Alpha of {lasso_alpha} is {score}.\n')
        
        return lasso_reg


### Cross Validation ###
class CrossValidation(self, n_splits, model, x_train, y_train):
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