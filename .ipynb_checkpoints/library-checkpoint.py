### SCALING ###

from sklearn.preprocessing import StandardScaler

def scale_features(x_test, x_train):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled

### Modelling ###


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
    

def poly_lasso(x_train, x_test, y_train, y_test, poly_order, lasso_alpha):

    
    poly = PolynomialFeatures(poly_order)
    x_poly_train = poly.fit_transform(x_train_scaled)
    x_poly_test = poly.transform(x_test_scaled)
    
    reg_poly = Lasso(alpha=lasso_alpha)
    fit = reg_poly.fit(x_poly_train, y_train)
    score = reg_poly.score(x_poly_test, y_test)
    
    return f'The R-2 for a model with with a Polynomial Order of {poly_order} and a Lasso Alpha of {lasso_alpha} is {score}.'



### Cross Validation ###
class CrossValidation(self, n_splits, model, x_train, y_train):
    def __init__(self, n_splits, model, x_train, y_train):
        self.crossval = KFold(n_splits, shuffle=True, random_state=42)
        self.scores = cross_val_score(reg_poly, x_poly_train, y_train, scoring='r2', cv=crossval)
        self.mean_score = np.mean(scores)
        
    def print_nice(self):
        print(f'A cross validation with {n_splits} splits' )
        
        
# Make DF like Joe #