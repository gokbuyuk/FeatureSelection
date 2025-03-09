# Automated Feature Selection

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.datasets import load_iris
import unittest
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
import dalex as dx


def prefixed_index(prefix, n, startindex=1):
    '''
    Creates list with indexed strings. For example
    a prefix of 'V' and n=3 creates list
    ['V1', 'V2', 'V3].
    This is useful for creating column names that are
    initially missing when converting from numpy arrays to
    pandas dataframes.

    Parameters:
    prefix(str): String defining the used prefix
    n(int): The number of items to create in result list
    startindex(int): The starting index. Default is '1'
    '''
    result = [prefix] * n
    for i in range(n):
        result[i] = result[i] + str(i+startindex)
    return result

def filter_correlations(
        data,
        target,
        ff_corr_max=0.9, # remove variables with too high correlation between features
        target_corr_max=0.9, # remove columns with too high correlation with target variable
        target_corr_max_p=0.1, # remove columns with too little correlation via P-value
        remove_flagged_columns=True):
    '''
    Checks pairwise correlation, return if it is higher than corr_ths (correlation threshold).
    Issues an assert error in case such a variable pair is found.
    Prints the correlations of both predictor variables with the
    target variable so that can decide to exclude the variable
    that shows _less_ correlation with the target variable.

    Parameters:
    data(pd.DataFrame): A Pandas data frame.
    target(np.ndarray,pd.Series): A numpy array or pandas Series containing the target variable.
    ff_corr_max(float): feature_feature correlation threshold
    target_corr_max_p(float): threshold for correlation P-value with target variable.
        A high P-value indicates low correlation which will be filtered out.
    remove_flagged_columns(bool): If True (default), remove columns that do not pass filter
        results.
    '''
    assert isinstance(data, pd.DataFrame)
    if not data.shape[1] == data.select_dtypes(include=np.number).shape[1]:
            raise ValueError("Not all columns of data frame are numeric")

    nc = len(data.columns)
    cn = data.columns
    assert isinstance(data,pd.DataFrame)
    if isinstance(target,int):
        target = cn[target]
    badcols = []
    corrv = [0] * nc
    reasons = {
        'undefined correlation with target':[],
        'too high correlation with target':[],
        'too low correlation with target':[],
        'too high correlation with variable':[]
        }
    target_correlations = {}
    for i in range(nc):
        xv = data[cn[i]]
        corrv[i], p = pearsonr(xv, target)
        target_correlations[cn[i]] = (corrv[i], p)
        if np.isnan(corrv[i]).any():
            print(f"Warning: {cn[i]} has undefined correlation with target")
            reasons['undefined correlation with target'].append(cn[i])
            badcols.append(i)
        elif abs(corrv[i]) > target_corr_max:
            print(f"Warning: {cn[i]} has too high correlation with target")
            reasons['too high correlation with target'].append(cn[i])
            badcols.append(i)
        elif p > target_corr_max_p:
            print(f"Warning: {cn[i]} has too low correlation with target")
            reasons['too low correlation with target'].append(cn[i])

    for i in range(nc-1):
        if i in badcols:
            continue
        x1 = data[cn[i]]
        for j in range(i+1,nc):
            if j in badcols:
                continue
            x2 = data[cn[j]]
            assert len(x1) == len(x2)
            corr, _ = pearsonr(x1, x2)
            if abs(corr) > ff_corr_max:
                corr1, p1 = pearsonr(target, x1)
                corr2, p2 = pearsonr(target, x2)
                if p1 > p2:
                    badcols.append(i)
                    reasons['too high correlation with variable'].append(cn[i])
                else:
                    badcols.append(j)
                    reasons['too high correlation with variable'].append(cn[j])
    badcolnames = []
    for i in range(len(badcols)):
        badcolnames.append(cn[badcols[i]])
    result = None
    if remove_flagged_columns:
        result = data.drop(columns=badcolnames)
    else:
        result = data.copy(deep=True)

    return {'data':result,
            'removed_columns':badcolnames,
            'reasons':reasons,
            'target_correlations': target_correlations}

class TestFilterCorrelations(unittest.TestCase):

    def test_filter_correlations(self):
        print("Testing filter_correlations with pandas dataframe")
        target_corr_max_p=0.2
        target_corr_max=0.9
        data = pd.DataFrame(data={
            'A':[1,2,3,4,5,6],
            'B':[1,2,3,4,6,6],
            'C':[6,11,4,10,2,1],
            'D':[1,1,1,1,1,1], # constant, should be filtered out
            'E':[-1,1,-1,1,-1,1],  # uncorrelated should be filtered out
            'F':[6,11,4,10,2,1.5], # too similar to 'C'
            'T':[1,2,4,6,8,12]})
        target = np.array([1, 2, 4, 6, 8, 12])
        result = filter_correlations(data, target=target,
            target_corr_max_p=target_corr_max_p,
            target_corr_max=target_corr_max)
        corrs = result['target_correlations']
        dat2 = result['data']
        cn2 = dat2.columns
        for name in cn2:
            r, p = corrs[name]
            print(f"Correlation of {name} with target: {r}, p-value: {p}")
            assert not np.isnan(r).any()
            assert not np.isnan(p).any()
            # assert p <= target_corr_max_p
            assert r <= target_corr_max
        print(result['reasons'])
        print(documentation_from_named_lists(result['reasons']))

def documentation_from_named_lists(
    x,
    prefix='Variables were removed due to the following reasons:',
    sep='\n', delim=','):
    '''
    Auto-generate documentation based on information we obtained
    from calling function filter_correlations

    Parameters:
    x(dict): A dictionary with legible reasons as keys and items
    that fall under this list as reasons.
    prefix(str): A potential prefix to the generated documentation.
    sep(str): A separator between line items, typically newline
    delim(str): A deliminator between listed items, typically comma
    '''
    assert isinstance(x,dict)
    result = prefix + sep
    for key in x.keys():
        lst = x[key]
        if len(lst) > 0:
            result += key + ": " + delim.join(lst) + sep
    return result

def get_estimator(method_estimate='xgb', task_type='regression'):
    '''
    Returns an estimator based on the specified method and task type.

    Parameters:
    method_estimate(str): Method for estimator. Options: 'xgb', 'knn', 'svm', 'rf', 'linear'
    task_type(str): Type of task. Options: 'regression', 'classification'

    Returns:
    estimator: A scikit-learn compatible estimator
    '''
    if task_type not in ['regression', 'classification']:
        raise ValueError("task_type must be 'regression' or 'classification'")

    if method_estimate == 'xgb':
        if task_type == 'regression':
            return xgb.XGBRegressor(
                max_depth=3,
                n_estimators=200,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:absoluteerror',
                importance_type='gain',
                tree_method='hist',
                reg_alpha=0,
                reg_lambda=1
            )
        else:  # classification
            return xgb.XGBClassifier(
                max_depth=3,
                n_estimators=200,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                importance_type='gain',
                tree_method='hist',
                reg_alpha=0,
                reg_lambda=1
            )
    elif method_estimate == 'knn':
        if task_type == 'regression':
            return KNeighborsRegressor(n_neighbors=5)
        else:  # classification
            return KNeighborsClassifier(n_neighbors=5)
    elif method_estimate == 'svm':
        if task_type == 'regression':
            return SVR(kernel='rbf', C=1.0)
        else:  # classification
            return SVC(kernel='rbf', C=1.0, probability=True)
    elif method_estimate == 'rf':
        if task_type == 'regression':
            return RandomForestRegressor(n_estimators=100, max_depth=None)
        else:  # classification
            return RandomForestClassifier(n_estimators=100, max_depth=None)
    elif method_estimate == 'linear':
        if task_type == 'regression':
            return LinearRegression()
        else:  # classification
            return LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"Unknown method_estimate: {method_estimate}. Choose from 'xgb', 'knn', 'svm', 'rf', 'linear'")

def filter_features_dalex(X, y,
        n_features_to_select=None,
        n_jobs=2,
        method_estimate='xgb',
        task_type='regression',
        n_permutations=10,
        random_state=42,
        loss_function=None,
        verbose=True):
    '''
    This function filters a dataframe using dalex permutation importance to identify
    and select the most important features for prediction.

    Parameters:
    X(pandas.DataFrame,numpy.ndarray): Feature data not including outcome variable
    y: Outcome variable
    n_features_to_select(int): Numbers of features to select. If None (default) automatically select features with importance > 0
    n_jobs(int): Number of jobs for parallelization
    method_estimate(str): Method for estimator. Options: 'xgb', 'knn', 'svm', 'rf', 'linear'
    task_type(str): Type of task. Options: 'regression', 'classification'
    n_permutations(int): Number of permutations for each variable
    random_state(int): Random seed for reproducibility
    loss_function(callable): Custom loss function for permutation importance
    verbose(bool): If True provide output during processing
    '''
    columns = []
    if (isinstance(method_estimate, list)):
        if len(method_estimate) > 0:
            method_estimate = method_estimate[0]
        else:
            method_estimate = 'xgb'

    # Determine task type if not specified
    if task_type == 'auto':
        # Check if y contains only integers or is categorical
        if isinstance(y, (pd.Series, np.ndarray)):
            unique_values = np.unique(y)
            if len(unique_values) < 10 and all(isinstance(val, (int, np.integer)) for val in unique_values):
                task_type = 'classification'
            else:
                task_type = 'regression'
        else:
            task_type = 'regression'  # default

    rows, cols = X.shape
    if verbose:
        print(f"Starting filter_features_dalex with {rows} rows and {cols} features (=columns).")
        print(f"Task type: {task_type}, Method: {method_estimate}")

    for i in range(cols):
        columns.append("V" + str(i+1))

    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        if not X.shape[1] == X.select_dtypes(include=np.number).shape[1]:
            raise ValueError("Not all columns of data frame are numeric")
        columns = X.columns
        X_array = X.to_numpy(copy=True)
    else:
        X_array = X
        X = pd.DataFrame(X, columns=columns)

    # Get appropriate estimator based on task type and method
    estimator = get_estimator(method_estimate, task_type)

    # Train the model on the full dataset
    estimator.fit(X_array, y)

    # Create a dalex explainer
    if task_type == 'regression':
        explainer = dx.Explainer(estimator, X, y, label=f"{method_estimate} model")
    else:  # classification
        explainer = dx.Explainer(estimator, X, y, label=f"{method_estimate} model", model_type="classification")

    # Calculate permutation importance using dalex
    # Use 'permutational' instead of 'permutation'
    perm_importance = explainer.model_parts(
        type='permutational',  # Corrected parameter value
        N=n_permutations,
        B=1,  # No bootstrap
        random_state=random_state,
        keep_raw_permutations=False,
        processes=n_jobs
    )

    # Get feature importances
    importance_df = perm_importance.result

    # Filter to get only variable importances (not the _baseline_ and _full_model_)
    importance_df = importance_df[~importance_df['variable'].isin(['_baseline_', '_full_model_'])]

    # Sort by dropout_loss (higher means more important)
    importance_df = importance_df.sort_values('dropout_loss', ascending=False)

    # Get feature names and their importances
    feature_names = importance_df['variable'].tolist()
    importances = importance_df['dropout_loss'].values

    # Determine which features to select
    if n_features_to_select is None:
        # Select features with positive importance (greater than baseline)
        selected_features = importance_df[importance_df['dropout_loss'] > 0]['variable'].tolist()
        if len(selected_features) == 0:  # If no features have importance > 0
            # Select the top 2 features or all if less than 2
            n_to_select = min(2, len(feature_names))
            selected_features = feature_names[:n_to_select]
    else:
        # Select the top n_features_to_select
        n_to_select = min(n_features_to_select, len(feature_names))
        selected_features = feature_names[:n_to_select]

    # Create support mask
    support = np.zeros(cols, dtype=bool)
    for feature in selected_features:
        idx = list(columns).index(feature)
        support[idx] = True

    # Transform X to include only selected features
    if is_dataframe:
        X_selected = X[selected_features]
    else:
        selected_indices = [list(columns).index(feature) for feature in selected_features]
        X_selected = X_array[:, selected_indices]
        X_selected = pd.DataFrame(X_selected, columns=selected_features)

    if verbose:
        print("Number of features remaining after feature selection step:", len(selected_features))
        print("Selected features:", selected_features)
        print("Feature importances:")
        for i, feature in enumerate(feature_names):
            print(f"{feature}: {importances[i]:.6f}")

    return {
        'data': X_selected,
        'column_flags': support,
        'new_columns': selected_features,
        'importances': dict(zip(feature_names, importances)),
        'importance_df': importance_df
    }

class TestFilterFeatures(unittest.TestCase):

    def test_filter_features_pandas(self):
        '''
        Test filter_features_dalex using simple
        example from a regression dataset.
        '''
        print("Testing filter_features_dalex with pandas dataframe")
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X)
        y = np.random.rand(len(X))  # Generate random continuous target variable
        result = filter_features_dalex(X, y)
        X2 = result['data']
        assert X2.shape[1] > 0

    def test_filter_features_classification(self):
        '''
        Test filter_features_dalex using simple
        example from a classification dataset.
        '''
        print("Testing filter_features_dalex with classification task")
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X)
        result = filter_features_dalex(X, y, task_type='classification', method_estimate='rf')
        X2 = result['data']
        assert X2.shape[1] > 0

 
def test_filter_features_pandas_large(self,
        method_estimate='xgb',
        task_type='regression',
        extracols=70, extrarows=100, n_jobs=2):
    '''
    Example that tests run times for large number of
    rows (samples) and columns (features).
    One can have more challenging tests for
    extracols > 200, extrarows > 1000.
    '''
    print("Testing filter_features_dalex with large number of rows and columns")
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X, columns=['I1', 'I2', 'I3', 'I4'])
    X = X[:100]

    if task_type == 'regression':
        y = np.random.rand(len(X))  # Generate random continuous target variable
    else:  # classification
        y = y[:100]  # Use original classification labels

    rows, cols = X.shape
    if extracols > 0:
        A = np.random.normal(0, 1, (rows, extracols))
        A = pd.DataFrame(data=A, columns=prefixed_index('V', n=extracols))
        assert not pd.Series(A.columns).duplicated().any()
        X = pd.concat([X, A], axis=1)  # add more columns
    if extrarows > 0:
        A2 = np.random.normal(0, 1, (extrarows, len(X.columns)))
        A2 = pd.DataFrame(data=A2, columns=X.columns)
        X = pd.concat([X, A2], axis=0)  # add more rows
        if task_type == 'regression':
            y = np.concatenate([y, np.random.rand(extrarows)])  # Extend target variable
        else:  # classification
            y = np.concatenate([y, np.random.choice(np.unique(y), size=extrarows)])  # Extend target variable
    X.index = range(len(X))
    assert len(y) == X.shape[0]  # number of rows must match
    assert not pd.Series(X.columns).duplicated().any()

    # Force feature selection to select fewer features
    n_features_to_select = max(1, X.shape[1] // 2)  # Select half the features or at least 1

    result = filter_features_dalex(X, y, n_jobs=n_jobs,
                             method_estimate=method_estimate,
                             task_type=task_type,
                             n_features_to_select=n_features_to_select)
    X2 = result['data']
    newcols = result['new_columns']
    assert not pd.Series(newcols).duplicated().any()

    # Check that we have the expected number of features after selection
    assert len(newcols) == n_features_to_select
    assert len(newcols) > 0, "Feature selection should return at least one feature"

    def test_filter_features(self):
        '''
        Testing filter_features_dalex using a regression dataset.
        '''
        print("Testing filter_features_dalex with numpy array")
        X, y = load_iris(return_X_y=True)
        y = np.random.rand(len(X))  # Generate random continuous target variable
        X2 = filter_features_dalex(X, y)['data']
        assert X2.shape[1] > 0


def AutoFeatures(data, target,
    method_estimate='xgb',
    task_type='auto',
    ff_corr_max=0.9,
    target_corr_max=0.9,
    target_corr_max_p=0.9,
    n_features_to_select=None,
    n_permutations=10,
    random_state=42,
    n_jobs=1):
    '''
    This function encapsulates the pipeline for feature selection.
    First, features are eliminated based on correlations with
    other features or the target variable. Next, features are
    eliminated based on dalex permutation importance.

    Parameters:
    data: A Pandas dataframe containing feature data.
    target: A numpy array or pandas Series containing the target variable.
    method_estimate: Method for estimator. Options: 'xgb' (default), 'knn', 'svm', 'rf', 'linear'
    task_type: Type of task. Options: 'auto' (default), 'regression', 'classification'
    ff_corr_max: Maximum correlation between features
    target_corr_max: Maximum correlation with target variable
    target_corr_max_p: Maximum correlation P-value (effectively minimal correlation) with target variable.
    n_features_to_select: Number of features to select. If None (default) automatically find best set of features
    n_permutations: Number of permutations for each variable
    random_state: Random seed for reproducibility
    n_jobs: Number of jobs for parallelization

    Returns(list): Returns a list with following elements:
    'data': A dataframe with reduced number of features
    'new_columns': List of column names of returned dataframe
    'column_flags': List with boolean logical values corresponding to
     which columns from the input data frame to keep.
    'importances': Feature importance values from permutation importance
    'importance_df': DataFrame with detailed importance information
    '''
    # print("Filtering features with AutoFeatures...")
    if not isinstance(target, (np.ndarray, pd.Series)):
        raise ValueError("Target has to be a numpy array or pandas Series")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data has to be a pandas DataFrame")

    # Determine task type if set to auto
    if task_type == 'auto':
        # Check if target contains only integers or is categorical
        unique_values = np.unique(target)
        if len(unique_values) < 10 and all(isinstance(val, (int, np.integer)) for val in unique_values):
            task_type = 'classification'
            print(f"Automatically detected task type: classification (found {len(unique_values)} unique classes)")
        else:
            task_type = 'regression'
            print(f"Automatically detected task type: regression")

    # Step 1: Filter based on correlations
    result = filter_correlations(data, target=target,
        ff_corr_max=ff_corr_max,
        target_corr_max_p=target_corr_max_p,
        target_corr_max=target_corr_max)
    data2 = result['data']
    assert data2.shape[1] > 0

    # Step 2: Filter based on dalex permutation importance
    result2 = filter_features_dalex(data2, target,
        method_estimate=method_estimate,
        task_type=task_type,
        n_features_to_select=n_features_to_select,
        n_permutations=n_permutations,
        random_state=random_state,
        n_jobs=n_jobs)

    return result2


class TestAutoFeatures(unittest.TestCase):

    def test_AutoFeatures(self):
        '''
        Tests AutoFeatures method using a simple
        synthetic dataset. Checks if the method
        detects and removes columns with too high
        or too low correlation.
        '''
        print("Testing AutoFeatures with pandas dataframe")
        target_corr_max_p=0.2
        target_corr_max=0.9
        data = pd.DataFrame(data={
            'A':[1,2,3,4,5,6],
            'B':[1,2,3,4,6,6],
            'C':[6,11,4,10,2,1],
            'D':[1,1,1,1,1,1], # constant, should be filtered out
            'E':[-1,1,-1,1,-1,1],  # uncorrelated should be filtered out
            'F':[6,11,4,10,2,1.5], # too similar to 'C'
            'G':[1.2,0.3, 4.7, 2.3, 0.5, 0.9],
            'H':[2.5,8.4, 5.3, -2.4,-1.5,0.0]})
        target = np.array([0, 1, 0, 1, 0, 1])
        result = AutoFeatures(data, target=target, task_type='classification')
        print('Selected features: ')
        print(result['new_columns'])
        assert len(result['new_columns'])>0

    def test_AutoFeatures_iris(self):
        '''
        Tests using AutoFeatures function a truncated
        Iris dataset.
        '''
        print("Testing AutoFeatures with truncated Iris dataset")
        X, y = load_iris(return_X_y=True)
        X = X[:99,:]
        y = np.random.rand(len(X))  # Generate random continuous target variable
        X = pd.DataFrame(X)
        result = AutoFeatures(X, target=y)
        X2 = result['data']
        assert X2.shape[1] > 0

    def test_AutoFeatures_iris_classification(self):
        '''
        Tests using AutoFeatures function with Iris dataset for classification.
        '''
        print("Testing AutoFeatures with Iris dataset for classification")
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X)
        result = AutoFeatures(X, target=y, task_type='classification', method_estimate='rf')
        X2 = result['data']
        print(f"Selected features for classification: {result['new_columns']}")
        assert X2.shape[1] > 0

    def test_AutoFeatures_iris_nfeatures(self):
        '''
        Tests using AutoFeatures function a truncated
        Iris dataset.
        '''
        print("Testing AutoFeatures with truncated Iris dataset and n_features_to_select=2")
        X, y = load_iris(return_X_y=True)
        X = X[:99,:]
        y = np.random.rand(len(X))  # Generate random continuous target variable
        X = pd.DataFrame(X)
        result = AutoFeatures(X, target=y,
            n_features_to_select=2)
        X2 = result['data']
        assert X2.shape[1] == 2


if __name__ == '__main__':
    unittest.main()
