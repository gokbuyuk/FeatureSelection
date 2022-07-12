# Automated Feature Selection
#
# This script was previously called ACL_useful_functions_Gokcen.py

# Usage:
# df = AutoClean(df)  # from AutoClean package
# df = AutoFeatures(df, target='Disposition')
# For more information look at example code in functions
# test_AutoClean
# test_AutoClean_iris
#
# Version history
# 0.1.0 2022-07-08 EB
#  * initial version that filters by correlation and xgboost feature elimination
# 0.1.2 2022-07-08 EB
#  * Faster version using recursive feature elimination
#  * added other estimators, now 3 available: Knn, SVM, XGBoost
# 0.1.3 2022-07-08 EB
#  * minor change of added eval_metric in xgb which made warning disappear
#  * added docstrings
# 0.1.4 2022-07-09 EB
#  * implemented function AutoFeaturesEnsemble
#  * Changed "assert" to "raise" statements
#  * Improved speed via xgboost parameters n_estimates and tree_depth
# 0.2.0 2022-07-11 EB
#  * added option n_features_to_select to restrict number of returned features

import pandas as pd
import numpy as np
import dalex as dx
import xgboost as xgb

from scipy.stats import pearsonr
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

# https://dalex.drwhy.ai/python/api/#dalex.Explainer.model_parts

# features = []
# exp = dx.Explainer(model, X_train[features], y_train)
# vi = exp.model_parts(loss_function='1-auc')
# order_features_model1 = vi.result.sort_values(by='dropout_loss', ascending=False)

# feature_imp = order_features_model1[order_features_model1['variable'].isin(features)]
# for feature in features:
#     temp_df = Xy_train[[feature, target]].dropna()
#     r, p = stats.pearsonr(temp_df[feature], temp_df[target])
#     feature_imp.loc[feature_imp['variable']==feature, 'Direction'] = np.where( r< 0,  
#     'Negative', 'Positive')
    
# sns.barplot(data=feature_imp, y='variable', x='dropout_loss', hue='Direction', dodge=False)

def prefixed_index(prefix,n,startindex=1):
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
    assert target in data.columns
    badcols = []
    corrv = [0] * nc
    reasons = {
        'undefined correlation with target':[],
        'too high correlation with target':[],
        'too low correlation with target':[],
        'too high correlation with variable':[]
        }
    target_correlations = {}
    targetv = data.loc[:,target]
    for i in range(nc):
        xv = data[cn[i]]
        corrv[i], p = pearsonr(xv, targetv)
        target_correlations[cn[i]] = (corrv[i], p)
        if cn[i] == target:
            continue
        if np.isnan(corrv[i]).any():
            reasons['undefined correlation with target'].append(cn[i])
            badcols.append(i)
        elif abs(corrv[i]) > target_corr_max:
            reasons['too high correlation with target'].append(cn[i])
            badcols.append(i)
        elif p > target_corr_max_p:
            reasons['too low correlation with target'].append(cn[i])
            badcols.append(i)
            
    for i in range(nc-1):
        if cn[i] == target:
            continue
        if i in badcols:
            continue
        x1 = data[cn[i]]
        for j in range(i+1,nc):
            if cn[j] == target:
                continue
            if j in badcols:
                continue
            x2 = data[cn[j]]
            assert len(x1) == len(x2)
            corr, _ = pearsonr(x1, x2)
            if abs(corr) > ff_corr_max:
                corr1, p1 = pearsonr(targetv, x1)
                corr2, p2 = pearsonr(targetv, x2)
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

def test_filter_correlations():
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
    result = filter_correlations(data,target='T',
        target_corr_max_p=target_corr_max_p,
        target_corr_max=target_corr_max)
    corrs = result['target_correlations']
    dat2 = result['data']
    cn2 = dat2.columns
    for name in cn2:
        if name == 'T':
            continue
        r, p = corrs[name]
        assert not np.isnan(r).any()
        assert not np.isnan(p).any()
        assert p <= target_corr_max_p
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


def filter_features(X, y,
        cv=None,
        n_features_to_select=None,
        n_jobs=2,
        method_estimate = ['xgb', 'knn', 'svm'],
        method_select = ['recursive', 'sequential'],
        verbose=True):
    '''
    This function filters a dataframe so that columns
    not helpful for prediction performance are removed.

    Parameters:
    X(pandas.DataFrame,numpyp.ndarray): Feature data not including outcome variable
    y: Outcome variable
    cv(int,str): describes cross-validation method (see SequentialFeatureSelector)
    n_features_to_select(int): Numbers of features to select. If None (default) automatically find best set of features
    n_jobs(int): Number of jobs for parallelization
    method(str): Method for classifier. Currently defined are 'knn' (default),'xgb' and 'svm'
    verbose(bool): If True provide output during processing
    '''
    columns = []
    if (isinstance(method_estimate,list)):
        if len(method_estimate) > 0:
            method_estimate = method_estimate[0]
        else:
            method_estimate = 'xgb'
    if (isinstance(method_select,list)):
        if len(method_select) > 0:
            method_select = method_select[0]
        else:
            method_select = 'recursive'
    rows, cols = X.shape
    if verbose:
        print(f"Starting filter_features with {rows} rows and {cols} features (=columns).")
    for i in range(cols):
        columns.append("V" + str(i+1))
    if isinstance(X, pd.DataFrame):
        if not X.shape[1] == X.select_dtypes(include=np.number).shape[1]:
            raise ValueError("Not all columns of data frame are numeric")
        columns = X.columns
        X = X.to_numpy(copy=True)
    knn = KNeighborsClassifier(n_neighbors=11)
    # scale_pos_weight = np.sqrt(scale)
    gb0 = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        tree_method = "auto",
        max_depth=3,
        n_estimators=50)
    svr = SVR(kernel="linear")
    classifier = None
    if method_estimate == 'knn':
        classifier = knn
    elif method_estimate == 'xgb':
        classifier = gb0
    elif method_estimate == 'svm':
        classifier = svr
    else:
        raise ValueError("Unknown method specified, currently defined are knn and xgb")
    fs = None
    if method_select == 'sequential':
        fs = SequentialFeatureSelector(classifier,
        n_features_to_select=n_features_to_select, cv=cv,
        n_jobs=n_jobs)
    elif method_select == 'recursive':
        fs = RFE(classifier,
            n_features_to_select=n_features_to_select)
    # print(X)
    fs.fit(X, y)
    # SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=3),
    #                      n_features_to_select=3)
    # print('keeping these columns:', sfs.get_support())
    X2 = fs.transform(X)
    np.array(columns)
    support = fs.get_support()
    assert not pd.Series(columns).duplicated().any()
    newcols = list(np.array(columns)[fs.get_support()])
    X2 = pd.DataFrame(X2, columns=newcols)
    assert not pd.Series(newcols).duplicated().any()
    print("Number of features remaining after feature selection step:", len(newcols))
    return {'data':X2,'column_flags':fs.get_support(),'new_columns':newcols}


def test_filter_features_pandas():
    '''
    Test filter_features using simple
    example from Iris dataset.
    '''
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X)
    y = list(y)
    result = filter_features(X,y)
    X2 = result['data']
    # print(X.head())
    # print(X2)
    # print(result)
    assert X2.shape[1] == 2

def test_filter_features_pandas_large(
        method_estimate='xgb',
        method_select='recursive',
        extracols=70, extrarows=100, n_jobs=2):
    '''
    Example that tests run times for large number of
    rows (samples) and columns (features).
    One can have more challenging tests for
    extracols > 200, extrarows > 1000.
    '''
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X,columns=['I1','I2','I3','I4'])
    X = X[:100]
    y = y[:100] 
    y = list(y)
    rows, cols = X.shape
    if extracols > 0:
        A = np.random.normal(0, 1, (rows, extracols))
        A = pd.DataFrame(data=A, columns=prefixed_index('V',n=extracols))
        assert not pd.Series(A.columns).duplicated().any()
        X = pd.concat([X,A],axis=1) # add more columns
    if extrarows > 0:
        A2 = np.random.normal(0, 1, (extrarows, len(X.columns)))
        A2 = pd.DataFrame(data=A2, columns=X.columns)
        X = pd.concat([X,A2],axis=0) # add more rows
        y.extend(list(np.random.randint(low=0,high=2,size=extrarows)))
    X.index = range(len(X))
    assert len(y) == X.shape[0] # number of rows must match
    assert not pd.Series(X.columns).duplicated().any()
    
    result = filter_features(X,y,n_jobs=n_jobs,
                method_estimate=method_estimate,
                method_select=method_select)
    X2 = result['data']
    newcols = result['new_columns']
    # print(X.head())
    # print(X2)
    # print(result)
    assert not pd.Series(newcols).duplicated().any()
    assert len(newcols) < X.shape[1] # must have reduced number of features


def test_AutoFeaturesEnsemble_large(
        method_estimate='xgb',
        method_select='recursive',
        extracols=100, extrarows=500, n_jobs=2):
    '''
    Example that tests run times for large number of
    rows (samples) and columns (features).
    One can have more challenging tests for
    extracols > 200, extrarows > 1000.
    '''
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X,columns=['I1','I2','I3','I4'])
    X = X[:100]
    y = y[:100] 
    y = list(y)
    rows, cols = X.shape
    if extracols > 0:
        A = np.random.normal(0, 1, (rows, extracols))
        A = pd.DataFrame(data=A, columns=prefixed_index('V',n=extracols))
        assert not pd.Series(A.columns).duplicated().any()
        X = pd.concat([X,A],axis=1) # add more columns
    if extrarows > 0:
        A2 = np.random.normal(0, 1, (extrarows, len(X.columns)))
        A2 = pd.DataFrame(data=A2, columns=X.columns)
        X = pd.concat([X,A2],axis=0) # add more rows
        y.extend(list(np.random.randint(low=0,high=2,size=extrarows)))
    X.index = range(len(X))
    assert len(y) == X.shape[0] # number of rows must match
    assert not pd.Series(X.columns).duplicated().any()
    factors = np.random.randint(low=1,high=4, size=len(y))
    X['Target'] = y
    assert 'Target' in X.columns 
    result = AutoFeaturesEnsemble(X,target='Target',factors=factors,
                n_jobs=n_jobs,
                method_estimate=method_estimate,
                method_select=method_select)
    X2 = result['data']
    newcols = result['new_columns']
    # print(X.head())
    # print(X2)
    # print(result)
    # print(newcols)
    assert not pd.Series(newcols).duplicated().any()
    # assert X2.shape[1] == 2


def test_filter_features():
    '''
    Testing filter_features using Iris dataset.
    '''
    X, y = load_iris(return_X_y=True)
    X2 = filter_features(X,y)['data']
    # print(X[:5,:])
    # print(X2)
    assert X2.shape[1] == 2


def AutoFeatures(data,target,
    method_estimate='xgb',
    method_select='recursive',
    cv=None,
    ff_corr_max=0.9,
    target_corr_max=0.9,
    target_corr_max_p=0.9,
    n_features_to_select=None,
    n_jobs=1):
    '''
    This function encapsulates the pipeline for feature selection.
    First, features are eliminated based on correlations with
    other features or the target variable. Next, features are 
    eliminated based on model training and iterative removal of worst
    features.

    Parameters:
    data: A Pandas dataframe containing both feature data as well as data for the target variable.
    All columns must be numeric. The target variable should correspond to integer
    values for different outcome classes.
    target(str): The name of the target variable in variable `data`. A numeric representation (integer)
     of the outcome classes.
    cv: Input parameter for cross-validation in SequentialFeatureSelector (see its documentation). An integer
    value corresponds to cross-validation splits (default:cv=5). For very small datasets the cv value may have to be
    reduced.
    ff_corr_max: Maximum correlation between features
    target_corr_max: Maximum correlation with target variable
    target_corr_max_p: Maximum correlation P-value (effectively minimal correlation) with target variable.
    
    Returns(list): Returns a list with following elements:
    'data': A dataframe with reduced number of features
    'new_columns': List of column names of returned dataframe
    'column_flags': List with boolean logical values corresponding to 
     which columns from the input data frame to keep.
    '''
    if not isinstance(target,str):
        raise ValueError("Target has to be a string variable equal to name of target column in data")
    if not target in data.columns:
        raise ValueError("Cannot find target feature column in data",)
    result = filter_correlations(data,target=target,
        ff_corr_max=ff_corr_max,
        target_corr_max_p=target_corr_max_p,
        target_corr_max=target_corr_max)
    data2 = result['data']
    data2x = data2.drop(columns=target)
    data2y = data2[target]
    assert data2x.shape[1] > 0
    result2 = filter_features(data2x, data2y,
        method_select=method_select,
        method_estimate=method_estimate, 
        cv=cv, n_features_to_select=n_features_to_select,
        n_jobs=n_jobs)
    return result2


def AutoFeaturesEnsemble(
    data,
    target,
    factors,
    method_estimate='xgb',
    method_select='recursive',
    cv=None,
    ff_corr_max=0.9,
    target_corr_max=0.9,
    target_corr_max_p=0.9,
    n_features_to_select=None,
    n_jobs=1,
    verbose=True):
    '''
    This function encapsulates the pipeline for feature selection.
    First, features are eliminated based on correlations with
    other features or the target variable. Next, features are 
    eliminated based on model training and iterative removal of worst
    features.

    Parameters:
    data: A Pandas dataframe containing both feature data as well as data for the target variable.
    All columns must be numeric. The target variable should correspond to integer
    values for different outcome classes.
    target(str): The name of the target variable in variable `data`. A numeric representation (integer)
     of the outcome classes.
    factors(list): A list of numbers or strings that are used as factors to divide the dataset into 
    different subsets. These factors decide which 'ensemble' of subsets the AutoFeatures method is run on.
    cv: Input parameter for cross-validation in SequentialFeatureSelector (see its documentation). An integer
    value corresponds to cross-validation splits (default:cv=5). For very small datasets the cv value may have to be
    reduced.
    ff_corr_max: Maximum correlation between features
    target_corr_max: Maximum correlation with target variable
    target_corr_max_p: Maximum correlation P-value (effectively minimal correlation) with target variable.
    
    Returns(list): Returns a list with following elements:
    'data': A dataframe with reduced number of features
    'new_columns': List of column names of returned dataframe
    'column_flags': List with boolean logical values corresponding to 
     which columns from the input data frame to keep.
    '''
    if not isinstance(target, str):
        raise ValueError("Variable \'target\' must be of type string and describe the \
            name of one of the data columns.")
    if not target in data.columns:
        raise ValueError("Variable \'target\' must be correspond to one of the column \
        in variable \'data\'")
    if not isinstance(factors,pd.Series):
        factors = pd.Series(factors)
    assert len(factors) == len(data)
    levels = list(factors.unique())
    result_all = {}
    columns_union = []
    for level in levels:
        flags = (factors == level)
        assert len(flags) == len(data)
        flags.index = range(len(data))
        assert isinstance(flags,pd.Series)
        assert target in data.columns
        data2 = data[flags]
        result2 = AutoFeatures(
            data = data2, target=target,
            method_estimate=method_estimate,
            method_select=method_select,
            cv=cv,
            ff_corr_max=ff_corr_max,
            target_corr_max=target_corr_max,
            target_corr_max_p=target_corr_max_p,
            n_features_to_select=n_features_to_select,
            n_jobs=n_jobs)
        result2 = result2['new_columns']
        # print("Columns for level", level,":",result2)
        # print("Columns union before add:", columns_union)
        result_all[level] = result2
        assert result2 is not None
        columns_union.extend(result2)
        # print("Columns union after add:", columns_union)
    columns_union = list(set(columns_union)) # make unique
    columns_union.sort()
    columns_counts = {}
    columns_sizes = {}
    for col in columns_union:
        count = 0
        sizes = {}
        for level in levels:
            if col in result_all[level]:
                count += 1
            sizes[level] = (len(result_all[level]))
        columns_counts[col] = count
        columns_sizes[col] = sizes
    data_final = data[columns_union]
    return {'data':data_final,
            'new_columns':columns_union,
            'counts':columns_counts,
            'sizes':columns_sizes,}

def test_AutoFeatures():
    '''
    Tests AutoFeatures method using a simple
    synthetic dataset. Checks if the method
    detects and removes columns with too high
    or too low correlation.
    '''
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
        'H':[2.5,8.4, 5.3, -2.4,-1.5,0.0],
        'T':[0,1,0,1,0,1]})
    result = AutoFeatures(data, target='T',cv=3)
    # print(result)
    # print(result['new_columns'])
    assert result['data'].shape[1] == 1
    assert result['new_columns'] == ['G']


def test_AutoFeatures_iris():
    '''
    Tests using AutoFeatures function a truncated
    Iris dataset.
    '''
    X, y = load_iris(return_X_y=True)
    X = X[:99,:]
    y = y[:99]
    X = pd.DataFrame(X)
    y = list(y)
    X.insert(0, 'Target', y)
    assert 'Target' in X.columns
    result = AutoFeatures(X,target='Target')
    X2 = result['data']
    # print(X.head())
    # print(result)
    assert X2.shape[1] == 1

def test_AutoFeatures_iris_nfeatures():
    '''
    Tests using AutoFeatures function a truncated
    Iris dataset.
    '''
    X, y = load_iris(return_X_y=True)
    X = X[:99,:]
    y = y[:99]
    X = pd.DataFrame(X)
    y = list(y)
    X.insert(0, 'Target', y)
    assert 'Target' in X.columns
    result = AutoFeatures(X,target='Target',
        n_features_to_select=2)
    X2 = result['data']
    assert X2.shape[1] == 2

def test_AutoFeaturesEnsemble():
    '''
    Tests AutoFeatures method using a simple
    synthetic dataset. Checks if the method
    detects and removes columns with too high
    or too low correlation.
    '''
    target_corr_max_p=0.2
    target_corr_max=0.9
    data = pd.DataFrame(data={
        'A':[1,2,3,4,5,6,7,8],
        'B':[1,2,3,4,6,6,6,1],
        'C':[6,11,4,10,2,1,1,6],
        'D':[1,1,1,1,1,1,1,1], # constant, should be filtered out
        'E':[-1,1,-1,1,-1,1,1,-1],  # uncorrelated should be filtered out
        'F':[6,11,4,10,2,1.5,1.5,6], # too similar to 'C'
        'G':[1.2,0.3, 4.7, 2.3, 0.5, 0.9,0.9,1.2],
        'H':[2.5,8.4, 5.3, -2.4,-1.5,0.0,0.0,2.5],
        'T':[0,1,0,1,0,1,1,0]})
    factors = ['X','X','X','X','Y','Y','Y','Y']
    result = AutoFeaturesEnsemble(data, 
        target='T', factors = factors, cv=3)
    # print(result)
    assert result['data'].shape[1] == 2


##### Automate saving the models trained and their accuracy 
# model_metrics = pd.DataFrame(columns=['Accuracy', 
#                           'FPR','FNR', 'Precision', 'Recall', 'Auc_train', 'Auc_val', 'Auc_test', 'n_bucket',
#                           'Q1-Positive Rate', 'Qn-Positive Rate', 'n_features', 'Features', 'Feature Importances', 'Model'])

# scale = y_train.value_counts()[0]/y_train.value_counts()[1]

# ## initilize the model
# xgb0 = xgb.XGBClassifier(scale_pos_weight = np.sqrt(scale)) 

def get_model_performance(model, target, features):
  ''' Get the performance of the binary classification model 
  using the given list of features on training, validation 
  and test sets

  Parameters: 
  model: classifier,
  features(list): list of column names to train the model with
  target(str): str target column name
  '''
  clf = model.fit(X_train[features],y_train)
  n_bucket = 4

  test_probs_0 = model.predict_proba(X_train[features])[:, 0]
  quarters = pd.qcut(test_probs_0, 4, labels=list(range(1,n_bucket+1)))
 
  probs = pd.DataFrame({'Prob_0' :test_probs_0, 
                          'Quarter' :quarters})
  cutoffs = probs.groupby('Quarter').max().reset_index()
  cutoff_probs = [-0.1] + list(cutoffs['Prob_0'])[:-1] + [1.1] 

  df_ml_probs_0 = model.predict_proba(X_val[features])[:, 0]
  prob_1 = [1 - x for x in df_ml_probs_0]
  quarters = pd.cut(df_ml_probs_0, cutoff_probs, labels=list(range(1,n_bucket+1)), duplicates='raise')
  df_ml_quarters = pd.DataFrame({'Probability_1': prob_1,
                                'Quarter': quarters,
                                target: y_val})
  perf = df_ml_quarters.groupby('Quarter')[target].mean().round(3).reset_index()

  # y_quarter1 = perf[perf['Quarter']==1][target].mean()
  # y_quarter4 = df_ml_quarters[df_ml_quarters['Quarter']==4][target].mean()
  Q1_Positive_rate = perf[target][0]
  Qn_Positive_rate = perf[target][n_bucket-1]

  y_pred = model.predict_proba(X_train[features])[:,1]
  fpr, tpr, thresholds = roc_curve(y_train, y_pred)
  gmean = np.sqrt(tpr * (1 - fpr))
  index = np.argmax(gmean)
  thresholdOpt = round(thresholds[index], ndigits = 4)
  pred = (model.predict_proba(X_val[features])[:,1] >= thresholdOpt).astype(bool)

  tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
  sum = tn+fp+fn+tp
  accuracy = round((tp+tn)/sum,3)
  tpr = round((tp/(tp+fn)),3)
  fpr = round(fp/(fp+tn),3)
  precision = round(tp/(tp+fp),3)
  recall = round((tp/(tp+fn)),3)
  fnr = round(fn/(fn+tp),3)
  auc_train = round(roc_auc_score(y_train, clf.predict_proba(X_train[features])[:,1]),3)
  auc_val = round(roc_auc_score(y_val, clf.predict_proba(X_val[features])[:,1]),3)
  auc_test = round(roc_auc_score(y_test, clf.predict_proba(X_test[features])[:,1]),3)
  # mcc = round(metrics.matthews_corrcoef(y_val, pred),3)
  # f1 = round(f1_score(y_val, pred, average='binary'),3)
              
  new_row = {'Accuracy': accuracy,
              'FPR': fpr, 
              'FNR': fnr,
              'Precision': precision,
              'Recall': recall,
              'Auc_train': auc_train,
              'Auc_val': auc_val, 
              'Auc_test': auc_test,
              'n_bucket': n_bucket,
              'Q1-Positive Rate': Q1_Positive_rate,
              'Qn-Positive Rate': Qn_Positive_rate,  
              'n_features': len(features),
              'Features': list(features),
              'Feature Importances': dict(zip(features, model.feature_importances_)),
              'Model': model.get_xgb_params()
   }
  return(new_row)


def risk_score(data, target, model, risk_categories, risk_names):
  '''
  Generates risk score based on the probability obtained using the given binary classification model.

  Parameters:
  data: DataFrame
  target(str): Column name for the target variable
  model: sklearn model to be used for calculating risk score
  risk_categories(list): list of categories that are list of features to be used to calculate risk score for that category
  risk_names(list): list of names of the categories
  '''
  for i,features in enumerate(risk_groups):
      model.fit(X_train[features], y_train)
      y_pred_logit = model.predict(X_test[features])
      auc_train = round(roc_auc_score(y_train, model.predict_proba(X_train[features])[:,1]),3)
      auc_test = round(roc_auc_score(y_test, model.predict_proba(X_test[features])[:,1]),3)
      print('AUC of logistic regression classifier on test set: {:.2f}'.format(auc_test))

      for df in [X_train, X_val, X_test]:
          prob_1 = model.predict_proba(df[features])[:, 1]
          df['Risk_score_{}'.format(i)] = prob_1


def test_all():
    test_AutoFeatures_iris_nfeatures()
    test_filter_features_pandas_large(
        extracols=500, extrarows=100,
        method_estimate='xgb', method_select='recursive',
        n_jobs=4)
    test_AutoFeaturesEnsemble()
    test_AutoFeaturesEnsemble_large()
    test_AutoFeatures()
    test_AutoFeatures_iris()
    test_filter_correlations()
    test_filter_features()
    test_filter_features_pandas()


# if __name__ == "__main__":  # only run if this script is exectuted directly
#     test_all()
