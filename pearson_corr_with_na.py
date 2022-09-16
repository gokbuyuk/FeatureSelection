### Credit: Eckart Bindewald

def my_pearsonr(x, y, mincases=4):
    """
    Pearson correlation coefficient 'sanitized' in the sense
    that it works with NA as some of the input values.
    It returns a 2-tuple of correlation coefficient r
    and P-value p.
    All x[i],y[i] pairs where either x[i] or y[i] or both
    are equal to numpy.nan are being ignored. If after removal
    of these cases less than `mincases` are remaining,
    numpy.nan are return for both r and p.

    Parameters:
    x(pandas.Series): A series of x values
    y(pandas.Series): A series of y values.

    Returns:
    A 2-tuple of r,p where r is the Pearson correlation
    coefficient and p is the P-value of the correlation test.
    """
    assert isinstance(x, pd.Series)
    assert isinstance(y, pd.Series)
    # x = pd.Series(x)
    # y = pd.Series(y)
    goodflags = (~x.isna()) & (~y.isna())
    if goodflags.sum() < mincases:
        return np.nan, np.nan
    return pearsonr(x[goodflags], y[goodflags])


def my_pearsonr_category(x, y, category, missing=["Empty", "Missing"], mincases=4):
    """
    Pearson correlation coefficient 'sanitized' in the sense
    that it works with NA as some of the input values.
    It returns a 2-tuple of correlation coefficient r
    and P-value p.
    All x[i],y[i] pairs where either x[i] or y[i] or both
    are equal to numpy.nan are being ignored. If after removal
    of these cases less than `mincases` are remaining,
    numpy.nan are return for both r and p.

    Parameters:
    x(pandas.Series): A series of x values but a categorical variable
    y(pandas.Series): A series of y values. Typically 0 and 1
    category: the value for the 'Yes' answer. After removing missing values, all
     remaining values are considered the 'No' answer.
    missing: A set of values in addition to np.nan that are viewed as missing data.
    Returns:
    A 2-tuple of r,p where r is the Pearson correlation
    coefficient and p is the P-value of the correlation test.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if missing is None:
        missine = []
    elif isinstance(missing, str):
        missing = [missing]
    yesv = x == category
    missingv = y.isna() | x.isna() | x.isin(missing)
    x2 = x[~missingv]
    y2 = y[~missingv]
    yesv2 = (x2 == category).astype(int)
    return my_pearsonr(yesv2, y2, mincases=mincases)
    # multiple categories, run Chi-Square test instead:
    # xcategs = x2.unique()
    # ycategs = y2.unique()
    # xl = [None]*len(xcategs)
    # yl = [None]*len(ycategs)
    # for i in range(len(xcategs)):
    #     xcategs[i] = [None]*len(ycategs)
    #     for j in range(len(ycategs)):
    #         count = 

