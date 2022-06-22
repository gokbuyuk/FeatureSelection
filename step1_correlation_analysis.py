### Remove columns with too many NAs
na_ths = 0.5
cols_to_remove = [col for col in data.columns if data[col].isnull().mean()>na_ths]
data.drop(cols_to_remove, axis=1, inplace=True)

## Remove outliers and counties with namrs investigations <=  10
def remove_outliers_from_column(data, col):
  ''' Removes outliers in given column name(col) from data
        data: pandas data frame
        col: str column name in data
  '''
  Q1 = data[col].quantile(0.25)
  Q3 = data[col].quantile(0.75)
  IQR = Q3 - Q1
  df = data[(data[col] > (Q1 - 1.5 * IQR)) & (data[col] < (Q3 + 1.5 * IQR))]
  return(df)


### Find correlation coefficients and p-values for each variable and the target
states = county_summaries['StateAbbr'].unique().tolist()

corr_coeffs = pd.DataFrame({'State': states})
p_values = pd.DataFrame({'State': states})
completeness_matrix = pd.DataFrame({'State': states})

namrs_outliers_removed = remove_outliers_from_column(county_summaries, target)  

for col in num_cols:
    temp_df = remove_outliers_from_column(namrs_outliers_removed, col) 
    for state in states:
        state_df = temp_df[temp_df['StateAbbr']==state][[col, target]].dropna(axis=0)
        # print('Removed outliers in column', col)
        try: 
            r, p = stats.spearmanr(state_df[col], state_df[target]) ## can be used with pearsonr as well.
            is_complete = True
        except ValueError:
            r, p = [0, 1]
            is_complete = False
        corr_coeffs.loc[corr_coeffs['State']==state, col] = round(r,2)
        p_values.loc[p_values['State']==state, col] = p
        completeness_matrix.loc[p_values['State']==state, col] = is_complete

corr_means = corr_coeffs[num_cols].median().to_dict()
corr_means['State'] = 'Median'
corr_coeffs = corr_coeffs.append(corr_means, ignore_index=True)

pval_means = p_values[num_cols].median().to_dict()
pval_means['State'] = 'Median'
p_values = p_values.append(pval_means, ignore_index=True)


### Export correlation coeffs and p-values
# corr_coeffs.to_csv('namrs_vs_external_correlation_coeffs.csv', index=False)
# p_values.to_csv('namrs_vs_external_corr_p_values.csv', index=False)
# completeness_matrix.to_csv('namrs_vs_external_completeness_matrix.csv', index=False)


## Significantly correlated features in 4+ states at alpha=0.01
alpha = 0.05
n_state = 5
signf_counts = np.sign(corr_coeffs[num_cols][(p_values[num_cols]<alpha)&(completeness_matrix[num_cols]==True)].fillna(0)).sum().abs().reset_index().rename(columns={'index': 'Feature', 0: 'n_significant'})

signf_features = signf_counts[signf_counts['n_significant']>=n_state].sort_values(by='n_significant', ascending=False)['Feature'].tolist()
signf_cols = ['State'] + signf_features

corrs_of_signf_cols = corr_coeffs[signf_features]
pvals_of_signf_cols = p_values[signf_features]

len(signf_features), signf_features

### FDR correction
## Find rejection matrix for H0: r = 0 at alpha
chain = itertools.chain(*p_values[num_cols].values)
fdr_obj = statsmodels.stats.multitest.multipletests(pvals=list(chain), alpha=alpha, method='fdr_bh')
rejection_matrix = fdr_obj[0]

n_states = county_summaries['StateAbbr'].nunique()
n_col = len(num_cols)
reject_df = pd.DataFrame(columns = num_cols)
Inputt = iter(rejection_matrix)
output = [list(islice(Inputt, n_col)) for i in range(n_states+1)]

for i in range(n_states+1):
    reject_df.loc[i, :] = output[i]

p_val_fdr = p_values[num_cols][reject_df[num_cols]==True].max().max()

reject_df

# Features that are still significantly correlated after FDR correction
signf_counts_fdrcontrolled = np.sign(corr_coeffs[num_cols][reject_df[num_cols]==True].fillna(0)).sum().abs().reset_index().rename(columns={'index': 'Feature', 0: 'n_significant'})

signf_features_fdr = signf_counts_fdrcontrolled[signf_counts_fdrcontrolled['n_significant']>=n_state].sort_values(by='n_significant', ascending=False)['Feature'].tolist()

corr_coeffs[signf_features_fdr]
p_values[signf_features_fdr]
len(signf_features), len(signf_features_fdr)


## Plot the correlations of features as a box plot
temp_df = corr_coeffs[signf_features].sort_values(by=n_states, axis = 1)
new_col_dict = {col:col.replace('_', ' ').replace('.', ' ').replace('Pct', '%').replace('plus', '+').capitalize() for col in temp_df.columns}
temp_df.rename(columns=new_col_dict, inplace=True)

still_signf = {new_col_dict[col]: value for col, value in zip(signf_features, [col in signf_features_fdr for col in signf_features])  }
color_dict = pd.DataFrame({'Feature': list(still_signf.keys()), 'still_signf': list(still_signf.values())})
color_dict['still_signf'] = np.where(color_dict['still_signf']==True, 'red', 'cornflowerblue')
color_dict = dict(zip(color_dict['Feature'], color_dict['still_signf']))
        
plt.figure(figsize=(5,15))
sns.boxplot(data=temp_df, orient='h', palette=color_dict)
plt.xlabel("Pearson's Correlation Coefficient")
plt.title("Distribution of Pearson's correlation Coeffients for each variable accross states")
plt.axvline(0, 0, 1, color='grey')
plt.show()
plt.savefig('namrs_figures/ext_var_corr_box_plots_fdr_corrected.png', bbox_inches='tight')
