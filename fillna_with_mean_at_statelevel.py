def get_mean_imputation_dict_by_state(df, state_col):
    ''' Computes and returns a dictionary of means to impute for all columns by state
    and imputed dataframe
    df: pandas data frame
    state_col: state column name in df
    returns: df_imputed, na_impute_dict'''
    states = df[state_col].unique().tolist()
    missing_values_per_state = pd.DataFrame({'State': states})
    state_means_per_feature = pd.DataFrame({'State': states})
    state_medians_per_feature = pd.DataFrame({'State': states})
    state_stds_per_feature = pd.DataFrame({'State': states})
    normality_stat = pd.DataFrame({'State': states})
    normality_pval = pd.DataFrame({'State': states})
    isnormal = pd.DataFrame({'State': states})
    alpha_normal = 0.05
    
    na_impute_dict = dict()
    for col in num_cols:
        col_dict = dict()
        for state in states:
            state_df = df[df[state_col]==state][[col, target]]
            missing = state_df[col].isna().mean()
            mean = state_df[col].mean()
            median = state_df[col].median()
            std = state_df[col].std()
            stat, p = shapiro(state_df[col])
            normal =( p < alpha_normal)
            
            missing_values_per_state.loc[missing_values_per_state['State']==state, col] = round(missing, 3)
            state_means_per_feature.loc[state_means_per_feature['State']==state, col] = round(mean, 3)
            state_medians_per_feature.loc[state_medians_per_feature['State']==state, col] = round(median, 3)
            state_stds_per_feature.loc[state_stds_per_feature['State']==state, col] = round(std, 3)
            normality_stat.loc[normality_stat['State']==state, col] = round(stat, 3)
            normality_pval.loc[normality_pval['State']==state, col] = round(p, 3)
            isnormal.loc[isnormal['State']==state, col] = normal
            col_dict[col] =    
            
            ## Imputation
            na_indicator = col + '_ismissing'
            df_imputed[df_imputed[state_col]==state][na_indicator] = df_imputed[df_imputed['StateAbbr']==state][col].isna()
            if normal == True:
                impute_value = mean
                df_imputed[df_imputed[state_col]==state][col].fillna(impute_value, inplace=True)
            else:
                if median != np.nan:
                    impute_value = median
                    df_imputed[df_imputed[state_col]==state][col].fillna(impute_value, inplace=True)
                else: 
                    stat, p = shapiro(county_summaries[col])
                    overall_mean = county_summaries[col].mean()
                    overall_median = county_summaries[col].median()
                    if p < alpha_normal:
                        impute_value = overall_mean
                        df_imputed[df_imputed[state_col]==state][col].fillna(impute_value, inplace=True)
                    else:
                        impute_value = overall_median
                        df_imputed[df_imputed[state_col]==state][col].fillna(impute_value, inplace=True)
            
            col_dict[col] = col_dict.append({state: impute_value})            
                        
                
        na_impute_dict[col] = col_dict 
    
    return(df_imputed, na_impute_dict)
        
