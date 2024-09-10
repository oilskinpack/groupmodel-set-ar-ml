import os
import numpy as np
import pandas as pd


class DfHelper():


    def union_all_dfs(directory):
        dfList = []
        fullDf = np.NaN
        files = os.listdir(directory)
        for file in files:
            fileName = fr'\{file}'
            fullPath = directory + fileName
            df = pd.read_csv(fullPath, sep=';')
            dfList.append(df)
            df['Объект'] = file[:3]
        fullDf = pd.DataFrame(columns=dfList[0].columns)
        for df in dfList:
            fullDf = pd.concat([fullDf, df], sort=False, axis=0)
        return fullDf

    def percent_missing(df):
        res = res = 100 * df.isnull().sum() / len(df)
        res = res[res > 0].sort_values()
        return res

    def convertToDouble(old_value):
        try:
            value = old_value.astype(str).str.replace(',','.')
            value = value.astype(float)
            return value
        except:
            return old_value

    def create_bool_feature_by_contains(fullDf,paramName,value,inverse = False):
        if(inverse):
            npArr = np.where(fullDf[paramName].str.contains(value, case=False) == True, 0, 1)
        else:
            npArr = np.where(fullDf[paramName].str.contains(value, case=False) == True, 1, 0)
        return npArr

    def replace_value(fullDf, search_col_name,search_value, new_value,changing_col):
        npArr = np.where(fullDf[search_col_name] == search_value
                         , new_value
                         , changing_col)
        return npArr

    def show_unique_by_two_conditions(fullDf,col_one,val_one,col_two,val_two,showed_cols):
        res = fullDf[(fullDf[col_one] == val_one) & (fullDf[col_two] == val_two)][
            showed_cols].value_counts()
        return res
