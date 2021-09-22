# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 07:57:05 2021

@author: Steve Adamo
sadamo@gmail.com

"""


#%%
# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm 

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing

#%%
# expand pandas print representation 

# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 120)
pd.options.mode.chained_assignment = None  # default='warn'

#%%
def tmm(df):
    """the function tell_me_more (tmm) takes in an argument, "df" that is a dataframe
    it then provides a variety of numeric and visual summary options 
    the function calles a separate function show_me_more for graphical 
    displays.  This function is useful for initial data cleaning and analysis
    as it allows the coder to view the data via a variety of dimensions to make
    quick changes."""
    
    # choose either numeric or graphical data summary
    num_plt = input('Information\nType "n" for numeric\nType "g" for graphic\n\nany other key to exit\n\n-----> ')
    
    # numeric data summery chosen
    if num_plt.lower() == 'n':
        # offers a menue of data summarization choices
        choose_info = input('What would you like to display?\n\
                                \n\t1. Header\
                                \n\t2. Tail\
                                \n\t3. Shape\
                                \n\t4. DataTypes\
                                \n\t5. Columns\
                                \n\t6. Features by Index\
                                \n\t7. Describe\
                                \n\t8. Null Value Counts\
                                \n\t9. Correlation Matrix\
                                \n\n\tAny other key to quit\n\n\t-----> ')
        # print first 5 rows to verify data looks accurate
        if choose_info =="1":
            print(df.head())
            tmm(df)
        # print first 5 rows to verify data looks accurate
        if choose_info =="2":
            print(df.tail())
            tmm(df)
        # provides shape information.  Quick summary for 
        # did I just blow up my data and remove soemthing I shouldn't have?
        elif choose_info == "3":
            print('\nThe dataframe has the following shape: ',df.shape)
            tmm(df)        # dtypes good for identifying attributes for columns
        if choose_info == "4":
            print(df.dtypes)
            tmm(df)
        # keys is excellent for listing column values in order.  Great
        # for large datasets where slicing many columns at once.  If 
        # index numbers are used instead of column names, there could be
        # problems down the road as the dataframes change.  
        elif choose_info == "5":
            print(list(df.columns))
            # print(df.keys())
            tmm(df)
        # lists column names by index number if searching for index        
        elif choose_info == "6":
            cols = df.columns.tolist()
            for i, item in enumerate(cols,0):
                print(i, '.\t' + item, '\n', sep='',end='')
            tmm(df)
        # provides summary data including count, mean, standard deviation, min, 
        # max and quartiles
        elif choose_info == "7":
            print(df.describe())
            # input('hit enter to view categorical variables')
            # print(df.describe(exclude = 'number'))
            tmm(df)
        # review dataframe to identify possible null values
        # print null value counts by attribute
        elif choose_info == "8":
            print('\nDataframe null value counts\n\n')
            print(df.isnull().sum())
            tmm(df)
        # prints a correlation matrix for the data
        elif choose_info == "9":
            print(df.dtypes)
            # moves the dependent variable to the end of the dataframe
            # if it is included with the dataframe
            dep_var = input('What is the dependent variable (or "none")?  ')
            if dep_var != "none":
                df1 = df.pop(dep_var) # remove column b and store it in df1
                df[dep_var]=df1
                X = df.iloc[:,:-1]
            else:
                X = df.iloc[:,:]
            # create the correlationmatrix
            correlation_matrix = X.corr()
            print ((correlation_matrix))
            tmm(df)
    # if the user wants to view graphical summaries, this response
    # to the input variable calls the show_me_more function
    elif num_plt.lower() == 'g':
        # graphical analysis can take time with large datasets
        # additionally, too much data can make the visualization
        # hard to read
        # the data_subset is created to help speed up analysis as well
        # as processing time.  This subset does not affect the MLR model
        data_subset = int(input("This is a large dataset for graphical analysis.\nIf you would like to randomly generate a smaller subset of that data,\nenter the number of rows to use: "))
        # crate sample subset of the larger dataframe
        df = df.sample(n=data_subset)
        # call show_me_more function
        show_me_more(df)

#%%
def show_me_more(df):
    """show_me_more is a function that takes in one argument, a dataframe
    it then offers the user a variety of visualization tools to analyse the 
    data.  This is a quick way to help clean and analyze data without 
    having to repeat the same commands"""
    
    # there has been intermittent issues with numpy.  Sources online have 
    # recommended uninstall/reinstall which worked for awhile.  
    # in the end, was easier to add to the function
    import numpy as np
    # offers the user a variety of graphical visualizations
    choose_graph = input('What would you like to display?\n\
                            \n\t1. Univariate Plot\
                            \n\t2. Univariate histogram\
                            \n\t3. Strip plot\
                            \n\t4. All histograms\
                            \n\t5. Box plot\
                            \n\t6. Scatter plot\
                            \n\t7. Visualize Correlation Matrix\
                            \n\t8. Pair plot***\
                            \n\t\t *** Large Dataframes will take a long time\
                            \n\n\tAny other key to quit\n\n\t-----> ')
    # privides a univariate lot of 1 y value stored as uni_y    
    if choose_graph == "1":
        # print dtypes for reference
        print(df.dtypes)
        uni_y = input("Choose an feature for the univeriate scatter plot ")
        # allows the user to group by an attribute
        attr = input('What feature to group by (or "none")? ')
        if attr.lower() == 'none':
            # graph scatterplot with seaborn
            sns.scatterplot(x=df.index, y = df[uni_y])
            # allows for customization of the graph
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(True)
            plt.grid(False)
            plt.show()
            show_me_more(df)

        else:    
            # graph scatterplot with seaborn
            sns.scatterplot(x=df.index, y = df[uni_y], hue = df[attr])
            # allows for customization of the graph
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(True)
            plt.grid(False)
            plt.show()
            show_me_more(df)

    elif choose_graph == "2":
        # print dtypes for reference
        print(df.dtypes)
        # create a specific histogram for 1 attribute
        x_attr = input("Choose an feature for the histogram ")
        # uses seaborn to plot the histogram
        sns.histplot(df[x_attr], kde = False, color = 'blue')
        plt.show()
        show_me_more(df)

    elif choose_graph == "3":
        # print dtypes for reference
        print(df.dtypes)
        # create a strip plot.  Very good for visualizing groupings
        # and shape over vertical axis
        strip_y = input("Choose an feature for the y-axis ")
        # create a grouping
        attr = input('What feature to group by? ')
        # plot with seaborn
        sns.stripplot(x=df[attr], y = df[strip_y])
        plt.show()
        show_me_more(df)

    elif choose_graph == "4":
        # print all histograms in a dataframe for quick visualization 
        # of shape
        for column in df:
            if (df[column].dtypes) == 'float64' or\
                (df[column].dtypes) == 'int64':
                # plot with seaborn
                sns.histplot(df[column], kde = False, color = 'blue')
                plt.show()

        show_me_more(df)

    elif choose_graph == "5":
        # print(df.dtypes)
        # prints all box plots for attributes in a dataframe
        for column in df:
            # identifies values that are only type float
            # future modifications will need to be made for different
            # float types
            if (df[column].dtypes) == 'float64':
                # set preferences
                sns.set_theme(style="whitegrid")
                tips = sns.load_dataset("tips")
                # plots with seaborn
                sns.boxplot(x=df[column])
                plt.show()
        show_me_more(df)

    elif choose_graph == "6":
        # print dtypes for reference
        print(df.dtypes)
        # asks users for x, y input for scatterplot
        # if a quick bivariate visualization is preferred
        x_attr = input("Choose a feature for the x-axis ")
        y_attr = input("Choose a feature for the y-axis ")
        # allows the data to be grouped
        attr = input('What feature to group by (or "none")? ')
        
        if attr.lower() == 'none':
            # plot with seaborn
            sns.scatterplot(data=df, x=x_attr, y=y_attr)
            plt.show()

        else:
            # plot with seaborn
            sns.scatterplot(data=df, x=x_attr, y=y_attr, hue=attr)
            plt.show()

        show_me_more(df)

    elif choose_graph == "7":
        # print dtypes for reference
        print(df.dtypes)
        # asks the user to input a hurdle for a correlation visualization
        # any correlation value below the hurdle will be colored in the
        # "base" heatmap color.  Good if you have many variables
        # and a lot of correlation and are having trouble 
        # identifying "most correlated"
        hurdle = float(input('What hurdle would you like for your correlation matrix visualization? '))
        # remove object variable types
        dep_var = input('Remove Objects? (or "none")?  ')
        if dep_var != "none":
            # move object to end of dataframe to ignore
            df1 = df.pop(dep_var) # remove column b and store it in df1
            df[dep_var]=df1
            X = df.iloc[:,:-1]
            variables = df.columns[:-1]

        else:
            # if no objects in the dataframe, choose everything
            X = df.iloc[:,:]
            variables = df.columns[:]
        
        # plot heatmap based on values from correlation matrix
        # hurdle represents a floor value to group small values together
        # (and ignore)
        R = np.corrcoef(X, rowvar=0)
        R[np.where(np.abs(R)<hurdle)] = 0.0
        # customize heatmap settings, colors and axes
        heatmap = plt.pcolor(R, cmap=mpl.cm.coolwarm, alpha=0.8)
        heatmap.axes.set_frame_on(False)
        heatmap.axes.set_yticks(np.arange(R.shape[0]) + 0.5, minor=False)
        heatmap.axes.set_xticks(np.arange(R.shape[1]) + 0.5, minor=False)
        heatmap.axes.set_xticklabels(variables, minor=False)
        plt.xticks(rotation=90)
        heatmap.axes.set_yticklabels(variables, minor=False)
        plt.tick_params(axis='both', which='both', bottom='off', \
                        top='off', left = 'off', right = 'off')
        plt.colorbar()
        plt.show()  
        
        show_me_more(df)
 
    elif choose_graph == "8":
        # print dtypes for reference
        print(df.dtypes)
        # choose variable to group by
        attr = input('What feature to group by (or "none")? ')
        if attr.lower() == 'none':
            # seaborn pairplot creates a matrix of scatter plots
            # for every variable against itself and every other
            # variable to identify possible correlation
            sns.pairplot(df)

        else:    
            sns.pairplot(df, hue=attr)

    else:
        return

#%%

def makeMLR (df):
    """makeMLR is a function that takes in one value, a dataframe.
    It then asks the user for a dependent variable form that dataframe and 
    creates a Multiplelinear Regression. The function then evaluates the MLR
    by providing an OLS regression summary"""
    # establish global variables, may be able to remove this for future versions
    global X, y, observations, variables, Xc, linear_regression, fitted_model
    # print dtypes for reference
    print(df.dtypes)
    # ask the user for the target variable, y
    dep_var = input('What is the dependent variable?  ')
    # move dependent variable column to the end
    # this eliminates it from the OLS summary if the slice is [:,:-1]
    df1 = df.pop(dep_var) # remove column b and store it in df1
    df[dep_var]=df1

    # assign y to the dependent variable choice values from the df    
    y = df[dep_var].values
    # assign X to the dataframe predictor variables.
    # dependent variable will not be included due to pop function above
    # moving the variable to the end
    X = df.iloc[:,:-1]
    # determint the length of the df and assign to observations
    observations = len(df)
    # assign variables from dataframe to variables
    variables = df.columns[:-1]
    # add constant to Xc via statsmodels
    Xc = sm.add_constant(X)
    # create the linear regression with statsmodels
    linear_regression = sm.OLS(y,Xc.astype(float))
    # create a fitted model of the regression
    fitted_model = linear_regression.fit()
    # print a summary OLS 
    print(fitted_model.summary())
    
#%%

# =============================================================================
# VARIANCE INFLATION FACTOR
# =============================================================================
  
# VIF dataframe
def vif(df):
    """vif is a function that takes a dataframe and analyzes the 
    Variance Inflaction Factor using statsmodels.  The function 
    then takes the VIF values and alerts the user to high correlation
    and Multicollinearity"""
    # create a pd dataframe and assign it the values from X
    X = df.iloc[:,:-1]

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
      
    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                              for i in range(len(X.columns))]
    
    # assign df to vif data to add a column
    df = vif_data

    df.loc[(df['VIF'] > 5), ''] = '<--- high correlation' 
    df.loc[df['VIF'] > 10, ''] = '<------  MULTICOLLINEARITY' 
    df.loc[df['VIF'] <= 5, ''] = '' 
    
    print(df)
#%%
##############################################################################
##############################################################################
##############################################################################
##############################################################################

def make_PCA(df, col_names):
# standardized the data
    #standardize data
    scaled_data  = StandardScaler().fit_transform(df.T)
    
    pca = PCA()
    
    pca.fit(scaled_data)    
    
    pca_data = pca.transform(scaled_data)
    per_var = np.round(pca.explained_variance_ratio_*100, decimals =1)
    
    mean_vec = np.mean(scaled_data, axis=0)
    cov_mat = (scaled_data - mean_vec).T.dot((scaled_data - mean_vec)) / (scaled_data.shape[0]-1)
    print('Covariance matrix \n%s' %cov_mat)

  
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

    # scree plot demonstrating percentage of explained variance
    plt.bar(x=range(1, len (per_var)+1), height = per_var, tick_label = labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

    input('Press Enter ')  

    #create PCA dataframe
    pca_df = pd.DataFrame(pca_data, index = col_names, columns = labels)
    # customize plot with ax prefix
    ax = plt.gca()
    # scatter plot of pca 1 vs pca 2 with variable names
    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title('PCA Graph')
    plt.xlabel('PC1 = {0}%'.format(per_var[0]))
    plt.ylabel('PC2 = {0}%'.format(per_var[1]))
    
    ax.axhline(linewidth=1, linestyle='dashed', color='r')
    ax.axvline(linewidth=1, linestyle='dashed', color='r')
    
    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    plt.show()

    