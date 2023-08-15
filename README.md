# Banking_Customber_Churn_prediction_using_Auto_ML
we will using ANN models and H2O in this project, In general churn is expressed as a degree of customber inactivity or disengagement, observed over a given time. This mainfests within the data in various forms such as the recency of account actions or change in the account  balance 

we aim to accomplist the following for this study

   . Identify and visualize which factors contribute to customer churn:
   
# Build a prediction model that will perform the following

   #### . Classify if a customber is going to churn or not
   #### . Perferably and based on model performance, choose a model that will attach a probability to the churn to make it easire for customer servies to target low hanging fruits in their efforts to prevent churn

   


# Time Line of this project :

  ### . Data Analysis
  ### . Feature Engineering 
  ### , Model Building using ANN
  ### . Model Building and Prediction using H2O Auto ML


    import pandas as pd 
    from matplotlib import pyplot as plt
    import numpy as np
    %matplotlib inline

    from google.colab improt drive
    drive.momunt ('/content/drive')
    df = pd.read_csv("/content/drive/MyDrive/Churn_Modelling.csv")

    df.head()

    df.drop(['customerId','RowNumber', 'Surname'],axis='columns',inplace=True)

    df.head()

    df.dtypes


# Data Analysis 

      df.head()
      df.nunique()



# We will plot a Pie Chart

     labels = 'Exited(Churned)', 'Retained'
     sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
     explode = (0, 0.1)
     fig1, ax1 = plt.subplots(figsize=(10, 8))
     ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
     ax1.axis('equal')
    plt.title("Proportion of customer churned and retained", size = 20)
    plt.show()


     import seabron as sns


     # We first review the 'Status' relation with categorical variables
    fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
    sns.countplot(x='Geography', hue = 'Exited',data = df, ax=axarr[0][0])
    sns.countplot(x='Gender', hue = 'Exited',data = df, ax=axarr[0][1])
    sns.countplot(x='HasCrCard', hue = 'Exited',data = df, ax=axarr[1][0])
    sns.countplot(x='IsActiveMember', hue = 'Exited',data = df, ax=axarr[1][1])



    # Relations based on the continuous data attributes
    fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
    sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = df, ax=axarr[0][0])
    sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = df , ax=axarr[0][1])
    sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][0])
    sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][1])
    sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][0])
    sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][1])


    tenure_churn_no = df[df.Exited==0].Tenure
    tenure_churn_yes = df[df.Exited==1].Tenure

    plt.xlabel("tenure")
    plt.ylabel("Number Of Customers")
    plt.title("Customer Churn Prediction Visualiztion")

    plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.95, color=['red','green'],label=  ['Churn=Yes','Churn=No'])
    plt.legend()


# Feature Engineering 

Making a new colume BalanceSalaryRatio


     df['BalanceSalaryRatio'] = df.Balance/df.EstimatedSalary
     sns.boxplot(y='BalanceSalaryRatio',x = 'Exited', hue = 'Exited',data = df)
     plt.ylim(-1, 5)


     df['TenureByAge'] = df.Tenure/(df.Age)
    sns.boxplot(y='TenureByAge',x = 'Exited', hue = 'Exited',data = df)
    plt.ylim(-1, 1)
    plt.show()

# Printing the categorical variables

    def print_unique_col_values(df):
        for colume in df:
             if df[column].dtypes=='object':
                print(d'{colum}:{df[column].unique()}')

    print_unique_col_cal_values(df)


# Label Encoding

    df['Gender'].replace({'Male' : 1, 'Female' : 0},inplace= True)


# One Hot Encoding method

    df1 = pd.get_dummies(data=df, columns=['Geography'])
    df1.head()


    scale_var = ['Tenure','CreditScore','Age','Balance','NumOfProducts','EstimatedSalary']

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df1[scale_var] = scaler.fit_transform(df1[scale_var])

    df1.head()

    X = df1.drop('Exited',axis='columns')  ##independent features
    y = df1['Exited']  ##dependent feature

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)

    len(X_train.columns)

    
# Model Building and prediction

  ## The Sequential and prediction 

  

    

