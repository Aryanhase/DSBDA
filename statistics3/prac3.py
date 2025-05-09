import pandas as pd

df = pd.read_csv(r"C:\Users\aryan\OneDrive\Desktop\DSBDA\prac3/evSales.csv")

df

# mean

df.Battery_Capacity_kWh.mean() #This works only when the column name is a valid Python identifier

df.loc[:, 'Battery_Capacity_kWh'].mean() #selects all rows (:) for the specified column 'Battery_Capacity_kWh'

df.Battery_Capacity_kWh[0:4].mean() #includes the first four rows of the Battery_Capacity_kWh

# median

df.Battery_Capacity_kWh.median()
df.loc[:, 'Battery_Capacity_kWh'].median()
df.Battery_Capacity_kWh[0:4].median()

# mode

df.mode()
df.loc[:, 'Battery_Capacity_kWh'].mode()

# minimum
df.Battery_Capacity_kWh.min(skipna=False) #skipna=False specifies that missing values should not be ignored

df.loc[:,'Battery_Capacity_kWh'].min(skipna=False)

# maximum
df.max() #maximum value for each column in the df excluding NaN default 
df.loc[:, 'Battery_Capacity_kWh'].max(skipna=True)

# Standard deviation
df.Battery_Capacity_kWh.std()
df.loc[:,'Battery_Capacity_kWh'].std()

#group by
df.groupby(['Battery_Capacity_kWh'])['Discount_Percentage'].mean() #calculates the mean of Discount_Percentage for each group

from sklearn import preprocessing
enc = preprocessing.OneHotEncoder() 
enc_df = pd.DataFrame(enc.fit_transform(df[['Region']]).toarray()) #converts categorical values into numbers
enc_df 
df_encode =df.join(enc_df) #joins the original DataFrame df with the newly created enc_df
df_encode 
