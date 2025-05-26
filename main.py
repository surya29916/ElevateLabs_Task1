import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Import the dataset
df = pd.read_csv("Titanic-Dataset.csv")  

# Step 2: Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)             
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  
df.drop(columns='Cabin', inplace=True)                         

# Step 3: Convert categorical features into numerical 
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Step 4: Normalize/Standardize numerical features
scaler = StandardScaler()
cols_to_scale = ['Age', 'Fare', 'SibSp', 'Parch']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Step 5: Visualize and remove outliers using IQR method 
# boxplot before removing outliers
sns.boxplot(data=df[['Fare']])
plt.title("Boxplot of 'Fare' Before Removing Outliers")
plt.show()

# IQR method to remove outliers in 'Fare'
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]

# boxplot after removing outliers
sns.boxplot(data=df[['Fare']])
plt.title("Boxplot of 'Fare' After Removing Outliers")
plt.show()

# Final shape of the cleaned dataset
print("Cleaned dataset shape:", df.shape)

# Save to new CSV
df.to_csv("Titanic_Cleaned.csv", index=False)
