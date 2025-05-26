# ElevateLabs_Task1
# Titanic Data Preprocessing

This project performs data cleaning and preprocessing on the Titanic dataset to prepare it for analysis or machine learning tasks.

## Steps Performed

1. **Import Dataset**  
   Loaded the Titanic dataset using pandas.

2. **Handle Missing Values**  
   - Filled missing `Age` values with the median.  
   - Filled missing `Embarked` values with the mode.  
   - Dropped the `Cabin` column due to too many missing values.

3. **Encode Categorical Variables**  
   Converted `Sex` and `Embarked` into numerical format using one-hot encoding.

4. **Standardize Numerical Features**  
   Applied standard scaling to `Age`, `Fare`, `SibSp`, and `Parch` columns.

5. **Detect and Remove Outliers**  
   Used the IQR method to detect and remove outliers in the `Fare` column, with boxplots before and after.

6. **Export Cleaned Data**  
   Saved the cleaned dataset to a new CSV file named `Titanic_Cleaned.csv`.

## Tools Used
- Python
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib
