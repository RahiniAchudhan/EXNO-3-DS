## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df.head()
```
# Output
![image](https://github.com/user-attachments/assets/b6c5d9e2-92fc-4090-a3a8-9b3a544a5ba1)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
# Output
![image](https://github.com/user-attachments/assets/24d78294-8983-4945-8d69-8a40413069bc)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
# Output
![image](https://github.com/user-attachments/assets/12831df5-714d-4b86-afbf-09e77743e8c9)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
# Output
![image](https://github.com/user-attachments/assets/c2c6ae80-e7df-4406-b53d-0316154ab7c1)

```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)  
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
```
df2=pd.concat([df2,enc],axis=1)
df2
```
# Output
![image](https://github.com/user-attachments/assets/a86f84b7-4719-47fd-9d1f-37b32e3418d1)

```
pd.get_dummies(df2,columns=["nom_0"])
```
# Output
![image](https://github.com/user-attachments/assets/7e879fa7-5826-4e54-a59e-e69acc6c887d)

```
pip install --upgrade category_encoders
```
# Output
![image](https://github.com/user-attachments/assets/ec765acb-162e-498b-ac1b-65a3c6b07eb8)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
# Output
![image](https://github.com/user-attachments/assets/f8ad9917-846d-413f-9b81-17948cee0daf)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
# Output
![Screenshot 2025-04-23 095707](https://github.com/user-attachments/assets/da71aadb-513e-4a15-8ff3-578700687b52)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
# Output
![Screenshot 2025-04-23 095732](https://github.com/user-attachments/assets/90cfb353-54b0-431f-bdb3-60e951cd1fee)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
# Output
![image](https://github.com/user-attachments/assets/84756de9-2ce9-49a4-a92d-c921f0366cab)

```
df.skew()
```
# Output
![Screenshot 2025-04-23 095920](https://github.com/user-attachments/assets/670f9e94-12de-4a1c-b8aa-f4e19d6ce142)

```
np.log(df["Highly Positive Skew"])
```
# Output
![Screenshot 2025-04-23 095942](https://github.com/user-attachments/assets/798d4946-c47f-4be1-8cdd-56e2d7134bda)

```
np.reciprocal(df["Moderate Positive Skew"])
```
# Output
![Screenshot 2025-04-23 100006](https://github.com/user-attachments/assets/9896a3d1-be02-4e3e-a9ec-75bc798f8f69)

```
np.sqrt(df["Highly Positive Skew"])
```
# Output
![image](https://github.com/user-attachments/assets/3635e1a6-cd81-4bf8-b515-de9972279afb)

```
np.square(df["Highly Positive Skew"])
```
# Output
![image](https://github.com/user-attachments/assets/fa6b42b1-c927-45da-b470-f84a8aa839ff)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
# Output
![image](https://github.com/user-attachments/assets/d0a88cdb-6ad0-4a69-8d09-2d36df088d0b)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
# Output
![image](https://github.com/user-attachments/assets/c3df4d61-d029-4a44-9545-3535357a4362)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
# Output
![image](https://github.com/user-attachments/assets/7139000b-555c-42a4-86b9-c53910a042e3)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
# Output
![image](https://github.com/user-attachments/assets/bac83b61-af67-4e05-8d3e-dc6f44e8ddd1)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
# Output
![Screenshot 2025-04-23 100430](https://github.com/user-attachments/assets/193230ff-a5cf-4236-9207-ffd1bd79ede2)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
# Output
![Screenshot 2025-04-23 100452](https://github.com/user-attachments/assets/d0e86c11-e4d3-4d50-83f7-37704bc9fb30)


# RESULT:
         Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
