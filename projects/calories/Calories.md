## **Project: Predition for the Diet Types from Recipt Data**

### **Participants:**  Preeti, Jialin Li(Cullen)
### **Date:** 08-14-2021
---
### **Research Hypothesis:** 
* From our perception, food's caliors have a high correlation with recipe intergration. With this perception, we made a prediction to each of the recipes from their intergrations is belongs to the type of Vegetarian,  or Gluten Free, etc.
### **Research Questions:** 
1. Does the volumes of ingredients affect the type of recipe?
2. Is there a correlation between the ingredients included in each of the recipes?

---

### **Data Source:**

1. **Food Caliors Tables** : resources from orginzation.
2. **Recipe box** : https://eightportions.com/datasets/Recipes/
3. (Not Sure the Usage yet) **RecipeDB: a resource for exploring recipes** : https://academic.oup.com/database/article/doi/10.1093/database/baaa077/6006228
4. **Calorie King** : https://www.calorieking.com/us/en/developers/food-api/
---
### **Steps:**
1. [**Download Dataset**](#1.-Download-Dataset)
2. [**Install dependentcies**](#2.-Install-dependentcies)
3. [**Describe Data**](#3.-Describe-Data)
4. [**Data Preparation**](#4.-Data-Preparation)
    - 4.1 [Reorganize Dataset](#4.1-Reorganize-Dataset)
    - 4.2 [Standardize Data](#4.2-Standardize-Data)
    - 4.3 [Null Values Checking](#4.3-Null-Values-Checking)
    - 4.4 [Outliers Checking](#4.4-Outliers-Checking)
    - 4.5 [Correlation and Regression Analysis](#4.5-Correlation-and-Regression-Analysis)
    - 4.6 [Data Evaluation](#4.4-Data-Evaluation)
5. [**Algorithm Implementation and Evaluation**](#5.-Algorithm-Implementation-and-Evaluation)
6. [**Conclusion**](#6.-Conclusion)
    - 6.1 [Answers of Research Questions](#6.1-Answers-of-Research-Questions)
    - 6.2 [Feasibility Conclusion](#6.2-Feasibility-Conclusion)
    - 6.3 [Improval Suggestions](#6.3-Improval-Suggestions)

### 1. **Download Dataset**

### 2. **Install dependentcies**



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
df = pd.read_csv('food_table.csv')
```

### 3. **Describe Data**


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>food_name</th>
      <th>calories</th>
      <th>carbs</th>
      <th>dietary_fibre</th>
      <th>sugar</th>
      <th>protein</th>
      <th>fats</th>
      <th>saturated_fat</th>
      <th>polyunsaturated_fat</th>
      <th>monounsaturated_fat</th>
      <th>sodium</th>
      <th>cholesterol</th>
      <th>glycemic_index</th>
      <th>image</th>
      <th>type</th>
      <th>quantities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Abiyuch, raw</td>
      <td>0.69</td>
      <td>1.760000e-01</td>
      <td>5.300000e-02</td>
      <td>8.550000e-02</td>
      <td>1.500000e-02</td>
      <td>0.001</td>
      <td>0.00014</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>2.000000e-01</td>
      <td>4.940656e-324</td>
      <td>None</td>
      <td>com.comidaforfamilias.caloriehask:drawable/abi...</td>
      <td>Food</td>
      <td>1/2 cup: 114g</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Adzuki beans, boiled</td>
      <td>1.28</td>
      <td>2.477000e-01</td>
      <td>7.300000e-02</td>
      <td>4.940656e-324</td>
      <td>7.520000e-02</td>
      <td>0.001</td>
      <td>0.00036</td>
      <td>2.100000e-04</td>
      <td>9.000000e-05</td>
      <td>8.000000e-02</td>
      <td>4.940656e-324</td>
      <td>None</td>
      <td>com.comidaforfamilias.caloriehask:drawable/adz...</td>
      <td>Food</td>
      <td>1 cup: 230g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Almond butter</td>
      <td>6.14</td>
      <td>1.882000e-01</td>
      <td>1.030000e-01</td>
      <td>4.430000e-02</td>
      <td>2.096000e-01</td>
      <td>0.555</td>
      <td>0.04152</td>
      <td>1.361300e-01</td>
      <td>3.244500e-01</td>
      <td>7.000000e-02</td>
      <td>4.940656e-324</td>
      <td>Low</td>
      <td>com.comidaforfamilias.caloriehask:drawable/alm...</td>
      <td>Food</td>
      <td>1 tbsp: 16g, 1 cup: 250g</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Almond flour</td>
      <td>5.71</td>
      <td>2.143000e-01</td>
      <td>1.070000e-01</td>
      <td>3.570000e-02</td>
      <td>2.143000e-01</td>
      <td>0.500</td>
      <td>0.01000</td>
      <td>3.500000e-02</td>
      <td>8.999000e-02</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>None</td>
      <td>com.comidaforfamilias.caloriehask:drawable/alm...</td>
      <td>Food</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Almond oil</td>
      <td>8.84</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>1.000</td>
      <td>0.08200</td>
      <td>1.740000e-01</td>
      <td>6.990000e-01</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>None</td>
      <td>com.comidaforfamilias.caloriehask:drawable/alm...</td>
      <td>Food</td>
      <td>1 tsp: 4.5g, 1 tbsp: 13.6g, 1 cup: 218g</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>food_name</th>
      <th>calories</th>
      <th>carbs</th>
      <th>dietary_fibre</th>
      <th>sugar</th>
      <th>protein</th>
      <th>fats</th>
      <th>saturated_fat</th>
      <th>polyunsaturated_fat</th>
      <th>monounsaturated_fat</th>
      <th>sodium</th>
      <th>cholesterol</th>
      <th>glycemic_index</th>
      <th>image</th>
      <th>type</th>
      <th>quantities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>841</th>
      <td>842</td>
      <td>Yogurt, plain, whole milk</td>
      <td>0.61</td>
      <td>0.0466</td>
      <td>4.940656e-324</td>
      <td>0.0466</td>
      <td>0.0347</td>
      <td>3.250000e-02</td>
      <td>2.096000e-02</td>
      <td>9.200000e-04</td>
      <td>8.930000e-03</td>
      <td>0.46</td>
      <td>1.300000e-01</td>
      <td>None</td>
      <td>com.comidaforfamilias.caloriehask:drawable/yogurt</td>
      <td>Food</td>
      <td>1 container: 170g, 1 cup: 245g</td>
    </tr>
    <tr>
      <th>842</th>
      <td>843</td>
      <td>Yogurt, vanilla, low fat</td>
      <td>0.85</td>
      <td>0.1380</td>
      <td>4.940656e-324</td>
      <td>0.1380</td>
      <td>0.0493</td>
      <td>1.250000e-02</td>
      <td>8.060000e-03</td>
      <td>3.600000e-04</td>
      <td>3.430000e-03</td>
      <td>0.66</td>
      <td>5.000000e-02</td>
      <td>None</td>
      <td>com.comidaforfamilias.caloriehask:drawable/yogurt</td>
      <td>Food</td>
      <td>1 container: 170g, 1 cup: 245g</td>
    </tr>
    <tr>
      <th>843</th>
      <td>844</td>
      <td>Yogurt, vanilla, non-fat</td>
      <td>0.78</td>
      <td>0.1704</td>
      <td>4.940656e-324</td>
      <td>0.0588</td>
      <td>0.0294</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>0.47</td>
      <td>3.000000e-02</td>
      <td>None</td>
      <td>com.comidaforfamilias.caloriehask:drawable/yogurt</td>
      <td>Food</td>
      <td>1 container: 170g, 1 cup: 245g</td>
    </tr>
    <tr>
      <th>844</th>
      <td>845</td>
      <td>Yam, boiled or baked</td>
      <td>1.16</td>
      <td>0.2748</td>
      <td>3.900000e-02</td>
      <td>0.0049</td>
      <td>0.0149</td>
      <td>1.400000e-03</td>
      <td>2.900000e-04</td>
      <td>6.000000e-04</td>
      <td>5.000000e-05</td>
      <td>0.08</td>
      <td>4.940656e-324</td>
      <td>51 (low)</td>
      <td>com.comidaforfamilias.caloriehask:drawable/yam</td>
      <td>Food</td>
      <td>1/2 cup cubes: 68g</td>
    </tr>
    <tr>
      <th>845</th>
      <td>846</td>
      <td>Zucchini, boiled</td>
      <td>0.15</td>
      <td>0.0269</td>
      <td>1.000000e-02</td>
      <td>0.0171</td>
      <td>0.0114</td>
      <td>3.600000e-03</td>
      <td>7.200000e-04</td>
      <td>1.510000e-03</td>
      <td>2.900000e-04</td>
      <td>0.03</td>
      <td>4.940656e-324</td>
      <td>15 (low)</td>
      <td>com.comidaforfamilias.caloriehask:drawable/zuc...</td>
      <td>Food</td>
      <td>1/2 cup mashed: 120g, 1 cup sliced: 180g</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 846 entries, 0 to 845
    Data columns (total 17 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   ID                   846 non-null    int64  
     1   food_name            846 non-null    object 
     2   calories             846 non-null    float64
     3   carbs                846 non-null    float64
     4   dietary_fibre        846 non-null    float64
     5   sugar                846 non-null    float64
     6   protein              846 non-null    float64
     7   fats                 846 non-null    float64
     8   saturated_fat        846 non-null    float64
     9   polyunsaturated_fat  846 non-null    float64
     10  monounsaturated_fat  846 non-null    float64
     11  sodium               846 non-null    float64
     12  cholesterol          846 non-null    float64
     13  glycemic_index       846 non-null    object 
     14  image                846 non-null    object 
     15  type                 846 non-null    object 
     16  quantities           845 non-null    object 
    dtypes: float64(11), int64(1), object(5)
    memory usage: 112.5+ KB
    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>calories</th>
      <th>carbs</th>
      <th>dietary_fibre</th>
      <th>sugar</th>
      <th>protein</th>
      <th>fats</th>
      <th>saturated_fat</th>
      <th>polyunsaturated_fat</th>
      <th>monounsaturated_fat</th>
      <th>sodium</th>
      <th>cholesterol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>846.000000</td>
      <td>8.460000e+02</td>
      <td>8.460000e+02</td>
      <td>8.460000e+02</td>
      <td>8.460000e+02</td>
      <td>8.460000e+02</td>
      <td>8.460000e+02</td>
      <td>8.460000e+02</td>
      <td>8.460000e+02</td>
      <td>8.460000e+02</td>
      <td>8.460000e+02</td>
      <td>8.460000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>423.500000</td>
      <td>2.293511e+00</td>
      <td>2.098634e-01</td>
      <td>2.718239e-02</td>
      <td>6.823900e-02</td>
      <td>1.120637e-01</td>
      <td>1.166775e-01</td>
      <td>3.293926e-02</td>
      <td>2.961825e-02</td>
      <td>4.446923e-02</td>
      <td>3.414255e+00</td>
      <td>3.056501e-01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>244.363459</td>
      <td>1.720517e+00</td>
      <td>2.311872e-01</td>
      <td>4.489292e-02</td>
      <td>1.369379e-01</td>
      <td>1.104459e-01</td>
      <td>1.672488e-01</td>
      <td>5.735277e-02</td>
      <td>6.399559e-02</td>
      <td>8.059633e-02</td>
      <td>1.428668e+01</td>
      <td>5.529062e-01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>212.250000</td>
      <td>8.425000e-01</td>
      <td>2.720000e-02</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>1.862500e-02</td>
      <td>5.325000e-03</td>
      <td>9.200000e-04</td>
      <td>1.452500e-03</td>
      <td>7.250000e-04</td>
      <td>1.000000e-01</td>
      <td>4.940656e-324</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>423.500000</td>
      <td>2.060000e+00</td>
      <td>1.319500e-01</td>
      <td>1.300000e-02</td>
      <td>1.975000e-02</td>
      <td>8.485000e-02</td>
      <td>5.915000e-02</td>
      <td>1.606000e-02</td>
      <td>7.510000e-03</td>
      <td>1.796500e-02</td>
      <td>7.850000e-01</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>634.750000</td>
      <td>3.090000e+00</td>
      <td>2.991500e-01</td>
      <td>3.300000e-02</td>
      <td>6.525000e-02</td>
      <td>1.827250e-01</td>
      <td>1.531500e-01</td>
      <td>4.149000e-02</td>
      <td>2.704500e-02</td>
      <td>5.412250e-02</td>
      <td>4.427500e+00</td>
      <td>5.500000e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>846.000000</td>
      <td>9.020000e+00</td>
      <td>9.998000e-01</td>
      <td>4.260000e-01</td>
      <td>9.980000e-01</td>
      <td>8.110000e-01</td>
      <td>1.000000e+00</td>
      <td>8.247500e-01</td>
      <td>6.990000e-01</td>
      <td>7.296100e-01</td>
      <td>3.875800e+02</td>
      <td>7.100000e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (846, 17)




```python
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>calories</th>
      <th>carbs</th>
      <th>dietary_fibre</th>
      <th>sugar</th>
      <th>protein</th>
      <th>fats</th>
      <th>saturated_fat</th>
      <th>polyunsaturated_fat</th>
      <th>monounsaturated_fat</th>
      <th>sodium</th>
      <th>cholesterol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ID</th>
      <td>1.000000</td>
      <td>-0.058599</td>
      <td>-0.011045</td>
      <td>0.014673</td>
      <td>-0.006776</td>
      <td>-0.027989</td>
      <td>-0.050375</td>
      <td>-0.135615</td>
      <td>0.093057</td>
      <td>-0.068097</td>
      <td>0.045568</td>
      <td>-0.060976</td>
    </tr>
    <tr>
      <th>calories</th>
      <td>-0.058599</td>
      <td>1.000000</td>
      <td>0.391108</td>
      <td>0.300849</td>
      <td>0.156680</td>
      <td>0.257943</td>
      <td>0.858892</td>
      <td>0.617144</td>
      <td>0.638449</td>
      <td>0.733660</td>
      <td>0.009330</td>
      <td>0.131186</td>
    </tr>
    <tr>
      <th>carbs</th>
      <td>-0.011045</td>
      <td>0.391108</td>
      <td>1.000000</td>
      <td>0.465955</td>
      <td>0.593079</td>
      <td>-0.258588</td>
      <td>-0.048570</td>
      <td>-0.057603</td>
      <td>0.036885</td>
      <td>-0.070076</td>
      <td>-0.015860</td>
      <td>-0.381133</td>
    </tr>
    <tr>
      <th>dietary_fibre</th>
      <td>0.014673</td>
      <td>0.300849</td>
      <td>0.465955</td>
      <td>1.000000</td>
      <td>0.082120</td>
      <td>-0.029324</td>
      <td>0.137369</td>
      <td>-0.016359</td>
      <td>0.206613</td>
      <td>0.114493</td>
      <td>-0.052501</td>
      <td>-0.302777</td>
    </tr>
    <tr>
      <th>sugar</th>
      <td>-0.006776</td>
      <td>0.156680</td>
      <td>0.593079</td>
      <td>0.082120</td>
      <td>1.000000</td>
      <td>-0.287023</td>
      <td>-0.075570</td>
      <td>0.008695</td>
      <td>-0.067690</td>
      <td>-0.086215</td>
      <td>-0.051474</td>
      <td>-0.220106</td>
    </tr>
    <tr>
      <th>protein</th>
      <td>-0.027989</td>
      <td>0.257943</td>
      <td>-0.258588</td>
      <td>-0.029324</td>
      <td>-0.287023</td>
      <td>1.000000</td>
      <td>0.157755</td>
      <td>0.126658</td>
      <td>0.046998</td>
      <td>0.151392</td>
      <td>0.006764</td>
      <td>0.449741</td>
    </tr>
    <tr>
      <th>fats</th>
      <td>-0.050375</td>
      <td>0.858892</td>
      <td>-0.048570</td>
      <td>0.137369</td>
      <td>-0.075570</td>
      <td>0.157755</td>
      <td>1.000000</td>
      <td>0.710193</td>
      <td>0.728704</td>
      <td>0.871327</td>
      <td>0.013859</td>
      <td>0.220909</td>
    </tr>
    <tr>
      <th>saturated_fat</th>
      <td>-0.135615</td>
      <td>0.617144</td>
      <td>-0.057603</td>
      <td>-0.016359</td>
      <td>0.008695</td>
      <td>0.126658</td>
      <td>0.710193</td>
      <td>1.000000</td>
      <td>0.256028</td>
      <td>0.469976</td>
      <td>0.028069</td>
      <td>0.302992</td>
    </tr>
    <tr>
      <th>polyunsaturated_fat</th>
      <td>0.093057</td>
      <td>0.638449</td>
      <td>0.036885</td>
      <td>0.206613</td>
      <td>-0.067690</td>
      <td>0.046998</td>
      <td>0.728704</td>
      <td>0.256028</td>
      <td>1.000000</td>
      <td>0.485628</td>
      <td>0.003696</td>
      <td>0.060922</td>
    </tr>
    <tr>
      <th>monounsaturated_fat</th>
      <td>-0.068097</td>
      <td>0.733660</td>
      <td>-0.070076</td>
      <td>0.114493</td>
      <td>-0.086215</td>
      <td>0.151392</td>
      <td>0.871327</td>
      <td>0.469976</td>
      <td>0.485628</td>
      <td>1.000000</td>
      <td>0.003758</td>
      <td>0.144841</td>
    </tr>
    <tr>
      <th>sodium</th>
      <td>0.045568</td>
      <td>0.009330</td>
      <td>-0.015860</td>
      <td>-0.052501</td>
      <td>-0.051474</td>
      <td>0.006764</td>
      <td>0.013859</td>
      <td>0.028069</td>
      <td>0.003696</td>
      <td>0.003758</td>
      <td>1.000000</td>
      <td>0.014929</td>
    </tr>
    <tr>
      <th>cholesterol</th>
      <td>-0.060976</td>
      <td>0.131186</td>
      <td>-0.381133</td>
      <td>-0.302777</td>
      <td>-0.220106</td>
      <td>0.449741</td>
      <td>0.220909</td>
      <td>0.302992</td>
      <td>0.060922</td>
      <td>0.144841</td>
      <td>0.014929</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
```


![svg](output_12_0.svg)


Green means positive, red means negative. The stronger the color, the larger the correlation magnitude.


```python
df.columns
```




    Index(['ID', 'food_name', 'calories', 'carbs', 'dietary_fibre', 'sugar',
           'protein', 'fats', 'saturated_fat', 'polyunsaturated_fat',
           'monounsaturated_fat', 'sodium', 'cholesterol', 'glycemic_index',
           'image', 'type', 'quantities'],
          dtype='object')




```python
df.drop(['type', 'image'], axis = 1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>food_name</th>
      <th>calories</th>
      <th>carbs</th>
      <th>dietary_fibre</th>
      <th>sugar</th>
      <th>protein</th>
      <th>fats</th>
      <th>saturated_fat</th>
      <th>polyunsaturated_fat</th>
      <th>monounsaturated_fat</th>
      <th>sodium</th>
      <th>cholesterol</th>
      <th>glycemic_index</th>
      <th>quantities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Abiyuch, raw</td>
      <td>0.69</td>
      <td>1.760000e-01</td>
      <td>5.300000e-02</td>
      <td>8.550000e-02</td>
      <td>1.500000e-02</td>
      <td>1.000000e-03</td>
      <td>1.400000e-04</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>2.000000e-01</td>
      <td>4.940656e-324</td>
      <td>None</td>
      <td>1/2 cup: 114g</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Adzuki beans, boiled</td>
      <td>1.28</td>
      <td>2.477000e-01</td>
      <td>7.300000e-02</td>
      <td>4.940656e-324</td>
      <td>7.520000e-02</td>
      <td>1.000000e-03</td>
      <td>3.600000e-04</td>
      <td>2.100000e-04</td>
      <td>9.000000e-05</td>
      <td>8.000000e-02</td>
      <td>4.940656e-324</td>
      <td>None</td>
      <td>1 cup: 230g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Almond butter</td>
      <td>6.14</td>
      <td>1.882000e-01</td>
      <td>1.030000e-01</td>
      <td>4.430000e-02</td>
      <td>2.096000e-01</td>
      <td>5.550000e-01</td>
      <td>4.152000e-02</td>
      <td>1.361300e-01</td>
      <td>3.244500e-01</td>
      <td>7.000000e-02</td>
      <td>4.940656e-324</td>
      <td>Low</td>
      <td>1 tbsp: 16g, 1 cup: 250g</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Almond flour</td>
      <td>5.71</td>
      <td>2.143000e-01</td>
      <td>1.070000e-01</td>
      <td>3.570000e-02</td>
      <td>2.143000e-01</td>
      <td>5.000000e-01</td>
      <td>1.000000e-02</td>
      <td>3.500000e-02</td>
      <td>8.999000e-02</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Almond oil</td>
      <td>8.84</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>1.000000e+00</td>
      <td>8.200000e-02</td>
      <td>1.740000e-01</td>
      <td>6.990000e-01</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>None</td>
      <td>1 tsp: 4.5g, 1 tbsp: 13.6g, 1 cup: 218g</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>841</th>
      <td>842</td>
      <td>Yogurt, plain, whole milk</td>
      <td>0.61</td>
      <td>4.660000e-02</td>
      <td>4.940656e-324</td>
      <td>4.660000e-02</td>
      <td>3.470000e-02</td>
      <td>3.250000e-02</td>
      <td>2.096000e-02</td>
      <td>9.200000e-04</td>
      <td>8.930000e-03</td>
      <td>4.600000e-01</td>
      <td>1.300000e-01</td>
      <td>None</td>
      <td>1 container: 170g, 1 cup: 245g</td>
    </tr>
    <tr>
      <th>842</th>
      <td>843</td>
      <td>Yogurt, vanilla, low fat</td>
      <td>0.85</td>
      <td>1.380000e-01</td>
      <td>4.940656e-324</td>
      <td>1.380000e-01</td>
      <td>4.930000e-02</td>
      <td>1.250000e-02</td>
      <td>8.060000e-03</td>
      <td>3.600000e-04</td>
      <td>3.430000e-03</td>
      <td>6.600000e-01</td>
      <td>5.000000e-02</td>
      <td>None</td>
      <td>1 container: 170g, 1 cup: 245g</td>
    </tr>
    <tr>
      <th>843</th>
      <td>844</td>
      <td>Yogurt, vanilla, non-fat</td>
      <td>0.78</td>
      <td>1.704000e-01</td>
      <td>4.940656e-324</td>
      <td>5.880000e-02</td>
      <td>2.940000e-02</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.940656e-324</td>
      <td>4.700000e-01</td>
      <td>3.000000e-02</td>
      <td>None</td>
      <td>1 container: 170g, 1 cup: 245g</td>
    </tr>
    <tr>
      <th>844</th>
      <td>845</td>
      <td>Yam, boiled or baked</td>
      <td>1.16</td>
      <td>2.748000e-01</td>
      <td>3.900000e-02</td>
      <td>4.900000e-03</td>
      <td>1.490000e-02</td>
      <td>1.400000e-03</td>
      <td>2.900000e-04</td>
      <td>6.000000e-04</td>
      <td>5.000000e-05</td>
      <td>8.000000e-02</td>
      <td>4.940656e-324</td>
      <td>51 (low)</td>
      <td>1/2 cup cubes: 68g</td>
    </tr>
    <tr>
      <th>845</th>
      <td>846</td>
      <td>Zucchini, boiled</td>
      <td>0.15</td>
      <td>2.690000e-02</td>
      <td>1.000000e-02</td>
      <td>1.710000e-02</td>
      <td>1.140000e-02</td>
      <td>3.600000e-03</td>
      <td>7.200000e-04</td>
      <td>1.510000e-03</td>
      <td>2.900000e-04</td>
      <td>3.000000e-02</td>
      <td>4.940656e-324</td>
      <td>15 (low)</td>
      <td>1/2 cup mashed: 120g, 1 cup sliced: 180g</td>
    </tr>
  </tbody>
</table>
<p>846 rows × 15 columns</p>
</div>




```python
# Step 1 - Make a scatter plot with square markers, set column names as labels

def heatmap(x, y, size):
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    
df = pd.read_csv('food_table.csv')
columns = ['fats', 'cholesterol', 'carbs', 'sugar', 'protein'] 
corr = df[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)
```


![svg](output_16_0.svg)



```python
x = df['protein']
y = df['carbs']
```


```python
plt.scatter(x, y) 
plt.show()
```


![svg](output_18_0.svg)


 created one more feature called food_categories because when examining carefully that name feature, the first word before the comma would be the food.


```python
df['food_categories'] = df['food_name'].apply(lambda x: x.split(',')[0])
```


```python
#If we try to visualize the columns one by one, it would be massive and kinda repetitive as it would not give us much information. You could try it thou if you want. I could give you the code below.
for i in df.select_dtypes('number').columns:
    sns.distplot(df[i])
    plt.title(i)
    plt.show()
```

    C:\Python38\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![svg](output_21_1.svg)


    C:\Python38\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![svg](output_21_3.svg)


    C:\Python38\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![svg](output_21_5.svg)


    C:\Python38\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![svg](output_21_7.svg)


    C:\Python38\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![svg](output_21_9.svg)


    C:\Python38\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![svg](output_21_11.svg)


    C:\Python38\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![svg](output_21_13.svg)


    C:\Python38\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![svg](output_21_15.svg)


    C:\Python38\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![svg](output_21_17.svg)


    C:\Python38\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![svg](output_21_19.svg)


    C:\Python38\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![svg](output_21_21.svg)


    C:\Python38\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![svg](output_21_23.svg)


As we know mean is the average of the data. Multiple features could have the same mean, but different in how they are spread around the mean and it signifies by the standard deviation (std). There is a rule called an empirical rule where we could get the probability of the data spreads via standard deviation. The empirical rule stated that:

68% of our data falls under mean±1*std

95% of our data falls under mean±2*std

99.7% of our data falls under mean±3*std

Empirical rule or some also say 68–95–99.7 rule are often used to analyzing the data outlier.
The main problem with this statistic is that they are affected by outlier or extreme value(s) and often causing the data to be skewed. I show you with an image what is skewed data.

Above is the plot of the total_fat(g) feature. It is skewed right as the tail is on the right. But, how skewed is the skewness? It is the purpose of the skew statistic. Some rule we could remember about skewness are:

If the skewness is between -0.5 and 0.5, the data are fairly symmetrical

If the skewness is between -1 and — 0.5 or between 0.5 and 1, the data are moderately skewed

If the skewness is less than -1 or greater than 1, the data are highly skewed

So we could see that if our data above is highly skewed, which actually most of the data that you would encounter is like that. Now, what about kurtosis? What is this statistic tell us? Kurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. The analysis could be summarized below:

If the kurtosis is close to 0, then a normal distribution is often assumed. These are called mesokurtic distributions.

If the kurtosis is less than 0, then the distribution is light tails and is called a platykurtic distribution.

If the kurtosis is greater than 0, then the distribution has heavier tails and is called a leptokurtic distribution.


To be precise what we have before is called Excess Kurtosis, where normal distribution is measured in kurtosis as 0. If we only talk about Kurtosis, the normal distribution would be equal to 3 so that is why in Excess Kurtosis we subtract the kurtosis by 3.
Turn out most of our data are skewed. Skewed data are actually really interesting as you could try to explore with it. For example, what food is considered to be an outlier based on the Calories.
As our data is skewed enough, I would not rely on the mean to find the outlier; instead, I would apply the IQR method which is based on the median.

IQR or Interquartile Range is based on the data position. For example, if we describe the ‘calories’ feature we would get the description below.


```python
#I use the .agg method of the DataFrame to gain information about the mean, 
#median, std, skewness and kurtosis of each column. This is where the number speak more than visualization
pd.set_option('display.max_columns', None)
df.agg(['mean', 'median', 'std', 'skew', 'kurtosis'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>calories</th>
      <th>carbs</th>
      <th>dietary_fibre</th>
      <th>sugar</th>
      <th>protein</th>
      <th>fats</th>
      <th>saturated_fat</th>
      <th>polyunsaturated_fat</th>
      <th>monounsaturated_fat</th>
      <th>sodium</th>
      <th>cholesterol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>423.500000</td>
      <td>2.293511</td>
      <td>0.209863</td>
      <td>0.027182</td>
      <td>0.068239</td>
      <td>0.112064</td>
      <td>0.116678</td>
      <td>0.032939</td>
      <td>0.029618</td>
      <td>0.044469</td>
      <td>3.414255</td>
      <td>3.056501e-01</td>
    </tr>
    <tr>
      <th>median</th>
      <td>423.500000</td>
      <td>2.060000</td>
      <td>0.131950</td>
      <td>0.013000</td>
      <td>0.019750</td>
      <td>0.084850</td>
      <td>0.059150</td>
      <td>0.016060</td>
      <td>0.007510</td>
      <td>0.017965</td>
      <td>0.785000</td>
      <td>4.940656e-324</td>
    </tr>
    <tr>
      <th>std</th>
      <td>244.363459</td>
      <td>1.720517</td>
      <td>0.231187</td>
      <td>0.044893</td>
      <td>0.136938</td>
      <td>0.110446</td>
      <td>0.167249</td>
      <td>0.057353</td>
      <td>0.063996</td>
      <td>0.080596</td>
      <td>14.286683</td>
      <td>5.529062e-01</td>
    </tr>
    <tr>
      <th>skew</th>
      <td>0.000000</td>
      <td>1.071669</td>
      <td>1.216123</td>
      <td>3.989044</td>
      <td>3.626230</td>
      <td>1.374615</td>
      <td>2.589782</td>
      <td>5.807189</td>
      <td>4.649364</td>
      <td>4.071530</td>
      <td>23.426311</td>
      <td>4.499850e+00</td>
    </tr>
    <tr>
      <th>kurtosis</th>
      <td>-1.200000</td>
      <td>1.258618</td>
      <td>0.526064</td>
      <td>23.757815</td>
      <td>15.010375</td>
      <td>3.298728</td>
      <td>8.145578</td>
      <td>57.131978</td>
      <td>28.609725</td>
      <td>21.774605</td>
      <td>621.724136</td>
      <td>3.785830e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['calories'].describe()
```




    count     8.460000e+02
    mean      2.293511e+00
    std       1.720517e+00
    min      4.940656e-324
    25%       8.425000e-01
    50%       2.060000e+00
    75%       3.090000e+00
    max       9.020000e+00
    Name: calories, dtype: float64



IQR would be based on the 25% position or Q1 and 75% position or Q3. We also get an IQR value by subtracting Q3 with Q1 (Q3-Q1). With the IQR method, we could decide which data are considered an outlier based on the upper or lower limit which is:

Lower Limit= Q1–1.5 * IQR

Upper Limit= Q3 + 1.5 * IQR

Any data above or below this limit would be considered as an outlier. Let’s try to implement this method and let’s see what kind of food is considered to be an outlier based on calories.


```python
#Specifying the limit
cal_Q1 = df.describe()['calories']['25%']
cal_Q3 = df.describe()['calories']['75%']
cal_IQR = cal_Q3 - cal_Q1
df[(df['calories'] < 1.5 * (cal_Q1 - cal_IQR)) | (df['calories'] > 1.5 * (cal_Q3 + cal_IQR)) ]['food_categories'].value_counts()
```




    Fish oil       2
    Olive oil      1
    Coconut oil    1
    Lard           1
    Almond oil     1
    Oil            1
    Name: food_categories, dtype: int64




```python
pip install python-dotenv

```

    Requirement already satisfied: python-dotenv in c:\python38\lib\site-packages (0.19.0)
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    


```python
pip install https://github.com/iCorv/tflite-runtime/raw/master/tflite_runtime-2.4.0-py3-none-any.whl
```

    Collecting tflite-runtime==2.4.0
      Using cached https://github.com/iCorv/tflite-runtime/raw/master/tflite_runtime-2.4.0-py3-none-any.whl (1.5 MB)
    Requirement already satisfied: numpy>=1.16.0 in c:\python38\lib\site-packages (from tflite-runtime==2.4.0) (1.19.5)
    Requirement already satisfied: pybind11>=2.4.3 in c:\python38\lib\site-packages (from tflite-runtime==2.4.0) (2.7.1)
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    


```python
pip install tensorflow
```

    Collecting tensorflow
      Using cached tensorflow-2.6.0-cp38-cp38-win_amd64.whl (423.2 MB)
    Collecting flatbuffers~=1.12.0
      Using cached flatbuffers-1.12-py2.py3-none-any.whl (15 kB)
    Collecting h5py~=3.1.0
      Using cached h5py-3.1.0-cp38-cp38-win_amd64.whl (2.7 MB)
    Collecting grpcio<2.0,>=1.37.0
      Using cached grpcio-1.39.0-cp38-cp38-win_amd64.whl (3.2 MB)
    Requirement already satisfied: numpy~=1.19.2 in c:\python38\lib\site-packages (from tensorflow) (1.19.5)
    Requirement already satisfied: wheel~=0.35 in c:\python38\lib\site-packages (from tensorflow) (0.37.0)
    Collecting typing-extensions~=3.7.4
      Using cached typing_extensions-3.7.4.3-py3-none-any.whl (22 kB)
    Collecting astunparse~=1.6.3
      Using cached astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Collecting keras-preprocessing~=1.1.2
      Using cached Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42 kB)
    Collecting clang~=5.0
      Using cached clang-5.0.tar.gz (30 kB)
    Collecting absl-py~=0.10
      Using cached absl_py-0.13.0-py3-none-any.whl (132 kB)
    Collecting google-pasta~=0.2
      Using cached google_pasta-0.2.0-py3-none-any.whl (57 kB)
    Collecting keras~=2.6
      Using cached keras-2.6.0-py2.py3-none-any.whl (1.3 MB)
    Collecting tensorboard~=2.6
      Using cached tensorboard-2.6.0-py3-none-any.whl (5.6 MB)
    Requirement already satisfied: protobuf>=3.9.2 in c:\python38\lib\site-packages (from tensorflow) (3.17.3)
    Collecting wrapt~=1.12.1
      Using cached wrapt-1.12.1.tar.gz (27 kB)
    Requirement already satisfied: six~=1.15.0 in c:\python38\lib\site-packages (from tensorflow) (1.15.0)
    Collecting termcolor~=1.1.0
      Using cached termcolor-1.1.0.tar.gz (3.9 kB)
    Collecting tensorflow-estimator~=2.6
      Using cached tensorflow_estimator-2.6.0-py2.py3-none-any.whl (462 kB)
    Collecting gast==0.4.0
      Using cached gast-0.4.0-py3-none-any.whl (9.8 kB)
    Collecting opt-einsum~=3.3.0
      Using cached opt_einsum-3.3.0-py3-none-any.whl (65 kB)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in c:\python38\lib\site-packages (from tensorboard~=2.6->tensorflow) (0.6.1)
    Requirement already satisfied: google-auth<2,>=1.6.3 in c:\python38\lib\site-packages (from tensorboard~=2.6->tensorflow) (1.35.0)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\python38\lib\site-packages (from tensorboard~=2.6->tensorflow) (1.8.0)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\python38\lib\site-packages (from tensorboard~=2.6->tensorflow) (2.24.0)
    Requirement already satisfied: werkzeug>=0.11.15 in c:\python38\lib\site-packages (from tensorboard~=2.6->tensorflow) (1.0.1)
    Collecting google-auth-oauthlib<0.5,>=0.4.1
      Downloading google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
    Collecting markdown>=2.6.8
      Using cached Markdown-3.3.4-py3-none-any.whl (97 kB)
    Requirement already satisfied: setuptools>=41.0.0 in c:\python38\lib\site-packages (from tensorboard~=2.6->tensorflow) (47.1.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\python38\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard~=2.6->tensorflow) (0.2.8)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in c:\python38\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard~=2.6->tensorflow) (4.2.2)
    Requirement already satisfied: rsa<5,>=3.1.4 in c:\python38\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard~=2.6->tensorflow) (4.7.2)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\python38\lib\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.6->tensorflow) (1.3.0)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\python38\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<2,>=1.6.3->tensorboard~=2.6->tensorflow) (0.4.8)
    Requirement already satisfied: chardet<4,>=3.0.2 in c:\python38\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in c:\python38\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in c:\python38\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow) (2020.6.20)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\python38\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow) (1.25.10)
    Requirement already satisfied: oauthlib>=3.0.0 in c:\python38\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.6->tensorflow) (3.1.1)
    Building wheels for collected packages: clang, termcolor, wrapt
      Building wheel for clang (setup.py): started
      Building wheel for clang (setup.py): finished with status 'done'
      Created wheel for clang: filename=clang-5.0-py3-none-any.whl size=30703 sha256=a2c01f200f2b612598645dbdcac9e80617f3c54a07e9acbe31cf6322a0f00611
      Stored in directory: c:\users\keyskull\appdata\local\pip\cache\wheels\f1\60\77\22b9b5887bd47801796a856f47650d9789c74dc3161a26d608
      Building wheel for termcolor (setup.py): started
      Building wheel for termcolor (setup.py): finished with status 'done'
      Created wheel for termcolor: filename=termcolor-1.1.0-py3-none-any.whl size=4829 sha256=34c2b9e75353a237528e20b0e90f13b3bb780b4bbc9b3a8370bebac56c987b01
      Stored in directory: c:\users\keyskull\appdata\local\pip\cache\wheels\a0\16\9c\5473df82468f958445479c59e784896fa24f4a5fc024b0f501
      Building wheel for wrapt (setup.py): started
      Building wheel for wrapt (setup.py): finished with status 'done'
      Created wheel for wrapt: filename=wrapt-1.12.1-cp38-cp38-win_amd64.whl size=33698 sha256=ab711aed5a05c2ec3e6fd38c444c545c52dc6d1e1f0cdfbdf74c2b675c8bd0c9
      Stored in directory: c:\users\keyskull\appdata\local\pip\cache\wheels\5f\fd\9e\b6cf5890494cb8ef0b5eaff72e5d55a70fb56316007d6dfe73
    Successfully built clang termcolor wrapt
    Installing collected packages: markdown, grpcio, google-auth-oauthlib, absl-py, wrapt, typing-extensions, termcolor, tensorflow-estimator, tensorboard, opt-einsum, keras-preprocessing, keras, h5py, google-pasta, gast, flatbuffers, clang, astunparse, tensorflow
    Successfully installed absl-py-0.13.0 astunparse-1.6.3 clang-5.0 flatbuffers-1.12 gast-0.4.0 google-auth-oauthlib-0.4.6 google-pasta-0.2.0 grpcio-1.39.0 h5py-3.1.0 keras-2.6.0 keras-preprocessing-1.1.2 markdown-3.3.4 opt-einsum-3.3.0 tensorboard-2.6.0 tensorflow-2.6.0 tensorflow-estimator-2.6.0 termcolor-1.1.0 typing-extensions-3.7.4.3 wrapt-1.12.1
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    


```python
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
from PIL import Image
import tensorflow as tf
import tflite_runtime.interpreter as tflite


class Classifier:
    def __init__(self, model_path, label_file, input_mean=127.5, input_std=127.5):
        self.input_mean = input_mean
        self.input_std = input_std
        self.labels = self.load_labels(label_file)

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # check the type of the input tensor
        self.floating_model = self.input_details[0]['dtype'] == np.float32

        # NxHxWxC, H:1, W:2
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

    def load_labels(self, filename):
        with open(filename, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def infer(self, image):
        """
        Infers the image in tf lite
        Args:
            image (string): Path of the image            
        Returns:
            label (string): Detected label of the Image
        """
        img = Image.open(image).resize((self.width, self.height))

        # add N dim
        input_data = np.expand_dims(img, axis=0)

        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        results = np.squeeze(output_data)

        i = results.argmax()

        if not results[i] > 0.5:
            return None
        
        idx = self.labels[i].index(" ")
        return self.labels[i][idx+1:]
```


```python
pip install --user caloriestracker
```

    Requirement already satisfied: caloriestracker in c:\users\keyskull\appdata\roaming\python\python38\site-packages (0.8.0)
    Requirement already satisfied: colorama in c:\python38\lib\site-packages (from caloriestracker) (0.4.3)
    Requirement already satisfied: pywin32 in c:\python38\lib\site-packages (from caloriestracker) (228)
    Requirement already satisfied: pytz in c:\python38\lib\site-packages (from caloriestracker) (2020.1)
    Requirement already satisfied: PyQtWebEngine in c:\users\keyskull\appdata\roaming\python\python38\site-packages (from caloriestracker) (5.15.4)
    Requirement already satisfied: PyQt5 in c:\users\keyskull\appdata\roaming\python\python38\site-packages (from caloriestracker) (5.15.4)
    Requirement already satisfied: PyQtChart in c:\users\keyskull\appdata\roaming\python\python38\site-packages (from caloriestracker) (5.15.4)
    Requirement already satisfied: setuptools in c:\python38\lib\site-packages (from caloriestracker) (47.1.0)
    Requirement already satisfied: psycopg2 in c:\users\keyskull\appdata\roaming\python\python38\site-packages (from caloriestracker) (2.9.1)
    Requirement already satisfied: officegenerator in c:\users\keyskull\appdata\roaming\python\python38\site-packages (from caloriestracker) (1.28.0)
    Requirement already satisfied: openpyxl in c:\users\keyskull\appdata\roaming\python\python38\site-packages (from officegenerator->caloriestracker) (3.0.7)
    Requirement already satisfied: odfpy==1.3.6 in c:\users\keyskull\appdata\roaming\python\python38\site-packages (from officegenerator->caloriestracker) (1.3.6)
    Requirement already satisfied: et-xmlfile in c:\users\keyskull\appdata\roaming\python\python38\site-packages (from openpyxl->officegenerator->caloriestracker) (1.1.0)
    Requirement already satisfied: PyQt5-Qt5>=5.15 in c:\users\keyskull\appdata\roaming\python\python38\site-packages (from PyQt5->caloriestracker) (5.15.2)
    Requirement already satisfied: PyQt5-sip<13,>=12.8 in c:\users\keyskull\appdata\roaming\python\python38\site-packages (from PyQt5->caloriestracker) (12.9.0)
    Requirement already satisfied: PyQtChart-Qt5>=5.15 in c:\users\keyskull\appdata\roaming\python\python38\site-packages (from PyQtChart->caloriestracker) (5.15.2)
    Requirement already satisfied: PyQtWebEngine-Qt5>=5.15 in c:\users\keyskull\appdata\roaming\python\python38\site-packages (from PyQtWebEngine->caloriestracker) (5.15.2)
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    WARNING: Ignoring invalid distribution -ip (c:\python38\lib\site-packages)
    


```python
conda uninstall pillow
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-29-9f3f9b88bb3a> in <module>
    ----> 1 get_ipython().run_line_magic('conda', 'uninstall pillow')
    

    C:\Python38\lib\site-packages\IPython\core\interactiveshell.py in run_line_magic(self, magic_name, line, _stack_depth)
       2324                 kwargs['local_ns'] = self.get_local_scope(stack_depth)
       2325             with self.builtin_trap:
    -> 2326                 result = fn(*args, **kwargs)
       2327             return result
       2328 
    

    <decorator-gen-108> in conda(self, line)
    

    C:\Python38\lib\site-packages\IPython\core\magic.py in <lambda>(f, *a, **k)
        185     # but it's overkill for just that one bit of state.
        186     def magic_deco(arg):
    --> 187         call = lambda f, *a, **k: f(*a, **k)
        188 
        189         if callable(arg):
    

    C:\Python38\lib\site-packages\IPython\core\magics\packaging.py in conda(self, line)
         77         """
         78         if not _is_conda_environment():
    ---> 79             raise ValueError("The python kernel does not appear to be a conda environment.  "
         80                              "Please use ``%pip install`` instead.")
         81 
    

    ValueError: The python kernel does not appear to be a conda environment.  Please use ``%pip install`` instead.



```python
from PyQt5.QtWidgets import  QDialog
from caloriestracker.ui.frmAccess import frmAccess
from caloriestracker.mem import MemCaloriestracker
import caloriestracker.images.caloriestracker_rc #Images of frmAccess were not loaded without this

from os import system, environ

from PyQt5 import QtWebEngineWidgets # To avoid error must be imported before QCoreApplication
dir(QtWebEngineWidgets)
print ("This script needs PGPASSWORD to be set")

password=environ['PGPASSWORD']

system("dropdb -U postgres -h 127.0.0.1 caloriestracker_autotest")

print("Emulating caloriestracker_init main function")

mem=MemCaloriestracker()
mem.run()
mem.frmAccess=frmAccess("caloriestracker", "frmAccess")
mem.frmAccess.setResources(":/caloriestracker/caloriestracker.png", ":/caloriestracker/caloriestracker.png")
mem.frmAccess.setLabel(mem.tr("Please login to the Calories Tracker database"))
mem.frmAccess.txtDB.setText("caloriestracker_autotest")
mem.frmAccess.txtPass.setText(password)
mem.frmAccess.on_cmdDatabaseNew_released()

print("You must select yes and ok to message")

print ("Emulating caloriestracker main function")

del mem
mem=MemCaloriestracker()
mem.run()
mem.frmAccess=frmAccess("caloriestracker", "frmAccess")
mem.frmAccess.setResources(":/caloriestracker/caloriestracker.png", ":/caloriestracker/caloriestracker.png")
mem.frmAccess.setLabel(mem.tr("Please login to the Calories Tracker database"))
mem.frmAccess.txtDB.setText("caloriestracker_autotest")
mem.frmAccess.txtPass.setText(password)
mem.frmAccess.on_cmdYN_accepted()


if mem.frmAccess.result()==QDialog.Accepted:
    mem.con=mem.frmAccess.con
    mem.settings=mem.frmAccess.settings
    mem.setLocalzone()#Needs settings in mem
    if mem.args.products_maintainer==True:
        from caloriestracker.ui.frmMainProductsMaintainer import frmMainProductsMaintainer
        mem.setProductsMaintainerMode(True)
        mem.frmAccess.languages.cambiar("en", "caloriestracker")
        mem.frmMain = frmMainProductsMaintainer(mem)
    else:
        from caloriestracker.ui.frmMain import frmMain
        mem.frmMain=frmMain(mem)
    mem.frmMain.show()
```

    This script needs PGPASSWORD to be set
    


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-1-b9dbeb24e066> in <module>
         10 print ("This script needs PGPASSWORD to be set")
         11 
    ---> 12 password=environ['PGPASSWORD']
         13 
         14 system("dropdb -U postgres -h 127.0.0.1 caloriestracker_autotest")
    

    C:\Python38\lib\os.py in __getitem__(self, key)
        673         except KeyError:
        674             # raise KeyError with the original key value
    --> 675             raise KeyError(key) from None
        676         return self.decodevalue(value)
        677 
    

    KeyError: 'PGPASSWORD'


### **4. Data Preparation**

4.1 Reorganize Dataset
4.2 Standardize Data
4.3 Null Values Checking
4.4 Outliers Checking
4.5 Correlation and Regression Analysis
4.6 Data Evaluation
Algorithm Implementation and Evaluation
Conclusion
6.1 Answers of Research Questions
6.2 Feasibility Conclusion
6.3 Improval Suggestions
