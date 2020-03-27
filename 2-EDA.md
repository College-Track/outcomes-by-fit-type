---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Outcomes By Fit Type

FY20 Learning Agenda to evaluate if the fit type of a school influences the outcomes for a student.

This notebook contains basic statistical analysis and visualization of the data.

### Data Sources
- summary : Processed file from notebook 1-Data_Prep

### Changes
- 03-19-2020 : Started project

```python
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dython.nominal import correlation_ratio, associations
import warnings

import statsmodels.formula.api as smf
import statsmodels.api as sm

# warnings.filterwarnings('ignore')


```

```python
%matplotlib inline

```

### File Locations

```python
today = datetime.today()
in_file = Path.cwd() / "data" / "processed" / "processed_data.pkl"
report_dir = Path.cwd() / "reports"
report_file = report_dir / "Excel_Analysis_{today:%b-%d-%Y}.xlsx"

in_file2 = Path.cwd() / "data" / "processed" / "processed_data_file2.pkl"

in_file3 = Path.cwd() / "data" / "processed" / "processed_data_file3.pkl"

```

```python
df = pd.read_pickle(in_file)

df2 = pd.read_pickle(in_file2)

# df3 = pd.read_pickle(in_file3)
```

```python
def local_or_best(fit_type):
    if fit_type == "Local Affordable" or fit_type == "Best Fit":
        return "Local / Best"
    else:
        return fit_type
```

```python
df['hybrid_fit_type'] = df.apply(
    lambda x: local_or_best(x['updated_fit_type']), axis=1)
```

```python
df['best_fit'] =  np.where(df['updated_fit_type'] == 'Best Fit', True, False)
```

```python
df['good_fit'] =  np.where(df['updated_fit_type'] == 'Good Fit', True, False)
```

```python
df['local_affordable'] =  np.where(df['updated_fit_type'] == 'Local Affordable', True, False)
df['none_fit'] =  np.where(df['updated_fit_type'] == 'None', True, False)
```

### Perform Data Analysis

```python
# Because some data points are limited for historical classes,
# setting up data frames to deal with various historical limitations 

df_matriculation_data = df[df.high_school_class >= 2015]

df_4_year_grades = df[df.high_school_class <= 2015]
df_5_year_grades = df[df.high_school_class <= 2014]
df_6_year_grades = df[df.high_school_class <= 2013]
```

<!-- #region heading_collapsed=true -->
#### Looking at Basic Distributions
<!-- #endregion -->

```python hidden=true
pd.crosstab(df.updated_fit_type, df.indicator_persisted_into_year_2_ct,
            normalize='index')
```

```python hidden=true
pd.crosstab(df_matriculation_data.updated_fit_type,
            df_matriculation_data.indicator_college_matriculation, normalize='index', margins=True)
```

```python hidden=true
pd.crosstab(df_5_year_grades.updated_fit_type,
            df_5_year_grades.graduated_4_year_degree, normalize='index', margins=True)
```

```python hidden=true
pd.crosstab(df_5_year_grades.updated_fit_type,
            df_5_year_grades.graduated_4_year_degree, normalize='index', margins=True)
```

```python hidden=true
pd.crosstab(df_5_year_grades.school,
            df_5_year_grades.graduated_4_year_degree, normalize=False, margins=True).sort_values(by=True, ascending=False)
```

```python hidden=true
pd.crosstab(df_6_year_grades.updated_fit_type,
            df_6_year_grades.graduated_4_year_degree, normalize=False, margins=True)
```

```python hidden=true
pd.crosstab(df_5_year_grades.updated_fit_type,
            df_5_year_grades.graduated_4_year_degree_less_5_years, normalize='index', margins=True)
```

```python hidden=true
pd.crosstab(df_5_year_grades.updated_fit_type,
            df_5_year_grades.graduated_4_year_degree_less_5_years, normalize=False, margins=True)
```

```python hidden=true
pd.crosstab(df_4_year_grades.updated_fit_type,
            df_4_year_grades.graduated_4_year_degree_less_4_years, normalize='index', margins=True)
```

```python hidden=true
pd.crosstab(df_4_year_grades.updated_fit_type,
            df_4_year_grades.graduated_4_year_degree_less_4_years, normalize=False, margins=True)
```

```python hidden=true
pd.crosstab(df.high_school_class, df.updated_fit_type, normalize='index')
```

```python hidden=true
pd.crosstab(df.high_school_class, df.updated_fit_type, margins=True)
```

#### Regressions / Correlations


##### Correlations

```python
associations(df_5_year_grades[
    ['graduated_4_year_degree', 'indicator_first_generation',
     'indicator_low_income', '11th_grade_gpa_bucket', 'ethnic_background',
     'updated_fit_type', 'year_1_gpa_bucket', 'college_elig_gpa_11th_cgpa',
     'school', 'hybrid_fit_type', 'best_fit', 'good_fit','local_affordable','none_fit', 'indicator_persisted_into_year_2_ct']
], theil_u=True,cmap="RdBu", figsize=(12,10))
```

```python

associations(df_4_year_grades[
    ['indicator_college_matriculation', 'indicator_persisted_into_year_2_ct', 'indicator_first_generation',
     'indicator_low_income', '11th_grade_gpa_bucket', 'ethnic_background',
     'updated_fit_type', 'year_1_gpa_bucket', 'college_elig_gpa_11th_cgpa',
     'school', 'hybrid_fit_type', 'best_fit', 'good_fit','local_affordable','none_fit']
], theil_u=True,cmap="RdBu", figsize=(12,10))
```

##### Logit Full DF 5 Year Grad Rate ~ All indicators 

```python
def C1(cat):
     return pd.get_dummies(cat, drop_first=True)
```

```python
mod = smf.logit(formula= "C1(graduated_4_year_degree_less_5_years) ~ year_1  + Q('11th_grade') + C(indicator_first_generation, Treatment) + C(updated_fit_type, Sum) + C(indicator_low_income) + C(indicator_persisted_into_year_2_ct)", data=df_5_year_grades).fit(method='bfgs', maxiter=100)
mod.summary()
```

<!-- #region heading_collapsed=true -->
##### Logit Best Fit Attend  5 Year Grad Rate ~ School

<!-- #endregion -->

```python hidden=true
df_5_year_grades_best_fit = df_5_year_grades[df_5_year_grades.best_fit == True]
```

```python hidden=true
mod = smf.logit(formula="C1(graduated_4_year_degree_less_5_years) ~ C(school, Sum)",
                data=df_5_year_grades_best_fit).fit(method='nm', maxiter=100000)
mod.summary()
```

<!-- #region heading_collapsed=true -->
##### Logit Above 3.0 GPA  5 Year Grad Rate ~ All Indicators

<!-- #endregion -->

```python hidden=true
df_5_year_grades_above_3_gpa = df_5_year_grades[df_5_year_grades.year_1 >= 3]
```

```python hidden=true
mod = smf.logit(formula= "C1(graduated_4_year_degree_less_5_years) ~ year_1  + Q('11th_grade') + C(indicator_first_generation, Treatment) + C(updated_fit_type, Sum) + C(indicator_low_income) + C(indicator_persisted_into_year_2_ct)", data=df_5_year_grades_above_3_gpa).fit(method='bfgs', maxiter=100)
mod.summary()
```

##### Logit Above 3.0 GPA  5 Year Grad Rate ~ School


```python
mod = smf.logit(formula="C1(graduated_4_year_degree_less_5_years) ~ C(school, Sum)",
                data=df_5_year_grades_above_3_gpa).fit(method='nm', maxiter=100000)
mod.summary()
```

##### Logit Below 3.0 GPA  5 Year Grad Rate ~ All Indicators


```python
df_5_year_grades_below_3_gpa = df_5_year_grades[df_5_year_grades.year_1 <= 2.75]
```

```python
mod = smf.logit(formula= "C1(graduated_4_year_degree_less_5_years) ~ year_1  + Q('11th_grade') + C(indicator_first_generation, Treatment) + C(updated_fit_type, Sum) + C(indicator_low_income) + C(indicator_persisted_into_year_2_ct)", data=df_5_year_grades_below_3_gpa).fit(method='bfgs', maxiter=100)
mod.summary()

```

```python
###
```

```python
np.exp(mod.params)
```

```python
mod = smf.ols(formula="year_1 ~ C(updated_fit_type, Sum) + C(indicator_first_generation, Treatment) + C(indicator_low_income) + Q('11th_grade')", data=df_5_year_grades_below_3_gpa).fit()
mod.summary()

```

```python
mod = smf.logit(formula= "C1(graduated_4_year_degree_less_5_years) ~ year_1  + Q('11th_grade') + C(indicator_first_generation, Treatment) + C(indicator_low_income) + C(indicator_persisted_into_year_2_ct) + C(good_fit, Treatment) +C(none_fit, Treatment) + C(best_fit, Treatment)", data=df_5_year_grades_below_3_gpa).fit(method='bfgs', maxiter=100)
mod.summary()


```

```python
np.exp(mod.params)
```

##### Logit Below 3.0 GPA  5 Year Grad Rate ~ School


```python
(df_4_year_grades[df_4_year_grades.best_fit == True].school.value_counts()>5).value_counts()
```

```python
mod = smf.logit(formula= "C1(graduated_4_year_degree) ~ year_1  + Q('11th_grade') + C(indicator_first_generation, Treatment) + C(best_fit, Treatment)  + C(good_fit, Sum) +C(none_fit, Treatment) + C(local_affordable, Treatment) + C(indicator_low_income) + C(indicator_persisted_into_year_2_ct)", data=df_6_year_grades).fit(method='bfgs', maxiter=100)
mod.summary()
```

```python
mod = smf.logit(formula= "C1(indicator_persisted_into_year_2_ct) ~ year_1  + Q('11th_grade') + C(indicator_first_generation, Treatment) + C(updated_fit_type, Treatment)", data=df_4_year_grades).fit(method='bfgs', maxiter=100)
mod.summary()

```

```python
mod = smf.logit(formula= "C1(graduated_4_year_degree_less_5_years) ~ year_1  + Q('11th_grade') + C(indicator_first_generation, Treatment) + C(updated_fit_type, Sum)", data=df_5_year_grades_above_3_gpa).fit(method='bfgs', maxiter=100)
mod.summary()

```

```python
mod = smf.logit(formula="C1(graduated_4_year_degree_less_5_years) ~ C(school, Sum)",
                data=df_5_year_grades_above_3_gpa).fit(method='nm', maxiter=100000)
mod.summary()
```

```python
mod = smf.logit(formula= "C1(graduated_4_year_degree_less_5_years) ~ year_1  + Q('11th_grade') + C(indicator_first_generation, Treatment) + C(best_fit, Treatment)  + C(good_fit, Sum) +C(none_fit, Treatment) + C(local_affordable, Treatment) + C(indicator_low_income) + C(indicator_persisted_into_year_2_ct) + C(school, Sum)", data=df_5_year_grades_above_3_gpa).fit(method='nm', maxiter=1000)
mod.summary()
```

### Save Excel file into reports directory

Save an Excel file with intermediate results into the report directory

```python
writer = pd.ExcelWriter(report_file, engine='xlsxwriter')
```

```python
df.to_excel(writer, sheet_name='Report')
```

```python
writer.save()
```
