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

## OP Learning Agenda: Influence of College Fit Type on Students' Outcomes


### Abstract

One of the key choices in a students' career is what college they attend. To help guide students, College Track has created a set of criteria to categorize schools to better inform students of their chance of success. 

The goal of this learning agenda was to evaluate if students chance of success was influenced by the fit type of school they attended. 

Overall, Fit Type is a moderately good indicator of students graduation rate, however, it still lags heavily behind GPA as the best predictors. Of note, it is more predictive than any income, ethnic, or first-generation status. 

For this project, students were broken up into different categories based on their GPA to see if fit type had a bigger influence on students in certain GPA ranges. Students were only divided based on GPA as ethnicity, income, and first-generation status didn't have a strong predictor and would only limit the sample sizes. 

For students with GPAs above 2.75, most fit types had no statistically significant influence on their likelihood to graduate. The only exception was attending a local affordable school did have a small negative influence on graduation success. 

For students below a 2.75 GPA, only attending a best fit school had a positive influence on graduation success, and all other fit types had a statistically significant negative impact. This positive relationship with attending a best fit school applies to all students below a 2.75 GPA that we have enough data on. 

Of note there, is attending a good fit, local affordable, or no fit type school has a negative impact on graduation success. In particular, attending a local affordable school deceased your chances of graduation by roughly half. 

One artifact of this analysis is the possibility that the specific school a student attends has a very high predictor of graduation success. Unfortunately, we do not yet have enough data on graduation rates from individual schools to fully explore this possibility, but it does open the door to reevaluating how we determine schools' fit types. 

```python
import pandas as pd
from pathlib import Path
from datetime import datetime
import statsmodels.api as sm
import numpy as np
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from dython.nominal import correlation_ratio, associations

import statsmodels.formula.api as smf
import statsmodels.api as sm

```

```python
%matplotlib inline
```

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
# Because some data points are limited for historical classes,
# setting up data frames to deal with various historical limitations 

df_matriculation_data = df[df.high_school_class >= 2015]

df_4_year_grades = df[df.high_school_class <= 2015]
df_5_year_grades = df[df.high_school_class <= 2014]
df_6_year_grades = df[df.high_school_class <= 2013]
```

```python
df_5_year_grades_above_3_gpa = df_5_year_grades[df_5_year_grades.year_1 >= 3]

df_5_year_grades_below_3_gpa = df_5_year_grades[df_5_year_grades.year_1 < 3]

df_5_year_grades_below_275_gpa = df_5_year_grades[df_5_year_grades.year_1 <= 2.75]

df_grades_below_275 = df[df.year_1 <= 2.75]
df_grades_above_3_gpa = df[df.year_1 > 3]
```

## General Distributions


#### Table 1. 5 Year Graduation Rate by Fit Type

```python
grad_rate_5_year = pd.crosstab(df_5_year_grades.updated_fit_type,
            df_5_year_grades.graduated_4_year_degree_less_5_years, normalize='index', margins=True)

grad_rate_5_year.round(2).style.format('{:.0%}')
```

#### Table 2. 5 Year Graduation Rate by Fit Type - Students Below 2.75 GPA in Year 1

```python
grad_rate_5_year = pd.crosstab(df_5_year_grades_below_275_gpa.updated_fit_type,
            df_5_year_grades_below_275_gpa.graduated_4_year_degree_less_5_years, normalize='index', margins=True)

grad_rate_5_year.round(2).style.format('{:.0%}')
```

#### Table 3. 5 Year Graduation Rate by Fit Type - Students Below 3.0 GPA in Year 1

```python
grad_rate_5_year = pd.crosstab(df_5_year_grades_above_3_gpa.updated_fit_type,
            df_5_year_grades_above_3_gpa.graduated_4_year_degree_less_5_years, normalize='index', margins=True)

grad_rate_5_year.round(2).style.format('{:.0%}')
```

#### Table 4. Persistence Rate by Fit Type

```python
pd.crosstab(df.updated_fit_type, df.indicator_persisted_into_year_2_ct,
            normalize='index').round(2).style.format('{:.0%}')
```

#### Table 5. Persistence Rate by Fit Type - Students Below 2.75 GPA in Year 1 

```python
pd.crosstab(df_grades_below_275.updated_fit_type, df_grades_below_275.indicator_persisted_into_year_2_ct,
            normalize='index').round(2).style.format('{:.0%}')
```

#### Table 6. Fit Type Breakdown by High School Class

```python
pd.crosstab(df.high_school_class, df.updated_fit_type, normalize='index').round(2).style.format('{:.0%}')
```

#### Table 7. Fit Type Breakdown by High School Class - Students Below 2.75 in Year 1 

```python
pd.crosstab(df_grades_below_275.high_school_class, df_grades_below_275.updated_fit_type, normalize='index').round(2).style.format('{:.0%}')
```

#### Table 8. Fit Type Breakdown by High School Class - Students Above 3.0 in Year 1 


```python
pd.crosstab(df_grades_above_3_gpa.high_school_class, df_grades_above_3_gpa.updated_fit_type, normalize='index').round(2).style.format('{:.0%}')
```

## Correlation Coefficients

```python
associations(df_5_year_grades[
    ['graduated_4_year_degree_less_5_years', 'indicator_first_generation',
     'indicator_low_income', '11th_grade_gpa_bucket', 'ethnic_background',
     'year_1_gpa_bucket', 'college_elig_gpa_11th_cgpa',
     'school', 'updated_fit_type', 'indicator_persisted_into_year_2_ct']
], theil_u=True, cmap="RdBu", figsize=(8, 6))
```

## Regressions

```python
def C1(cat):
     return pd.get_dummies(cat, drop_first=True)
```

#### Regression 1. 5 Year Graduation Rate - All Eligible Students

Note: The significant p value for Best Fit and Local Affordable schools - best fit with the positive coefficient and local affordable with the negative coefficient. 

```python
mod = smf.logit(formula= "C1(graduated_4_year_degree_less_5_years) ~ year_1  + Q('11th_grade') + C(indicator_first_generation, Treatment) + C(updated_fit_type, Sum) + C(indicator_low_income) + C(indicator_persisted_into_year_2_ct)", data=df_5_year_grades).fit(method='bfgs', maxiter=100)
mod.summary()
```

#### Regression 2. 5 Year Graduation Rate - Students above 3.0 GPA

Note, none of the fit types are statistically significant

```python
mod = smf.logit(formula= "C1(graduated_4_year_degree_less_5_years) ~ year_1  + Q('11th_grade') + C(indicator_first_generation, Treatment) + C(updated_fit_type, Sum) + C(indicator_low_income) + C(indicator_persisted_into_year_2_ct)", data=df_5_year_grades_above_3_gpa).fit(method='bfgs', maxiter=100)
mod.summary()
```

#### Regression 3. 5 Year Graduation Rate - Students below 2.75 GPA

Note, once again only best and local affordable are statistically significant. 


```python
mod = smf.logit(formula= "C1(graduated_4_year_degree_less_5_years) ~ year_1  + Q('11th_grade') + C(indicator_first_generation, Treatment) + C(updated_fit_type, Sum) + C(indicator_low_income) + C(indicator_persisted_into_year_2_ct)", data=df_5_year_grades_below_275_gpa).fit(method='bfgs', maxiter=100)
mod.summary()
```

##### Odds Ratio for Previous Regression

```python
np.exp(mod.params)
```

```python


%%html

<script>
$(document).ready(function(){
    window.code_toggle = function() {
        (window.code_shown) ? $('div.input').hide(250) : $('div.input').show(250);
        window.code_shown = !window.code_shown
    }
    if($('body.nbviewer').length) {
        $('<li><a href="javascript:window.code_toggle()" title="Show/Hide Code"><span class="fa fa-code fa-2x menu-icon"></span><span class="menu-text">Show/Hide Code</span></a></li>').appendTo('.navbar-right');
        window.code_shown=false;
        $('div.input').hide();
    }
});
</script>


<style>

div.prompt {display:none}


h1, .h1 {
    font-size: 33px;
    font-family: "Trebuchet MS";
    font-size: 2.5em !important;
    color: #2a7bbd;
}

h2, .h2 {
    font-size: 10px;
    font-family: "Trebuchet MS";
    color: #2a7bbd; 
    
}


h3, .h3 {
    font-size: 10px;
    font-family: "Trebuchet MS";
    color: #5d6063; 
    
}

.rendered_html table {

    font-size: 14px;
}

.output_png {
  display: flex;
  justify-content: center;
}

.cell {
    padding: 0px;
}


</style>
```

```python

```
