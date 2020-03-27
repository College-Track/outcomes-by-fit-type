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

### Data Sources
- Contact Report: https://ctgraduates.lightning.force.com/lightning/r/Report/00O1M0000077ZUVUA2/view?queryScope=userFolders
- Academic Term Report:  https://ctgraduates.lightning.force.com/lightning/r/Report/00O1M0000077ZWMUA2/view?queryScope=userFolders
- College Accounts:  https://ctgraduates.lightning.force.com/lightning/r/Report/00O1M0000077aI6UAI/view

### Changes
- 03-19-2020 : Started project

```python
# ALWAYS RUN
# General Setup 

%load_ext dotenv
%dotenv

from salesforce_reporting import Connection, ReportParser
import pandas as pd
from pathlib import Path
from datetime import datetime
import helpers
import os
import numpy as np
from reportforce import Reportforce
from scipy.stats import mode



SF_PASS = os.environ.get("SF_PASS")
SF_TOKEN = os.environ.get("SF_TOKEN")
SF_USERNAME = os.environ.get("SF_USERNAME")

sf = Reportforce(username=SF_USERNAME, password=SF_PASS, security_token=SF_TOKEN)
```

### File Locations

```python
# ALWAYS RUN
today = datetime.today()


in_file1 = Path.cwd() / "data" / "raw" / "sf_output_file1.csv"
summary_file = Path.cwd() / "data" / "processed" / "processed_data.pkl"


in_file2 = Path.cwd() / "data" / "raw" / "sf_output_file2.csv"
summary_file2 = Path.cwd() / "data" / "processed" / "processed_data_file2.pkl"


in_file3 = Path.cwd() / "data" / "raw" / "sf_output_file3.csv"
summary_file3 = Path.cwd() / "data" / "processed" / "processed_data_file3.pkl"


in_file4 = Path.cwd() / "data" / "raw" / "sf_output_file4.csv"
summary_file4 = Path.cwd() / "data" / "processed" / "processed_data_file4.pkl"
```

### Load Report From Salesforce

```python
# Run if downloading report from salesforce
# File 1 
report_id_file1 = "00O1M0000077ZUVUA2"
file_1_id_column = '18 Digit ID' # adjust as needed
sf_df = sf.get_report(report_id_file1, id_column=file_1_id_column)



# File 2 (As needed)
# report_id_file2 = "00O1M0000077ZWMUA2"
# file_2_id_column = 'Academic Term: ID' # adjust as needed
# sf_df_file2 =  sf.get_report(report_id_file2, id_column=file_2_id_column)

# File 3 (As needed)
report_id_file3 = "00O1M0000077aI6UAI"
file_3_id_column = 'Account ID' # adjust as needed
sf_df_file3 =  sf.get_report(report_id_file3, id_column=file_3_id_column)



```

```python
len(sf_df),len(sf_df_file3)
```

#### Save report as CSV

```python
# Only run if ran above cell
# File 1
sf_df.to_csv(in_file1, index=False)


# File 2 and 3 (As needed)
# sf_df_file2.to_csv(in_file2, index=False)
sf_df_file3.to_csv(in_file3, index=False)

```

### Load DF from saved CSV
* Start here if CSV already exist 

```python
# ALWAYS RUN 
# Data Frame for File 1 - if using more than one file, rename df to df_file1
df = pd.read_csv(in_file1)


# Data Frames for File 1 and 2 (As needed)

df_file2 = pd.read_csv(in_file2)
df_file3 = pd.read_csv(in_file3)
```

### Data Manipulation

```python
def create_gpa_bucket(gpa):
    if gpa < 2.5:
        return "2.5 or less"
    elif gpa < 2.75:
        return "2.5 - 2.74"
    elif gpa < 3.0:
        return "2.75 - 2.9"
    elif gpa < 3.5:
        return "3.0 - 3.49"
    elif gpa >= 3.5:
        return "3.5 or greater"
    else:
        return "NULL"
```

```python
df_file3['local_affordable_sites'] = df_file3.apply(lambda x: list([x['Local Fit Site 1'],
                                        x['Local Fit Site 2'],
                                        x['Local Fit Site 3'],
                                        x['Local Fit Site 4'],
                                        x['Local Fit Site 5']]),axis=1)      

```

```python
df_file2.columns
```

```python
len(df_file2)
```

```python
def determine_fit_type(school_id, site, current_fit_type):

    if pd.isna(school_id):
        return current_fit_type
    _tmp_df = df_file3[df_file3['18 Digit ID'] == school_id]
    
    if len(_tmp_df) == 0:
        return current_fit_type

    if _tmp_df['Best Fit College'].values == "Yes":
        return "Best Fit"
    elif site in _tmp_df.local_affordable_sites.iloc[0]:
        return "Local Affordable"
    else:
        return current_fit_type
    
```

```python
df_file2['updated_fit_type'] = df_file2.apply(lambda x: determine_fit_type(
    x['School: Account 18 Digit ID'], x['Site'], x['Fit Type']), axis=1)
```

```python
# File 1
df = helpers.shorten_site_names(df)
df = helpers.clean_column_names(df)

# File 2
df_file2 = helpers.shorten_site_names(df_file2)
df_file2 = helpers.clean_column_names(df_file2)

df_file3 = helpers.clean_column_names(df_file3)


```

#### Merging Data into File 1

```python
# Determining the most common fit type a student attended

df_file_2_drop_dups = df_file2.drop_duplicates(
    subset=["18_digit_id", "global_academic_term"]
)



df_file_2_drop_dups = df_file_2_drop_dups[df_file_2_drop_dups.academic_term_record_type == "College/University Semester"]

```

```python
school_merge = df_file_2_drop_dups.groupby(
    '18_digit_id')['school'].apply(lambda x: mode(x)[0][0]).reset_index()

df = df.merge(school_merge, on="18_digit_id", how="left")
```

```python
fit_type_merge = df_file_2_drop_dups.groupby(
    '18_digit_id')['fit_type'].apply(lambda x: mode(x)[0][0]).reset_index()

df = df.merge(fit_type_merge, on="18_digit_id", how="left")
```

```python
fit_type_merge = df_file_2_drop_dups.groupby(
    '18_digit_id')['updated_fit_type'].apply(lambda x: mode(x)[0][0]).reset_index()


df = df.merge(fit_type_merge, on="18_digit_id", how="left")
```

```python
df.loc[df.updated_fit_type == 0,'updated_fit_type'] = 'None'

df.loc[df.fit_type == 0,'updated_fit_type'] = 'None'
```

```python
df_file_2_spring = df_file2[df_file2.global_academic_term.str.match('Spring*', na=False)]
```

```python
df_file_2_spring = df_file_2_spring.drop_duplicates(
    subset=["18_digit_id", "global_academic_term"]
)
```

```python
df_file_2_spring = df_file_2_spring.drop_duplicates(subset=['18_digit_id', 'grade_at'])
```

```python
grade_pivot = df_file_2_spring.pivot(
    index="18_digit_id", columns="grade_at", values="gpa_running_cumulative",
)
```

```python
df = df.merge(grade_pivot, on="18_digit_id", how="left")
```

```python
df = helpers.clean_column_names(df)
```

```python
df["9th_grade_gpa_bucket"] = df.apply(
    lambda x: create_gpa_bucket(x['9th_grade']), axis=1
)

df["10th_grade_gpa_bucket"] = df.apply(
    lambda x: create_gpa_bucket(x['10th_grade']), axis=1
)

df["11th_grade_gpa_bucket"] = df.apply(
    lambda x: create_gpa_bucket(x['11th_grade']), axis=1
)

df["12th_grade_gpa_bucket"] = df.apply(
    lambda x: create_gpa_bucket(x['12th_grade']), axis=1
)

df["year_1_gpa_bucket"] = df.apply(
    lambda x: create_gpa_bucket(x['year_1']), axis=1
)

df["year_2_gpa_bucket"] = df.apply(
    lambda x: create_gpa_bucket(x['year_2']), axis=1
)


df["year_3_gpa_bucket"] = df.apply(
    lambda x: create_gpa_bucket(x['year_3']), axis=1
)


df["year_4_gpa_bucket"] = df.apply(
    lambda x: create_gpa_bucket(x['year_4']), axis=1
)
```

```python
df[df['18_digit_id'] == '0034600001TQzctAAD'].updated_fit_type
```

```python
df_file2[df_file2['18_digit_id'] == '0034600001TQzctAAD']
```

```python

df_file3[df_file3['18_digit_id']=='0014600001LaiNnAAJ']
```

### Save output file into processed directory

Save a file in the processed directory that is cleaned properly. It will be read in and used later for further analysis.

```python
# Save File 1 Data Frame (Or master df)
df.to_pickle(summary_file)

df_file2.to_pickle(summary_file2)

# df_file3.to_pickle(summary_file3)
```

```python

```
