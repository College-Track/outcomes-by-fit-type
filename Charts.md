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

## Setup

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
import matplotlib.ticker as mtick


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

### Plotting Setup

```python
sns.set_context("talk")
sns.set(style="white")

colors = ["#00a9e0", "#ffb81c", "#78be20", "#da291c", "#9063cd", "#97999b", "#007398", "#d0006f"]
current_palette = sns.color_palette(colors)
sns.set_palette(current_palette)


```

```python
rc = {'figure.figsize':(6,4),
      'axes.facecolor':'white',
      'axes.grid' : False,
      'grid.color': '.8',
      'font.family':'Avenir',
      'font.size' : 12,
      'axes.spines.top': False, 'axes.spines.right': False,
      'axes.titlesize': 116,
      'xtick.labelsize': 14,
      'ytick.labelsize': 14,
      'axes.labelsize': 14,
      'figure.titlesize':18,
      'figure.dpi': 200


     }



plt.rcParams.update(rc)
```

```python
def add_percent_bar_one_graph(ax):
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0%'), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
```

## Report Specific Setup

```python
df['best_fit'] =  np.where(df['updated_fit_type'] == 'Best Fit', True, False)
df['good_fit'] =  np.where(df['updated_fit_type'] == 'Good Fit', True, False)
df['local_affordable'] =  np.where(df['updated_fit_type'] == 'Local Affordable', True, False)
df['none_fit'] =  np.where(df['updated_fit_type'] == 'None', True, False)
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

### Creating Tables for Plots


#### Chart 1

```python
# Chart 1: 5 Year Grad Rate (All Students)
grad_rate_5_year = pd.crosstab(df_5_year_grades.updated_fit_type,
            df_5_year_grades.graduated_4_year_degree_less_5_years, normalize='index', margins=True).reset_index()

grad_rate_5_year_graph = grad_rate_5_year.drop(columns=[False])
```

```python
grad_rate_5_year_below_275 = pd.crosstab(df_5_year_grades_below_275_gpa.updated_fit_type,
            df_5_year_grades_below_275_gpa.graduated_4_year_degree_less_5_years, normalize='index', margins=True).reset_index()

grad_rate_5_year_below_275.round(2).style.format('{:.0%}')

grad_rate_5_year_below_275 = grad_rate_5_year_below_275.drop(columns=[False])
```

```python
grad_rate_5_year_graph['below_275'] = "All Students"
grad_rate_5_year_below_275['below_275'] = "Students with < 2.75 GPA"
```

```python
grad_rate_5_year_graph_all = grad_rate_5_year_graph.append(grad_rate_5_year_below_275,ignore_index=True)
```

```python
overall_grad_rate = grad_rate_5_year_graph.iloc[4,1]
```

#### Chart 2

```python
persistence_table = pd.crosstab(df.updated_fit_type, df.indicator_persisted_into_year_2_ct,
                                normalize='index', margins=True).reset_index()

persistence_table_below_275 = pd.crosstab(df_grades_below_275.updated_fit_type, df_grades_below_275.indicator_persisted_into_year_2_ct,
                                          normalize='index', margins=True).reset_index()

persistence_table = persistence_table.drop(columns=[False])
persistence_table_below_275 = persistence_table_below_275.drop(columns=[False])

persistence_table['below_275'] = "All Students"
persistence_table_below_275['below_275'] = "Students with < 2.75 GPA"
```

```python
overall_persistence_rate = persistence_table.iloc[4,1]
```

```python
persistence_table_all = persistence_table.append(persistence_table_below_275,ignore_index=True)
```

#### Chart 3

```python
fit_type_breakdown = pd.crosstab(df.high_school_class, df.updated_fit_type, normalize='index')
```

```python
fit_type_breakdown_below_275 = pd.crosstab(df_grades_below_275.high_school_class, df_grades_below_275.updated_fit_type, normalize='index')
```

```python
fit_type_breakdown_below_275['below_275'] = "Students with < 2.75 GPA"
fit_type_breakdown['below_275'] = "All Students"

```

```python
fit_type_breakdown_all = fit_type_breakdown.append(fit_type_breakdown_below_275,ignore_index=False)
```

```python
fit_type_breakdown_all = fit_type_breakdown_all.reset_index()
```

```python
fit_type_breakdown_original = fit_type_breakdown.drop(columns='below_275')
```

```python
fit_type_breakdown_long = fit_type_breakdown_original.unstack().reset_index()
```

```python
fit_type_breakdown_long.rename(columns={0:'Percent'}, inplace=True)
```

## Plots

<!-- #region heading_collapsed=true -->
#### Plot 1 
<!-- #endregion -->

```python hidden=true
# Plot 1
fig, ax = plt.subplots(figsize=(7, 4))

g = sns.barplot(data=grad_rate_5_year_graph_all[grad_rate_5_year_graph_all.updated_fit_type != "All"], hue='below_275',
                x='updated_fit_type', y=True, saturation=1, ax=ax)


g.set_xlabel('Fit Type')
g.set_ylabel('% of Students')
ax.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0., frameon=False)


ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))


for p in g.patches:
    g.annotate(format(p.get_height(), '.0%'), (p.get_x() + p.get_width() / 2., p.get_height()),
               ha='center', va='center', xytext=(0, 10), textcoords='offset points')


g.axhline(overall_grad_rate, ls='--',c='black')
g.text(2.65,.46, "Overall Grad Rate", fontsize=10)



fig.suptitle('5 Year Graduation Rate By Fit Type (2011-2014)')

```

<!-- #region heading_collapsed=true -->
#### Plot 2
<!-- #endregion -->

```python hidden=true
# Plot 
fig, ax = plt.subplots(figsize=(7, 4))

g = sns.barplot(data=persistence_table_all[persistence_table_all.updated_fit_type != "All"], hue='below_275',
                x='updated_fit_type', y=True, saturation=1, ax=ax)


g.set_xlabel('Fit Type')
g.set_ylabel('% of Students')
ax.legend( loc='upper center',bbox_to_anchor=(0.5, 1.08), borderaxespad=0., frameon=False, ncol=2)


ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))


for p in g.patches:
    g.annotate(format(p.get_height(), '.0%'), (p.get_x() + p.get_width() / 2., p.get_height()),
               ha='center', va='center', xytext=(0, 5), textcoords='offset points')


g.axhline(overall_persistence_rate, ls='--',c='black')
g.text(2.42,overall_persistence_rate+.01, "Overall Persistence Rate", fontsize=10)



fig.suptitle('Persistence Rate By Fit Type (2011-2018)', y=1.05)

```

<!-- #region heading_collapsed=true -->
#### Plot 3
<!-- #endregion -->

```python hidden=true
# Plot 
fig, ax = plt.subplots(figsize=(7, 4))

g = sns.lineplot(data=fit_type_breakdown_all, hue='below_275',
                x='high_school_class', y="Best Fit", ax=ax)


g.set_xlabel('High School Class')
g.set_ylabel('% of Students')
handles, labels = ax.get_legend_handles_labels()
# ax.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0., frameon=False, handles=handles[1:], labels=labels[1:])
ax.legend( loc='upper center',bbox_to_anchor=(0.5, 1.08), borderaxespad=0., frameon=False, ncol=2, handles=handles[1:], labels=labels[1:])
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))


# for p in g.patches:
#     g.annotate(format(p.get_height(), '.0%'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                ha='center', va='center', xytext=(0, 5), textcoords='offset points')


# g.axhline(overall_persistence_rate, ls='--',c='black')
# g.text(2.42,overall_persistence_rate+.01, "Overall Persistence Rate", fontsize=10)



fig.suptitle('Percent of Students Attending Best Fit Schools ', y=1.05)


```

```python hidden=true



```

```python hidden=true
# Plot 
fig, ax = plt.subplots(figsize=(7, 4))

g = sns.lineplot(data=fit_type_breakdown_long, hue='updated_fit_type',
                x='high_school_class', y="Percent", ax=ax)


g.set_xlabel('High School Class')
g.set_ylabel('% of Students')
handles, labels = ax.get_legend_handles_labels()
# ax.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0., frameon=False, handles=handles[1:], labels=labels[1:])
ax.legend( loc='upper center',bbox_to_anchor=(0.5, 1.08), borderaxespad=0., frameon=False, ncol=2, handles=handles[1:], labels=labels[1:])
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))





fig.suptitle('Fit Type Enrollment Over Time', y=1.05)

```

#### Chart 4 

```python
def C1(cat):
     return pd.get_dummies(cat, drop_first=True)
```

```python
columns = ['graduated_4_year_degree_less_5_years', 'year_1', '11th_grade', 'indicator_first_generation','updated_fit_type', 'indicator_low_income', 'indicator_persisted_into_year_2_ct']

df_5_year_grads_logit = df_5_year_grades.dropna(axis=0, subset=columns)

df_5_year_grads_below_275_logit = df_5_year_grades_below_275_gpa.dropna(axis=0, subset=columns)


```

```python
model_1 = smf.logit(formula="C1(graduated_4_year_degree_less_5_years) ~ year_1  + Q('11th_grade') + C(indicator_first_generation, Treatment) + C(updated_fit_type, Sum) + C(indicator_low_income) + C(indicator_persisted_into_year_2_ct)", data=df_5_year_grads_logit)

model_1_predict = model_1.fit(method='bfgs', maxiter=100).predict()


```

```python
model_2 = smf.logit(formula="C1(graduated_4_year_degree_less_5_years) ~ year_1  + Q('11th_grade') + C(indicator_first_generation, Treatment) + C(updated_fit_type, Sum) + C(indicator_low_income) + C(indicator_persisted_into_year_2_ct)", data=df_5_year_grads_below_275_logit)

model_2_predict = model_2.fit(method='bfgs', maxiter=100).predict()


```

```python
fig, ax = plt.subplots()
sns.regplot(df_5_year_grads_logit['year_1'].values,model_1_predict,order = 2, scatter_kws={"color": "#97999B", "alpha":1, "s":6}, ax=ax)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

ax.set_ylim((-.1, 1))


fig.suptitle('Probability of Graduating (5-Year) By Freshman Year GPA')

# ax.axhline(.75, ls='--',c='black')
# ax.axvline(3.25, ls='--',c='black')

# g.text(2.42,overall_persistence_rate+.01, "Overall Persistence Rate", fontsize=10)

plt.show()

```

```python
# pd.concat([df_5_year_grads_logit['updated_fit_type'], model_1_predict], axis=1)
# df_5_year_grads_logit['updated_fit_type'].to_frame().join(model_1_predict)

probablity_table_1 = pd.DataFrame({'updated_fit_type': list(df_5_year_grads_logit['updated_fit_type'].values),'prob':list(model_1_predict)})
```

```python
probablity_table_1.groupby('updated_fit_type').mean()
```

```python

probability_table_2 = pd.DataFrame({'updated_fit_type': list(df_5_year_grads_below_275_logit['updated_fit_type'].values),'prob':list(model_2_predict)})

probability_table_2.groupby('updated_fit_type').mean()
```

```python
fig, ax = plt.subplots()

sns.boxplot(df_5_year_grads_logit['updated_fit_type'].values, model_1_predict, saturation=1, ax=ax, showfliers=False, showmeans=True,
            meanprops={"marker": "x", "markerfacecolor": "black", "markeredgecolor": "black"}, order=["Best Fit", "Good Fit", "Local Affordable", "None"])

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

fig.suptitle('Probability of Graduating (5-Year) By Fit Type')
```

```python
fig, ax = plt.subplots()

sns.boxplot(df_5_year_grads_below_275_logit['updated_fit_type'].values,model_2_predict, saturation=1,showmeans=True, ax=ax,showfliers = False,
            meanprops={"marker":"x","markerfacecolor":"black", "markeredgecolor":"black"},order=["Best Fit", "Good Fit", "Local Affordable", "None"])

means = probability_table_2.groupby('updated_fit_type').mean()

for xtick in ax.get_xticks():
    print(xtick)
#     ax.text(xtick,means[xtick] + vertical_offset,means[xtick], 
#             horizontalalignment='center',size='x-small',color='w',weight='semibold')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

fig.suptitle('Probability of Graduating By Fit Type - Students Below 2.75 GPA')

plt.show()

```

```python
means.iloc[0]
```

```python

```
