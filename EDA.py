#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from scipy.stats import kruskal, f_oneway

plt.style.use('ggplot')

# read in the data
X1 = pd.read_csv('data/X1.csv')
Y1 = pd.read_csv('data/Y1.csv', header=None, names=['shares'])


df = X1.copy()
shares  = Y1.copy()
df['shares'] = shares

# Reorganize the feature dataset by feature category
word_features = ['n_tokens_title', 'n_tokens_content', 'average_token_length',
                 'n_non_stop_words', 'n_unique_tokens', 'n_non_stop_unique_tokens']
links_features = ['num_hrefs', 'num_self_hrefs',  'self_reference_min_shares',
                  'self_reference_avg_sharess', 'self_reference_max_shares']
digital_media_features = ['num_imgs', 'num_videos']
time_features = ['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 
                 'weekday_is_thursday', 'weekday_is_friday','weekday_is_saturday', 
                 'weekday_is_sunday', 'is_weekend']
keyword_features = ['num_keywords', 'kw_min_min', 'kw_avg_min', 'kw_max_min', 'kw_min_avg', 
                    'kw_avg_avg', 'kw_avg_max', 'kw_min_max', 'kw_max_avg',  'kw_max_max',
                    'data_channel_is_lifestyle','data_channel_is_entertainment', 'data_channel_is_bus',
                    'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world']
nlp_features = ['LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'title_subjectivity', 'global_subjectivity',
                'abs_title_subjectivity', 'title_sentiment_polarity', 'rate_positive_words', 'rate_negative_words',
                'global_rate_positive_words', 'global_rate_negative_words', 'min_positive_polarity',
                'avg_positive_polarity', 'max_positive_polarity', 'min_negative_polarity', 'avg_negative_polarity',
                'max_negative_polarity', 'global_sentiment_polarity', 'abs_title_sentiment_polarity']

boolean_features = [ colname for colname in df.columns if                    (colname.startswith('weekday_is_') or                     colname.startswith('data_channel_is_') or                    colname.startswith('is_weekend'))]

# Add the target into the feature dataset
df = pd.concat(
    [shares, df[word_features], df[links_features], df[digital_media_features], 
     df[time_features], df[keyword_features], df[nlp_features]], 
    axis=1
)
df['log_shares'] = np.log(shares)

df[word_features].describe()

# Check missing values for boolean_features
df.isnull().sum(axis=0)[boolean_features]


# given any of the boolean features does not have missing value, we can assign dtype as np.int16
# This will help in making difference with other columns
df = df.astype(dtype={col:np.int16 for col in boolean_features}, copy=True)


def set_day_of_week(df):
    dow = [var for var in df.columns if var.startswith('weekday_is_')]
    df['day_of_week'] = ''
    for v in dow:
        df.loc[df[v] == 1.0, 'day_of_week'] = v[len('weekday_is_'):].title()
    return df

def set_article_category(df):
    cat = [var for var in df.columns if var.startswith('data_channel_is_')]
    df['category'] = 'Viral'
    for v in cat:
        df.loc[df[v] == 1.0, 'category'] = v[len('data_channel_is_'):].title()
    return df

df = set_day_of_week(df)
df = set_article_category(df)
df[['day_of_week', 'category']].head(3)


df['day_of_week'].value_counts().plot(kind='bar');

dow = df['day_of_week'].unique()
dow
df['dow_num'] = df['day_of_week'].map({
    'Monday':1, 'Tuesday':2, 'Wednesday': 3, 'Thursday':4, 
    'Friday':5, 'Saturday':6, 'Sunday':7})
plt.figure(figsize=(12, 8))
sns.boxplot(x='day_of_week', y='log_shares', data=df.sort_values(by='dow_num'));
plt.savefig('graphs/log_shares_by_dow.png')


plt.figure(figsize=(4,8))
sns.boxplot(x='is_weekend', y='log_shares', data=df);
plt.savefig('graphs/log_shares_by_is_weekend.png')



kruskal(
    df.loc[df['day_of_week'] == 'Monday','log_shares'],
    df.loc[df['day_of_week'] == 'Tuesday','log_shares'],
    df.loc[df['day_of_week'] == 'Wednesday','log_shares'],
    df.loc[df['day_of_week'] == 'Thursday','log_shares'],
    df.loc[df['day_of_week'] == 'Friday','log_shares'],
    df.loc[df['day_of_week'] == 'Saturday','log_shares'],
    df.loc[df['day_of_week'] == 'Sunday','log_shares']
)
kruskal(
    df.loc[df['is_weekend']==1.0, 'log_shares'],
    df.loc[df['is_weekend']==0.0, 'log_shares'],
)

f_oneway(
    df.loc[df['day_of_week'] == 'Monday','log_shares'],
    df.loc[df['day_of_week'] == 'Tuesday','log_shares'],
    df.loc[df['day_of_week'] == 'Wednesday','log_shares'],
    df.loc[df['day_of_week'] == 'Thursday','log_shares'],
    df.loc[df['day_of_week'] == 'Friday','log_shares'],
    df.loc[df['day_of_week'] == 'Saturday','log_shares'],
    df.loc[df['day_of_week'] == 'Sunday','log_shares']
)
f_oneway(
    df.loc[df['is_weekend']==1.0, 'log_shares'],
    df.loc[df['is_weekend']==0.0, 'log_shares'],
)


df['category'].value_counts().plot(kind='bar');


plt.figure(figsize=(12, 8))
sns.boxplot(x='category', y='log_shares', data=df);
plt.savefig('graphs/log_shares_by_category.png')


# Compute the correlation matrices with different methods, exclude dummy columns
numeric_features = list(df.dtypes[ df.dtypes != 'int16' ].index)
numeric_features.remove('dow_num')
numeric_features.remove('log_shares')

def print_corr(df, method, features):
    corr = df[features].corr(method=method)
    plt.figure(figsize=(21, 21))
    sns.heatmap(corr, cmap='YlGnBu');
    plt.savefig(fname=f'graphs/{method}_correlation.png')
    return corr


# Histogram of numerical features
df[numeric_features].hist(bins=50, figsize=(20, 15));
plt.savefig('graphs/hist_numeric_features.png')
plt.show()


df[word_features].hist(bins=50, figsize=(20,15));
plt.show()


pearson = print_corr(df=df, method='pearson', features=numeric_features)
kendall = print_corr(df=df, method='kendall', features=numeric_features)
spearman = print_corr(df=df, method='spearman', features=numeric_features)


# df[word_features].corr().style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)
spearman.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)



def filter_correlations(corr: pd.DataFrame, th: float):
    # code from https://izziswift.com/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas/
    filtered = corr[ ((corr >= th) | (corr <= -th)) & (corr != 1.00)]
    flattened = filtered.unstack().sort_values().drop_duplicates().dropna(axis=0)
    return flattened


top_corrs = filter_correlations(pearson, 0.7)
top_corrs.to_frame()
plt.figure(figsize=(12, 6))
top_corrs.plot(kind='barh', title="Top Pearson correlations")
plt.savefig('graphs/top_corrs.png')


# In[44]:


top_corrs_features = list(top_corrs.index)
n = len(top_corrs_features)
cols = 3
rows = int( np.ceil( n/cols ) )
fig, axes = plt.subplots(rows, cols, figsize=(21, 15))
plt.subplots_adjust(hspace=0.3)                         
for index in range(n):
    i = index // cols
    j = index % cols
    x_col = top_corrs_features[index][0]
    y_col = top_corrs_features[index][1]
#     sns.scatterplot(data=dfs[index], x='feature1', y='feature2', ax=axes[i, j]);
    axes[i, j].scatter(x=df[x_col], y=df[y_col], c='b')
    axes[i, j].set_xlabel(x_col)
    axes[i, j].set_ylabel(y_col)
# fig.tight_layout()
fig.savefig('graphs/scatter_matrix_top_corrs.png')


pearson[ np.abs(pearson) > 0.5 ].to_html('pearson.html')
kendall[ np.abs(kendall) > 0.5 ].to_html('kendall.html')
spearman[ np.abs(spearman) > 0.5 ].to_html('spearman.html')
