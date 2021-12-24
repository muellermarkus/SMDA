import pandas as pd
import numpy as np
import os
import datetime as dt
import re
from bs4 import BeautifulSoup as bs
import matplotlib.pyplot as plt

def prep_df(file_name, data_path, score_lb = 1, num_hours = 1):

    # load posts
    df = pd.read_json(data_path + f'/posts/{file_name}.json')

    # get time window of interest
    start_time = dt.datetime(2015, 1, 1)
    end_time = dt.datetime(2021,10,31)

    # convert unix time stamp
    df['date'] = pd.to_datetime(df['created_utc'], unit = 's')
    df.sort_values('date', inplace = True, ignore_index=True)

    # ensure posts were only posted in intended time windows
    df.drop(df[df['date'] < start_time].index, inplace=True)
    df.drop(df[df['date'] > end_time].index, inplace=True)

    # remove automod posts and stickied posts
    del_idx = df[df['stickied'] == True].index
    print(f"remove {len(del_idx)} stickied posts")
    df.drop(del_idx, inplace=True)
    
    # remove posts that have score <= score_lb
    # posts that only creator cared about
    # this filters out noise
    del_idx = df[df['score'] <= score_lb].index
    print(f"remove {len(del_idx)} posts with score <= {score_lb}")
    df.drop(del_idx, inplace=True)
    # split in terms of score -> x variable for intensity (heterogeneity); pos vs neg/zero
    # also do that for vote_ratio

    # keep variables of interest
    if file_name == 'electricvehicles':
        keep_vars = ['date', 'num_comments', 'score', 'upvote_ratio']
    elif file_name == 'news':
        keep_vars = ['date', 'num_comments', 'score', 'upvote_ratio']
    elif file_name == 'worldnews':
        keep_vars = ['date', 'num_comments', 'score', 'upvote_ratio']
        
    # subset dataframe
    df = df[keep_vars]

    # round the date to hours
    df.loc[:,'date'] = df['date'].round(f'{num_hours}H')

    # collect means, sums and post counts
    mean_df = df.groupby('date').mean()
    sum_df = df.groupby('date').sum()
    counts = df.groupby('date').count()['num_comments']

    # adjust labels
    mean_df.columns = mean_df.columns + '_mean'
    sum_df.columns = sum_df.columns + '_sum'
    counts.name = 'post_count'

    # combine to one dataframe
    df = pd.concat([mean_df, sum_df, counts], axis = 1)
    
    # make date a separate column
    df.reset_index(level=0, inplace=True)
    
    # set T relative to start of observational window in minutes
    df.loc[:,'date'] = (df['date'] - start_time).astype('timedelta64[m]')
    
    # make time in num_hours intervals
    df.loc[:,'date'] = df['date']/(num_hours*60)
    
    # save data
    df.to_parquet(data_path + f'/processed/{file_name}_cleaned.gzip', compression = 'gzip')
    print(f'{file_name} cleaned and new dataframe saved')
    
    return df

#################################################################################

# define data path
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# define files to clean
file_names = ['electricvehicles', 'news', 'worldnews']

# prepare dataframes
for file_name in file_names:
    df = prep_df(file_name, data_path, score_lb = 1, num_hours = 0.25)
    
# combine worldnews and news data
file_name = 'news'
news = pd.read_parquet(data_path + f'/processed/{file_name}_cleaned.gzip') 
file_name = 'worldnews'
worldnews = pd.read_parquet(data_path + f'/processed/{file_name}_cleaned.gzip') 

df = pd.concat([news, worldnews])
df.reset_index(inplace=True, drop=True)
df.sort_values(by = 'date')
sum(df.duplicated('date')) # 130

# remove duplicates for now
df = df[~df.duplicated('date')]

# select columns
df = df[['date']]

# save dataframe
df.to_parquet(data_path + '/processed/disasters.gzip', compression = 'gzip')

# prepare electricvehicles data
file_name = 'electricvehicles'
df = pd.read_parquet(data_path + f'/processed/{file_name}_cleaned.gzip') 

# remove duplicates for now
df = df[~df.duplicated('date')]

# select columns
df = df[['date']]

# save dataframe
df.to_parquet(data_path + '/processed/electricvehicles.gzip', compression = 'gzip')

#################################################################################

# check distributions
file_name = 'electricvehicles'
df_ev = pd.read_parquet(data_path + f'/processed/{file_name}_cleaned.gzip')    
file_name = 'news'
df_news = pd.read_parquet(data_path + f'/processed/{file_name}_cleaned.gzip')    
plt.plot(df['date'], df['post_count'])
file_name = 'worldnews'
df_wnews = pd.read_parquet(data_path + f'/processed/{file_name}_cleaned.gzip')    
plt.plot(df['date'], df['post_count'])

# combine news and worldnews data
df_plot = pd.concat([df_news, df_wnews]).groupby('date').sum()
df_plot.reset_index(level=0, inplace=True)

# prepare figure save path
fig_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')

fig = plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(df_plot['date'], df_plot['post_count'])
plt.title('Climate Change Data')
plt.xlabel('time in 15 min intervals')
plt.ylabel('number of posts')

plt.subplot(1, 2, 2)
plt.plot(df_ev['date'], df_ev['post_count'])
plt.title('Electric Vehicles (EV) Data')
plt.xlabel('time in 15 min intervals')
plt.ylabel('number of posts')

plt.savefig(fig_path + '/dataplot.png', dpi = 300)

# clean comments of car subreddit (weekly megathread)

# function that does basic cleaning to comment on post
def comment_cleaner(body):
    # parse html
    body_html = bs(body, 'html.parser')
    body = body_html.text.lower()
    
    # basic cleaning of spaces and slashes
    body = re.sub(r'\n', ' ', body)
    body = re.sub(r'/', ' ', body)
        
    return body

search_models = ['model 3', 'model s', 'model x', 'model y', 'tesla', 'phev', 'ev', 'mhev', 'shev', 'bev', 'fchev']
search_general = ['electric', 'hybrid', 'environment'] # also matches electrical, environmental, etc.

# load cached post ids of megathreads in cars subreddit
cache = pd.read_csv(data_path + f'/posts/cars_cache.csv')['0'].tolist()

# init lists
comment_dfs = []
post_data_list = []

for post_id in cache:
    
    # load comments for a post
    try:
        df = pd.read_json(data_path + f'/comments/cars/{post_id}.json')
    except: # for two post ids scraper found no comments
        continue

    # remove non-toplevel comments
    del_idx = df[df['comment_parent_id'].str.contains('t1')].index 
    # print(f"remove {len(del_idx)} child comments")
    df.drop(del_idx, inplace=True)
    
    # get time window of interest
    start_time = dt.datetime(2015, 1, 1)
    end_time = dt.datetime(2021,10,31)

    # convert unix time stamp
    df['date'] = pd.to_datetime(df['comment_created_utc'], unit = 's')

    # ensure comments were only posted in intended time windows
    df.drop(df[df['date'] < start_time].index, inplace=True)
    df.drop(df[df['date'] > end_time].index, inplace=True)

    # init dict and list
    comment_dict = {word : 0 for word in search_models + search_general}
    comment_data_list = []

    # get mentions of search_words per comment
    for comment in df['comment_body']:
        
        # basic cleaning
        comment = comment_cleaner(comment)

        # init dict for comment
        comment_dict = {word : 0 for word in search_models + search_general}
        
        # update dict with car model counts
        for search_word in search_models:
            comment_dict[search_word] += len(re.findall(f'(?<=\s){search_word}(?=[^\d^\w])', comment))
        
        for search_word in search_general:
            comment_dict[search_word] += len(re.findall(f'(?<=\s){search_word}', comment))
            
        # save dict to
        comment_data_list.append(comment_dict)
        
    # combine dictionaries to dataframe
    comment_df = pd.DataFrame(comment_data_list)
    save_df = pd.concat([df['date'].reset_index(drop = True), comment_df.sum(axis = 1)], axis=1)
    save_df.columns = ['date', 'ev_mention']
    # construct dummy = 1 if comment mentions electricvehicle (model or general word above)
    save_df['ev_mention'] = np.where(save_df['ev_mention'] > 0, 1, 0)
    comment_dfs.append(save_df)

    # sum over comments
    post_data = comment_df.sum(axis = 0).to_dict()  
    post_data['date'] = min(df['date'])
    
    # add to dataframe on post level
    post_data_list.append(post_data)
    
# create dataframe on post level
df = pd.DataFrame(post_data_list)
df.describe()

# create dataframe on comment level
comment_df = pd.concat(comment_dfs)

# keep only those comments which mention EVs
comment_df = comment_df[comment_df['ev_mention'] == 1] # starts only in 2016

for data in [df, comment_df]:
    data.loc[:,'date'] = (data['date'] - start_time).astype('timedelta64[m]')
    data.loc[:,'date'] = data['date']/(0.25*60)
    data.sort_values('date', inplace = True, ignore_index=True)

# create sum of word occurrences
df['sum'] = df.iloc[:, 0:len(df.columns)-1].sum(axis=1)

# check distribution of EV mentions
plt.plot(df['date'], df['hybrid'])

plt.plot(df['date'], df['tesla'])

plt.plot(df['date'], df['electric'])

plt.plot(df['date'], df['sum'])
