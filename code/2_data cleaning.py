import pandas as pd
import os
import datetime as dt
import re
from bs4 import BeautifulSoup as bs
import matplotlib.pyplot as plt



# how many hours to round to
num_hours = 1

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
    df.loc[:,'date'] = df['date'].round(f'{num_hours}h')

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
    
    # construct time differences to previous post
    time_diffs = [df.loc[i, 'date'] - df.loc[i-1, 'date'] for i in range(1, df.shape[0])]

    # set initial post to time = 0
    time_diffs = [pd.to_timedelta(0)] + time_diffs

    # include in df
    df['time_diff'] = time_diffs
    
    # convert time differences to hours
    df.loc[:, 'time_diff'] = df['time_diff'].dt.seconds/60/60
    
    # save data
    df.to_parquet(data_path + f'/processed/{file_name}_cleaned.gzip', compression = 'gzip')
    print('f{file_name} cleaned and new dataframe saved')
    
    return df

# what time unit to use? we have at least one post per day, so there is no variation, the older the subreddit, the more users post, the less variation in greater time units, e.g. one post every hour?
# somehow select subset of posts, i.e. that contain certain words? or with self.text? 

# inverse weight scores etc. by subscriber count at that date (data not available)

# could also look at sentiment of post titles on climate change and how the average sentiment changes over time






# define data path
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

file_names = ['electricvehicles', 'news', 'worldnews']

for file_name in file_names:
    df = prep_df(file_name, data_path, score_lb = 1, num_hours = 1)

    
    
# 15min
    
# check distributions

file_name = 'electricvehicles'
file_name = 'news'
file_name = 'worldnews'
df = pd.read_parquet(data_path + f'/processed/{file_name}_cleaned.gzip')    
plt.plot(df['date'], df['post_count'])

# see coincidence in 2017!

   


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


search_models = ['model 3', 'model s', 'model x', 'model y', 'tesla', 'phev', 'ev', 'mhev', 'shev', 'bev', 'fchev'] #e-tron? can add to car models
search_general = ['electric', 'hybrid', 'environment'] # also matched electrical, environmental, etc.


# load cached post ids of megathreads in cars subreddit
cache = pd.read_csv(data_path + f'/posts/cars_cache.csv')['0'].tolist()

# init list
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
    # comment_df.describe()

    # sum over comments
    post_data = comment_df.sum(axis = 0).to_dict()  
    post_data['date'] = min(df['date'])
    
    # add to dataframe on post level
    post_data_list.append(post_data)
    
# create dataframe on post level
df = pd.DataFrame(post_data_list)
df.describe()

# round the date to hours
df.loc[:,'date'] = df['date'].round(f'{num_hours}h')

# sort df
df.sort_values('date', inplace = True, ignore_index=True)

# create sum of word occurrences
df['sum'] = df.iloc[:, 0:len(df.columns)-1].sum(axis=1)

# check distribution
plt.plot(df['date'], df['hybrid'])

plt.plot(df['date'], df['tesla'])

plt.plot(df['date'], df['electric'])

plt.plot(df['date'], df['sum'])

# current caveats
# only if before these words there is not a 'not'
# electric may not relate vehicle

# compare subreddits in terms of subscribers by comparing to trends, eg in new posts, in more general subreddits


# start with univariate hawkes processes


# generate hawkes process and simulate data
# set up ML to reestimate parameters used in simulations
# only if that works go
# SHOW HIM THE DATA FOR SIMULATED HAWKES PROCESS!!! SEND HIM ONCE I HAVE SOMETHING

# poisson / hawkes
# relate parameter lambda to x variables (also include variables of other subreddits) lambda = exp(Xbeta)
# generalized autoregressive scores: looks at each likelihood contribution: would this be better if I increase the lambda bit; could also model tri-variate poisson process, 3 lambdas, each of those adapts to other likelihood contributions as well.

