
# can get
# posts and comments from particular time intervals and subreddits
# comments that contain a certain word
# posts that contain a certain word

# content is scraped once created, so votes do not reflect current votes
# have to first scrape pushshift, then retrieve votes from reddit api
# for a given user, can create profile of number of posts / comments per subreddit


import praw
import prawcore
from psaw import PushshiftAPI
import datetime as dt
import pandas as pd
import os
from tqdm import tqdm

# turn on logging
import logging

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger = logging.getLogger('psaw')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# define credentials
reddit_username = 'Doxxton'
reddit_pw = 'paradoxx1'
app_id = 'Fb7jNz6yHgK9HloByAVBdA'
app_secret = 'fg3yVGPWVQh7VPnmv7iHWuQn6a4w-w'

# define data path
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# check if necessary folder exist
if not os.path.exists(data_path + '/posts'):
    os.makedirs(data_path + '/posts')
    
# authenticate with reddit api
r = praw.Reddit(client_id = app_id,
                client_secret = app_secret,
                user_agent = 'Webscraper by Markus Mueller',
                username = reddit_username,
                password = reddit_pw)

# assert r.read_only == True, "reddit instance not correctly initialized"

api = PushshiftAPI(r)

# define subreddits to scrape
# subreddits = ['electricvehicles']
subreddits = ['electricvehicles']

subreddits = ['worldnews', 'news']


sub = 'worldnews'
sub = 'news'

sub = 'cars'


# define time period to scrape
start_time = int(dt.datetime(2015, 1, 1).timestamp())
end_time = int(dt.datetime(2021,10,31).timestamp())

for sub in subreddits:
    
    # create generator object for posts in subreddit
    gen = api.search_submissions(subreddit = sub,
                                 after = start_time,
                                 before = end_time,
                                 limit = None,
                                 filter = ['subreddit'],
                                 metadata = True)
    
    
    # search posts containing term
    # query_list = ['climate change', 'natural desaster', 'green energy', 'fridays for future']
    
    gen = api.search_submissions(q = '"climate change"|"natural disaster"|"green energy"', 
                                 limit = None,
                                 after = start_time, 
                                 before = end_time,
                                 subreddit = sub,
                                 filter = ['subreddit'])
    
    
    gen = api.search_submissions(q = '"what car should I buy" + megathread', 
                                 limit = None,
                                 after = start_time, 
                                 before = end_time,
                                 subreddit = sub,
                                 filter = ['subreddit'])
    
    # scrape weekly megathreads of 'what car should i buy'
    
    # retrieve post IDs
    # later used to retrieve up-to-date data from reddit
    print("retrieve post IDs...")
    cache = []
    for post in gen:
        cache.append(post.id)
        
        # if len(cache) >= 100:
        #     break
        
    # save cache
    pd.DataFrame(cache).to_csv(data_path + f'/posts/{sub}_cache.csv', index = False)
    cache = pd.read_csv(data_path + f'/posts/{sub}_cache.csv')['0'].tolist()
  
    print(f"found {len(cache)} posts in r/{sub}")
        
    post_data_list = []

    for post_id in tqdm(cache):

        # retrieve up-to-date reddit data
        post = r.submission(id = post_id)
       
        post_data = [post.id,
                          post.created_utc,
                          post.title,
                          post.selftext, #empty if picture or link
                          post.url, # permalink if selfpost
                          post.permalink,
                          post.subreddit.display_name,
                          post.num_comments,
                          post.score, # num of upvotes
                          post.upvote_ratio, # percentage of upvotes
                          post.edited,
                          post.distinguished,
                          post.spoiler,
                          post.is_original_content,
                          post.link_flair_text,
                          post.stickied,
                          post.over_18]
        
        # collect author data if available (deleted, automod, etc.)
        try:
            if post.author is None:
                post_data.extend([''] * 2)
            else: 
                try:
                    post_data.extend([post.author.id, post.author.name])
                except AttributeError:
                    post_data.extend([''] * 2)
                    
        except prawcore.exceptions.NotFound:
            post_data.extend([''] * 2)
            
        post_data_list.append(post_data)

    post_data = pd.DataFrame(post_data_list)

    var_names = ['id', 'created_utc', 'title', 'selftext',
                'url', 'permalink', 'subreddit', 'num_comments',
                'score', 'upvote_ratio', 'edited', 'distinguished',
                'spoiler', 'is_original_content', 'flair_text',
                'stickied', 'over18', 'author_id', 'author_name']

    post_data.columns = var_names

    # save post data
    post_data.to_json(data_path + f'/posts/{sub}.json', compression = "infer", force_ascii=True)


# test = pd.read_json(data_path + f'/posts/{sub}.json')


##################################################


# check why many author ids and names missing 







# SETUP REDDIT INSTANCE

# load reddit credentials from separate file


# clean data: remove automod and posts that got deleted?
        
# get sentiment of posts towards electric cars?

# what about those posts that have no self-text?

# should also scrape the comments? (but only for electriccars?)

# should also look at r/cars to avoid bias?

# what is the event?! positive sentiments towards electric cars? or just the posting of threads in general?
    
    
    
    
    
    









            'author_create_utc',
            'author_comment_karma', 'author_link_karma',
            'author_submissions', 'author_comments',
            'author_is_gold', 'author_verified_email'
             
# make for each post a dataframe with all comments! account for level of comments (1st level child, 2md level, etc.)
             
# then scrape all author properties for authors of posts and comments from other dataframes

# scrape

# post.comments gives commentforest

# see https://praw.readthedocs.io/en/stable/code_overview/models/submission.html

# how to get subscribers to subreddit over time?
# check https://subredditstats.com/


# can access submission
'reddit.com' + submission.permalink




# iterate through all comments of to this subission
# also save level of comment (top-level, second, etc.)
submission.comments.replace_more(limit=None)

submission.permalink
    
    
    print(vars(comment))
    
    
list(vars(submission.comments.list()[0]))