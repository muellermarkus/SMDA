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
reddit_username = ''
reddit_pw = ''
app_id = ''
app_secret = ''

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
subreddits = ['worldnews', 'news', 'electricvehicles']

# define time period to scrape
start_time = int(dt.datetime(2015, 1, 1).timestamp())
end_time = int(dt.datetime(2021,10,31).timestamp())

for sub in subreddits:
    
    if sub == 'news' or sub == 'worldnews':
        gen = api.search_submissions(q = '"climate change"|"natural disaster"|"green energy"', 
                                    limit = None,
                                    after = start_time, 
                                    before = end_time,
                                    subreddit = sub,
                                    filter = ['subreddit'])
    else:
        # scrape weekly megathreads of 'what car should i buy'
        gen = api.search_submissions(q = '"what car should I buy" + megathread', 
                                    limit = None,
                                    after = start_time, 
                                    before = end_time,
                                    subreddit = sub,
                                    filter = ['subreddit'])
    
    # retrieve post IDs
    # later used to retrieve up-to-date data from reddit
    print("retrieve post IDs...")
    cache = []
    for post in gen:
        cache.append(post.id)
        
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
