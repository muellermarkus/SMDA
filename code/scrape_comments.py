import os
import pandas as pd
from tqdm import tqdm
import praw
import prawcore
import datetime as dt

# define credentials
reddit_username = 'Doxxton'
reddit_pw = 'paradoxx1'
app_id = 'Fb7jNz6yHgK9HloByAVBdA'
app_secret = 'fg3yVGPWVQh7VPnmv7iHWuQn6a4w-w'

# define data path
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
# authenticate with reddit api
r = praw.Reddit(client_id = app_id,
                client_secret = app_secret,
                user_agent = 'Webscraper by Markus Mueller',
                username = reddit_username,
                password = reddit_pw)

# scrape all comments on previously scrape posts
subreddits = ['electricvehicles', 'cars']

for sub in subreddits:
    
    print(f"scrape comments from {sub}...")

    post_data = pd.read_json(data_path + f'/posts/{sub}.json')
    post_data = list(post_data["id"])

    # check if necessary folder exist
    if not os.path.exists(data_path + f'/comments/{sub}'):
        os.makedirs(data_path + f'/comments/{sub}')

    for post_id in tqdm(post_data):

        comment_data_list = []

        num_comment = 1
        
        # extend 'more comments' objects
        post = r.submission(id = post_id)
        post.comments.replace_more(limit=None)

        for comment in post.comments.list():
            
            if 't3' in comment.parent_id:
                toplevel = 1
            else:
                toplevel = 0
            
            comment_data = [comment.id,
                            comment.parent_id,
                            comment.created_utc,
                            comment.body_html,
                            toplevel,
                            comment.distinguished,
                            comment.edited,
                            comment.is_submitter,
                            comment.score,
                            comment.stickied]
            
            try:
                if comment.author is None:
                    comment_data.extend([''] * 2)
                else:
                    print(comment.body_html)
                    print(comment.id)
                    if comment.author.id is None:
                        comment_data.extend([''] * 2)
                    else:
                        comment_data.extend([comment.author.id, comment.author.name])
            except prawcore.exceptions.NotFound:
                comment_data.extend([''] * 2)
            
            comment_data_list.append(comment_data)
            
        comment_df = pd.DataFrame(comment_data_list)
        
        if len(comment_data_list) == 0:
            continue
        
        var_names = ['comment_id', 'comment_parent_id',
                    'comment_created_utc', 'comment_body',
                    'comment_toplevel', 'comment_distinguished',
                    'comment_edited', 'comment_is_submitter',
                    'comment_score', 'comment_stickied',
                    'author_id', 'author_name']
        
        comment_df.columns = var_names
        
        # save comment data for a given post
        comment_df.to_json(data_path + '/comments' + f'/{sub}/{post_id}.json', compression = "infer", force_ascii=True)
        
        # as gzip csv file
        
        # how to retrieve comments from api itself, up to which level?
        
        
        
        
        
        # comment_data.to_parquet(data_path + '/comments' + f'/{sub}/{post_id}.gzip', compression='gzip')
        
        
        
# no comments for electricvehicles?