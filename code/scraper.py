import requests

# define credentials
reddit_username = 'Doxxton'
reddit_pw = 'paradoxx1'
app_id = 'Fb7jNz6yHgK9HloByAVBdA'
app_secret = 'fg3yVGPWVQh7VPnmv7iHWuQn6a4w-w'

# retrieve OAUTH token
base_url = 'https://www.reddit.com/'
data = {'grant_type': 'password', 'username': reddit_username, 'password': reddit_pw}
auth = requests.auth.HTTPBasicAuth(app_id, app_secret)
r = requests.post(base_url + 'api/v1/access_token',
                  data=data,
                  headers={'user-agent': f'Webscrape by {reddit_username}'}, 
                  auth=auth)
d = r.json()


#

token = 'bearer ' + d['access_token']
api_url = 'https://oauth.reddit.com'
headers = {'Authorization': token, 'User-Agent': 'APP-NAME by REDDIT-USERNAME'}

    



# specify subreddits to be scraped
subreddits = ['superstonk']

subreddit = 'superstonk'

payload = {'q': subreddit, 'limit': 1, 'sort': 'relevance'}

response = requests.get(api_url + '/subreddits/search', headers=headers, params=payload)
print(response.status_code)

for child in response.json()['data']['children']:
    title = child['title']
    print(title)
    
    
    
    
    
    
    
payload = {'sr_name': subreddit}
response = requests.get(api_url + '/api/info', headers=headers, params=payload)

# retrieve subreddit id
subreddit_id = response.json()['data']['children'][0]['data']['name']

# for id scrape all posts in time frame





print(response.status_code)

response.json()



# scrape username, title, post, votes, date, awards




len(response.json()['data']['children'])
