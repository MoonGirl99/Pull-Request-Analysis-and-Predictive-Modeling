import pandas as pd
import requests
import time

PER_PAGE = 100
RATE_LIMIT = 5000


def get_pull_requests(repo_url):
    owner, repo = repo_url.split('/')[-2:]
    parameters = {"state": "all", "per_page": 100}
    headers = {'Authorization': 'Bearer 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'}
    response = requests.get(repo_url, headers=headers, params=parameters)

    if response.status_code != 200:
        print(f'Error: {response.status_code}')
        return None

    pull_requests = response.json()
    pe = []

    for pr in pull_requests:

        if bool(pr['merged_at']) == True:
            pr['merged_at'] = 1
        elif bool(pr['merged_at']) == False:
            pr['merged_at'] = 0

        if pr['state'] == "closed":
            pr['state'] = 1
        elif pr['state'] == "open":
            pr['state'] = 0

        user_id = pr['user']['id']
        comments_url = pr["comments_url"]
        review_comments_url = pr["review_comments_url"]

        comments = get_comment(comments_url, headers)
        review_comments = get_comment(review_comments_url, headers)

        pe.append({
            'number': int(pr["number"]),
            'user_id': user_id,
            'title': pr["title"],
            'comments': comments,
            'review_comments': review_comments,
            'commits_url': pr["commits_url"],
            'status': pr["state"],
            'created_at': pr['created_at'],
            'pr_merged': int(pr['merged_at']),
            'closed_at': pr['closed_at'],
        })
        # Respect API rate limiting from Github
        time.sleep(60/RATE_LIMIT)
    return pe


def get_comment_count(comments):
    return len(comments)


def get_commit_count(commits):
    return len(commits)


def get_des_count(description):
    return len(description)


def get_comment(comments_url, headers):
    parameters = {"per_page": 100}
    response = requests.get(comments_url, headers=headers, params=parameters)

    comments = response.json()
    all_comments = []

    for curr_comm in comments:
        if 'body' in curr_comm:
            
            all_comments += [curr_comm['body']]

    return "\n".join(all_comments)


req = get_pull_requests('https://api.github.com/repos/apache/commons-math/pulls')
rows = []
for re in req:


    comments = re['comments']
    review_comments = re['review_comments']
    commit_url = re['commits_url']
    title = re['title']

    comment_length = get_comment_count(comments)
    review_comment_length = get_comment_count(review_comments)
    commit_url_length = get_commit_count(commit_url)
    title_length = get_comment_count(title)


    rows.append({
        'number': re['number'],
        'title': re['title'],
        'user_id': re['user_id'],
        'title_length': title_length,
        'comments': comments,
        'review_comments': review_comments,
        'comments_length': comment_length,
        'commit_url': re["commits_url"],
        'commit_url_length': commit_url_length,
        'review_comments_length': review_comment_length,
        'status': re['status'],
        'created_at': re['created_at'],
        'pr_merged': re['pr_merged'],
        'closed_at': re['closed_at'],
    })


df = pd.DataFrame(rows)
df.to_csv('commons-math-last.csv', index=False)
