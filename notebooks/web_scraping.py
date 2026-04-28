import requests
import pandas as pd
import time

#def scrape_steam_to_match_format(appid, num_reviews=3000, output_name="stardew_valley_reviews.csv"):
def scrape_steam_to_match_format(appid, num_reviews=3000, output_name="Divinity_Original_Sin_2_reviews.csv"):
    url = f"https://store.steampowered.com/appreviews/{appid}?json=1"
    params = {
        'filter': 'all',
        'language': 'english',
        'cursor': '*',
        'num_per_page': 100
    }

    review_list = []

    while len(review_list) < num_reviews:
        try:
            res = requests.get(url, params=params).json()
            if res.get('success') != 1 or not res.get('reviews'):
                break

            for r in res['reviews']:
                review_list.append({
                    'recommendationid': r.get('recommendationid'),
                    'language': r.get('language'),
                    'review': r.get('review'),
                    'timestamp_created': r.get('timestamp_created'),
                    'timestamp_updated': r.get('timestamp_updated'),
                    'voted_up': r.get('voted_up'),
                    'votes_up': r.get('votes_up'),
                    'votes_funny': r.get('votes_funny'),
                    'weighted_vote_score': r.get('weighted_vote_score'),
                    'written_during_early_access': r.get('written_during_early_access'),
                    'comment_count': r.get('comment_count'),
                    'steam_purchase': r.get('steam_purchase'),
                    'received_for_free': r.get('received_for_free')
                })

            params['cursor'] = res['cursor']
            print(f"Captured {len(review_list)} / {num_reviews}...")

            time.sleep(1)

        except Exception as e:
            print(f"Error: {e}")
            break

    df = pd.DataFrame(review_list)
    df = df.head(num_reviews)

    df.to_csv(output_name, index=False, encoding='utf-8-sig')
    print(f"The file has been saved as: {output_name}")

if __name__ == "__main__":
    # 星露谷物语 AppID: 413150
    # scrape_steam_to_match_format(413150, num_reviews=5000)
    # 神界原罪2：435150
    scrape_steam_to_match_format(435150, num_reviews=5000)