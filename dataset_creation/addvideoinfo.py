channel_req_size = 49
import requests
import pandas as pd
import numpy as np

key_path = "allkeys.txt"
api_key = ""
unsafe_characters = ["\n", '"']


def setkey():
    with open(key_path) as file:
        allkeys = [x.rstrip() for x in file]
    return allkeys.pop()


# Credit to Mitchell J
def prepare_feature(feature):
    # Removes any character from the unsafe characters list and surrounds the whole item in quotes
    for ch in unsafe_characters:
        feature = str(feature).replace(ch, "")
    return f'"{feature}"'


def get_tags(tags_list):
    # Takes a list of tags, prepares each tag and joins them into a string by the pipe character
    return prepare_feature("|".join(tags_list))


# get missing data for videos found through search
def add_video_data(dframe):
    allids = dframe.loc[:, "video_id"].to_list()

    start = 0
    end = channel_req_size
    while start != len(allids):
        id_string = ",".join(allids[start:end])
        get_video_data = f"https://www.googleapis.com/youtube/v3/videos?part=id,statistics,snippet&id={id_string}&key={api_key}"

        response = requests.get(get_video_data)
        if response.status_code != 200:
            print(response.json())
            return
        videos = response.json()["items"]
        for i in range(len(videos)):
            # if statistics not found don't bother, as in original script
            if "statistics" not in videos[i]:
                continue
            statistics = videos[i]["statistics"]
            snippet = videos[i]["snippet"]
            vid = videos[i]["id"]
            modify = dframe.loc[dframe["video_id"] == vid]

            modify["view_count"] = statistics.get("viewCount", 0)
            modify["categoryId"] = snippet.get("categoryId", np.NaN)
            modify["tags"] = get_tags(snippet.get("tags", ["[none]"]))

            if "likeCount" in statistics and "dislikeCount" in statistics:
                modify["likes"] = statistics["likeCount"]
                modify["dislikes"] = statistics["dislikeCount"]
                modify["ratings_disabled"] = False
            else:
                modify["ratings_disabled"] = True
                modify["likes"] = 0
                modify["dislikes"] = 0

            if "commentCount" in statistics:
                modify["comment_count"] = statistics["commentCount"]
                modify["comments_disabled"] = False
            else:
                modify["comment_count"] = 0
                modify["comments_disabled"] = True
            dframe.loc[dframe["video_id"] == vid] = modify
        start = end
        end = min(end + channel_req_size, len(allids))


api_key = setkey()
nontrending = "05.18_videos.csv"
df2 = pd.read_csv(f"./output/{nontrending}")
add_video_data(df2)
df2.to_csv(f"{nontrending}-withvdata.csv")
