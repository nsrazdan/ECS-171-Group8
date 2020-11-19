channel_req_size = 49
import requests
import pandas as pd
import numpy as np

# Deprecated. Future data sets should have these attributes from the start

key_path = "allkeys.txt"
api_key = ""
unsafe_characters = ["\n", '"']


content_features = ["duration", "dimension", "definition"]


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
    video_id_dict = {}
    for i in dframe.index:
        video_id_dict[dframe.loc[i, "video_id"]] = i

    start = 0
    end = channel_req_size
    while start != len(allids):
        print(start)
        id_string = ",".join(allids[start:end])
        get_video_data = f"https://www.googleapis.com/youtube/v3/videos?part=id,statistics,contentDetails,snippet&id={id_string}&key={api_key}"

        response = requests.get(get_video_data)
        if response.status_code != 200:
            print(response.json())
            return
        videos = response.json()["items"]
        for video in videos:
            # if statistics not found don't bother, as in original script
            if "statistics" not in video:
                continue
            statistics = video["statistics"]
            snippet = video["snippet"]
            details = video["contentDetails"]

            vid = video["id"]
            dframe.at[video_id_dict[vid], "view_count"] = statistics.get("viewCount", 0)
            dframe.at[video_id_dict[vid], "categoryId"] = snippet.get(
                "categoryId", np.NaN
            )
            dframe.at[video_id_dict[vid], "tags"] = get_tags(
                snippet.get("tags", ["[none]"])
            )

            for i in content_features:
                dframe.at[video_id_dict[vid], i] = prepare_feature(details.get(i, ""))

            if "likeCount" in statistics and "dislikeCount" in statistics:
                dframe.at[video_id_dict[vid], "likes"] = statistics["likeCount"]
                dframe.at[video_id_dict[vid], "dislikes"] = statistics["dislikeCount"]
                dframe.at[video_id_dict[vid], "ratings_disabled"] = False
            else:
                dframe.at[video_id_dict[vid], "ratings_disabled"] = True
                dframe.at[video_id_dict[vid], "likes"] = 0
                dframe.at[video_id_dict[vid], "dislikes"] = 0

            if "commentCount" in statistics:
                dframe.at[video_id_dict[vid], "comment_count"] = statistics[
                    "commentCount"
                ]
                dframe.at[video_id_dict[vid], "comments_disabled"] = False
            else:
                dframe.at[video_id_dict[vid], "comment_count"] = 0
                dframe.at[video_id_dict[vid], "comments_disabled"] = True
        start = end
        end = min(end + channel_req_size, len(allids))


api_key = setkey()
nontrending = "11.09-nontrending.csv"
df2 = pd.read_csv(f"./output/{nontrending}")
add_video_data(df2)
newfname = nontrending[0:-4] + "-withcdata.csv"

df2.to_csv(f"./output/{newfname}", columns=df2.columns.to_list()[1:])
