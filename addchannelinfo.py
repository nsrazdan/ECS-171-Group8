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


def add_columns(dframe, columns):
    for col in columns:
        if col not in dframe.columns:
            dframe.insert(len(dframe.columns), col, np.NaN)


# get missing data for videos found through search
def add_channel_data(dframe):
    stats_atts = ["viewCount", "subscriberCount", "hiddenSubscriberCount", "videoCount"]
    snippet_atts = ["title", "description", "publishedAt", "country"]

    columns = ["Channel_" + att for att in stats_atts + snippet_atts]
    add_columns(dframe, columns)

    allids = set(dframe.loc[:, "channelId"].to_list())
    allids = list(allids)

    start = 0
    end = channel_req_size
    while start != len(allids):
        id_string = ",".join(allids[start:end])
        get_channel_data = f"https://www.googleapis.com/youtube/v3/channels?part=id,statistics,snippet&id={id_string}&key={api_key}"

        response = requests.get(get_channel_data)

        channels = response.json()["items"]
        for i in range(len(channels)):
            cid = channels[i]["id"]
            for j in stats_atts:
                dframe.loc[dframe["channelId"] == cid, "Channel_" + j] = channels[i][
                    "statistics"
                ].get(j, np.NaN)
            for j in snippet_atts:
                feature = prepare_feature(channels[i]["snippet"].get(j, ""))
                dframe.loc[dframe["channelId"] == cid, "Channel_" + j] = feature

        start = end
        end = min(end + channel_req_size, len(allids))


api_key = setkey()
data = "11.09-nontrending-withvdata.csv"
df2 = pd.read_csv(f"./output/{data}")
add_channel_data(df2)
newfname = data[0:-4] + "-withcdata.csv"
df2.to_csv(f"./output/{newfname}")
