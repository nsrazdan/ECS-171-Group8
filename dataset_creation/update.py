import asyncio
import aiohttp
import time
import pandas as pd
import numpy as np

api_key = "AIzaSyB1gOa2afwLgO6S7LT8Jfby56D9vA-W_kg"
req_size = 49

stats_atts = ["viewCount", "subscriberCount", "videoCount"]

# augment dframe with new columns
def add_columns(dframe, index, columns):
    for col in columns:
        if col not in dframe.columns:
            dframe.insert(index, col, np.NaN)
            index += 1


# prepare all request urls ahead of time based on ids
def prepare_urls(allids, channels=False):
    start = 0
    end = req_size
    urls = []
    while start != len(allids):
        id_string = ",".join(allids[start:end])
        if channels:
            urls += [
                f"https://www.googleapis.com/youtube/v3/channels?part=statistics&id={id_string}&key={api_key}"
            ]
        else:
            urls += [
                f"https://www.googleapis.com/youtube/v3/videos?part=statistics&id={id_string}&key={api_key}"
            ]
        start = end
        end = min(end + req_size, len(allids))
    return urls


async def make_request(url):
    async with aiohttp.ClientSession() as session:
        async with await session.get(url) as response:
            json = await response.json()
            return {"response": json, "status": response.status, "url": url}


async def c_update(dframe, name):
    add_columns(dframe, len(dframe.columns), channel_columns)
    allids = list(set(dframe.loc[:, "channelId"].to_list()))

    c_urls = prepare_urls(allids, channels=True)
    channel_id_dict = {}
    for i in dframe.index:
        if dframe.loc[i, "channelId"] not in channel_id_dict:
            channel_id_dict[dframe.loc[i, "channelId"]] = [i]
        else:
            channel_id_dict[dframe.loc[i, "channelId"]] += [i]
    c_responses = await asyncio.gather(*[make_request(url) for url in c_urls])

    for r in c_responses:
        response = r["response"]
        if r["status"] != 200:
            print(response)
            continue
        # update dframe
        for channel in response["items"]:
            statistics = channel["statistics"]
            cid = channel["id"]
            for att in stats_atts:
                dframe.at[
                    channel_id_dict[cid], f"Channel_{att}_update_{today}"
                ] = statistics.get(att, 0)


async def update(dframe, name, withchannels=False):
    add_columns(dframe, len(dframe.columns), columns)
    if withchannels:
        await c_update(dframe, name)
    if f"{today}_update_timestamp" not in dframe:
        dframe.insert(len(dframe.columns), f"{today}_update_timestamp", "")

    # get list of video ids and dictionary of video id: index
    allids = dframe.loc[:, "video_id"].to_list()
    video_id_dict = {}
    for i in dframe.index:
        video_id_dict[dframe.loc[i, "video_id"]] = i

    urls = prepare_urls(allids)

    now = time.strftime("20%y-%m-%dT%H:%M:%SZ")
    # make all requests simultaneously and return array of responses
    responses = await asyncio.gather(*[make_request(url) for url in urls])

    for r in responses:
        response = r["response"]
        if r["status"] != 200:
            print(response)
            continue
        # update dframe
        for video in response["items"]:
            statistics = video["statistics"]
            vid = video["id"]
            dframe.at[
                video_id_dict[vid], f"view_count_update_{today}"
            ] = statistics.get("viewCount", 0)
            dframe.at[video_id_dict[vid], f"likes_update_{today}"] = statistics.get(
                "likeCount", 0
            )
            dframe.at[video_id_dict[vid], f"dislikes_update_{today}"] = statistics.get(
                "dislikeCount", 0
            )
            dframe.at[
                video_id_dict[vid], f"comment_count_update_{today}"
            ] = statistics.get("commentCount", 0)
            dframe.at[video_id_dict[vid], f"{today}_update_timestamp"] = now
        # if any requests failed, save those urls so they can be repeated
    dframe.to_csv(name, columns=dframe.columns.to_list()[1:])


today = time.strftime("%m_%d_%H")
print(today)
columns = [
    col + f"_update_{today}"
    for col in ["view_count", "likes", "dislikes", "comment_count"]
]

channel_columns = [f"Channel_{col}_update_{today}" for col in stats_atts]

trending = "11.09 trending.csv"
nontrending = "11.09 nontrending.csv"
dft = pd.read_csv(f"./datasets/{trending}")
dfn = pd.read_csv(f"./datasets/{nontrending}")
asyncio.run(update(dft, trending))
asyncio.run(update(dfn, nontrending))

# recent = "11.16 recent.csv"
# dfr = pd.read_csv(f"./datasets/{recent}")
# asyncio.run(update(dfr, recent, withchannels=True))

exec(open("./dataset_creation/gettrending.py").read())
