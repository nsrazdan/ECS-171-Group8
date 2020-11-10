import asyncio
import aiohttp
import time
import pandas as pd
import numpy as np

api_key = ""
req_size = 49

# augment dframe with new columns
def add_columns(dframe, index, columns):
    for col in columns:
        if col not in dframe.columns:
            dframe.insert(index, col, np.NaN)
            index += 1
    dframe.insert(index, f"{today}_update_timestamp", "")


# prepare all request urls ahead of time based on ids
def prepare_urls(allids):
    start = 0
    end = req_size
    urls = []
    while start != len(allids):
        id_string = ",".join(allids[start:end])
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


async def update(dframe):
    # get list of video ids and dictionary of video id: index
    allids = dframe.loc[:, "video_id"].to_list()
    video_id_dict = {}
    for i in dframe.index:
        video_id_dict[dframe.loc[i, "video_id"]] = i

    urls = prepare_urls(allids)
    failed_urls = []
    now = time.strftime("20%y-%m-%dT%H:%M:%SZ")

    # make all requests simultaneously and return array of responses
    responses = await asyncio.gather(*[make_request(url) for url in urls])
    for r in responses:
        response = r["response"]
        if r["status"] != 200:
            failed_urls += [r["url"]]
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
        if len(failed_urls) > 0:
            with open("failed.txt", "w+", encoding="utf-8") as file:
                for row in failed_urls:
                    file.write(f"{row}\n")
    dframe.to_csv(f"{data[0:-4]}-updated.csv", columns=dframe.columns.to_list()[1:])


today = time.strftime("%m_%d_%H")
columns = [
    col + f"_update_{today}"
    for col in ["view_count", "likes", "dislikes", "comment_count"]
]

data = "11.09 trending.csv"
dfx = pd.read_csv(f"./output/{data}")
add_columns(dfx, len(dfx.columns), columns)
asyncio.run(update(dfx))
