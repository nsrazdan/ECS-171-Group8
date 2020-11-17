import requests, sys, time, os
import asyncio
import aiohttp
import datetime
import dateutil.parser as parser
import numpy as np
import pandas as pd

# Inspired by Mitchell J's trending video script (https://github.com/mitchelljy/Trending-YouTube-Scraper)
# modified to gather nontrending videos from the same time interval

# List of simple to collect features
snippet_features = ["title", "publishedAt", "channelId", "channelTitle"]
content_features = ["duration", "dimension", "definition"]

channel_snippet_atts = ["title", "description", "publishedAt", "country"]
channel_stats_atts = [
    "viewCount",
    "subscriberCount",
    "hiddenSubscriberCount",
    "videoCount",
]

channel_columns = [
    "Channel_" + att for att in channel_stats_atts + channel_snippet_atts
]

# Any characters to exclude, generally these are things that become problematic in CSV files
unsafe_characters = ["\n", '"']
country_codes = ["US"]

# source file for trending videos. Used to determine time range.
key_path = "allkeys.txt"

# status codes
switching_key = -1
no_more_keys = -2

# Used to identify columns, currently hardcoded order
header = (
    ["video_id"]
    + snippet_features
    + ["categoryId"]
    + content_features
    + [
        "is_trending",
        "time_retrieved",
        "tags",
        "view_count",
        "likes",
        "dislikes",
        "comment_count",
        "thumbnail_link",
        "comments_disabled",
        "ratings_disabled",
        "description",
    ]
    + channel_columns
)


def advance_time(start_time, end_time):
    return (end_time, end_time + delta)


def next_key():
    if len(allkeys) > 0:
        return allkeys.pop(0)
    else:
        return no_more_keys


def setup(api_path):
    with open(api_path) as file:
        allkeys = [x.rstrip() for x in file]
    return allkeys


def prepare_feature(feature):
    # Removes any character from the unsafe characters list and surrounds the whole item in quotes
    for ch in unsafe_characters:
        feature = str(feature).replace(ch, "")
    return f'"{feature}"'


async def video_request(id_string):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=id,statistics,contentDetails,snippet&id={id_string}&key={last_key}"
    async with aiohttp.ClientSession() as session:
        async with await session.get(url) as response:
            return await response.json()


async def channel_request(id_string):
    url = f"https://www.googleapis.com/youtube/v3/channels?part=id,statistics,snippet&id={id_string}&key={last_key}"
    async with aiohttp.ClientSession() as session:
        async with await session.get(url) as response:
            return await response.json()


def api_request(page_token, country_code, start_time, end_time):
    # Builds the URL and requests the JSON from it
    global api_key
    start = f"{start_time.date()}T{start_time.time()}Z"
    end = f"{end_time.date()}T{end_time.time()}Z"
    request_url = f"https://www.googleapis.com/youtube/v3/search?part=id,snippet{page_token}&type=video&publishedAfter={start}&publishedBefore={end}&order=date&regionCode=US&relevanceLanguage=en&maxResults=50&key={api_key}"
    request = requests.get(request_url)
    if request.status_code == 429:
        print("Temp-Banned due to excess requests, please wait and continue later")
        sys.exit()
    if request.status_code == 403:
        print("Quota Exceeded")
        print(request.json())
        api_key = next_key()
        if api_key == no_more_keys:
            return no_more_keys
        return switching_key
    if request.status_code != 200:
        print(request.json())
    return request.json()


def get_tags(tags_list):
    # Takes a list of tags, prepares each tag and joins them into a string by the pipe character
    return prepare_feature("|".join(tags_list))


async def get_videos(items):
    lines = []
    ids = [video["id"]["videoId"] for video in items]
    id_string = ",".join(ids)
    c_ids = list(set([video["snippet"]["channelId"] for video in items]))
    c_id_string = ",".join(c_ids)
    if len(ids) == 0 or len(c_ids) == 0:
        return []
    video_data, channel_data = await asyncio.gather(
        video_request(id_string), channel_request(c_id_string)
    )
    video_data = video_data["items"]
    channel_data = channel_data["items"]
    for video in items:

        # We can assume something is wrong with the video if it has no statistics, often this means it has been deleted
        # so we can just skip it

        # A full explanation of all of these features can be found on the GitHub page for this project
        video_id = video["id"]["videoId"]
        channel_id = video["snippet"]["channelId"]

        for i in video_data:
            if i["id"] == video_id:
                this_video_data = i
                break
        for i in channel_data:
            if i["id"] == channel_id:
                this_channel_data = i
                break

        # Snippet and statistics are sub-dicts of video, containing the most useful info
        snippet = video["snippet"]
        video_details = this_video_data["contentDetails"]
        video_statistics = this_video_data["statistics"]
        channel_snippet = this_channel_data["snippet"]
        channel_statistics = this_channel_data["statistics"]

        # This list contains all of the features in snippet that are 1 deep and require no special processing
        features = [
            prepare_feature(snippet.get(feature, "")) for feature in snippet_features
        ]
        details_features = [
            prepare_feature(video_details.get(feature, ""))
            for feature in content_features
        ]

        # The following are special case features which require unique processing, or are not within the snippet dict
        description = snippet.get("description", "")
        thumbnail_link = (
            snippet.get("thumbnails", dict()).get("default", dict()).get("url", "")
        )
        is_trending = False
        time_retrieved = time.strftime("20%y-%m-%dT%H:%M:%SZ")
        tags = get_tags(this_video_data["snippet"].get("tags", ["[none]"]))
        category_id = prepare_feature(this_video_data["snippet"].get("categoryId", ""))

        view_count = video_statistics.get("viewCount", 0)

        ratings_disabled = False
        if "likeCount" in video_statistics and "dislikeCount" in video_statistics:
            likes = video_statistics["likeCount"]
            dislikes = video_statistics["dislikeCount"]
        else:
            ratings_disabled = True
            likes = 0
            dislikes = 0

        comments_disabled = False
        if "commentCount" in video_statistics:
            comment_count = video_statistics["commentCount"]
        else:
            comments_disabled = True
            comment_count = 0
        channel_stats_features = [
            channel_statistics.get(feature, np.NaN) for feature in channel_stats_atts
        ]
        channel_snippet_features = [
            prepare_feature(channel_snippet.get(feature, ""))
            for feature in channel_snippet_atts
        ]

        # Compiles all of the various bits of info into one consistently formatted line
        line = (
            [video_id]
            + features
            + [category_id]
            + details_features
            + [
                prepare_feature(x)
                for x in [
                    is_trending,
                    time_retrieved,
                    tags,
                    view_count,
                    likes,
                    dislikes,
                    comment_count,
                    thumbnail_link,
                    comments_disabled,
                    ratings_disabled,
                    description,
                ]
                + channel_stats_features
                + channel_snippet_features
            ]
        )
        lines.append(",".join(line))
    return lines


async def get_pages(country_code, next_page_token="&"):
    country_data = []
    start_time = true_start
    end_time = start_time + delta
    # Because the API uses page tokens (which are literally just the same function of numbers everywhere) it is much
    # more inconvenient to iterate over pages, but that is what is done here.
    while start_time < true_end:
        while next_page_token is not None:
            # A page of data i.e. a list of videos and all needed data
            video_data_page = api_request(
                next_page_token, country_code, start_time, end_time
            )
            if video_data_page == switching_key:
                print("Switching key. Total videos:", len(country_data))
                print("Current start time: ", start_time)
                break
            if video_data_page == no_more_keys:
                return country_data
            # Get the next page token and build a string which can be injected into the request with it, unless it's None,
            # then let the whole thing be None so that the loop ends after this cycle
            next_page_token = video_data_page.get("nextPageToken", None)
            next_page_token = (
                f"&pageToken={next_page_token}&"
                if next_page_token is not None
                else next_page_token
            )

            # Get all of the items as a list and let get_videos return the needed features
            items = video_data_page.get("items", [])
            country_data += await get_videos(items)
            if len(items) < 50:
                break
        next_page_token = "&"
        start_time, end_time = advance_time(start_time, end_time)
    return country_data


def write_to_file(country_code, country_data):

    print(f"Writing {country_code} data to file...")

    with open(f"{time.strftime('%m.%d recent')}.csv", "w+", encoding="utf-8",) as file:
        for row in country_data:
            file.write(f"{row}\n")


def get_data():
    for country_code in country_codes:
        pages = asyncio.run(get_pages(country_code))
        country_data = [",".join(header)] + pages
        write_to_file(country_code, country_data)


allkeys = setup(key_path)
api_key = allkeys.pop(0)
last_key = allkeys[len(allkeys) - 1]

true_end = datetime.datetime.now()
true_start = true_end - datetime.timedelta(hours=24)
delta = datetime.timedelta(minutes=30)

get_data()

