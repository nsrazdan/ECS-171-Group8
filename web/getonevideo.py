import requests
import pandas as pd
import numpy as np
import time


def prepare_feature(feature):
    # Removes any character from the unsafe characters list and surrounds the whole item in quotes
    unsafe_characters = ["\n", '"']
    for ch in unsafe_characters:
        feature = str(feature).replace(ch, "")
    return f'"{feature}"'


def get_video(video_id):
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

    column_names = (
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

    # get the api key from a file api_key.txt to avoid putting it in the code
    with open("api_key.txt", "r") as file:
        api_key = file.readline()

    response = requests.get(
        f"https://www.googleapis.com/youtube/v3/videos?part=id,statistics,contentDetails,snippet&id={video_id}&key={api_key}"
    )
    if response.status_code != 200:
        print(response.json())
        # or throw an exception or something, we don't have the data
        return "error"
    video = response.json()["items"][0]
    video_id = video["id"]
    channel_id = video["snippet"]["channelId"]
    response = requests.get(
        f"https://www.googleapis.com/youtube/v3/channels?part=id,statistics,snippet&id={channel_id}&key={api_key}"
    )
    if response.status_code != 200:
        print(response.json())
        # or throw an exception or something, we don't have the data
        return "error"
    channel = response.json()["items"][0]

    # Snippet and statistics are sub-dicts of video, containing the most useful info
    snippet = video["snippet"]
    statistics = video["statistics"]
    details = video["contentDetails"]
    channel_snippet = channel["snippet"]
    channel_statistics = channel["statistics"]

    # This list contains all of the features in snippet that are 1 deep and require no special processing
    features = [
        prepare_feature(snippet.get(feature, "")) for feature in snippet_features
    ]
    details_features = [
        prepare_feature(details.get(feature, "")) for feature in content_features
    ]

    # The following are special case features which require unique processing, or are not within the snippet dict
    description = snippet.get("description", "")
    thumbnail_link = (
        snippet.get("thumbnails", dict()).get("default", dict()).get("url", "")
    )
    is_trending = False
    time_retrieved = time.strftime("20%y-%m-%dT%H:%M:%SZ")
    tags_list = snippet.get("tags", ["[none]"])
    tags = prepare_feature("|".join(tags_list))
    category_id = prepare_feature(snippet.get("categoryId", ""))

    view_count = statistics.get("viewCount", 0)

    ratings_disabled = False
    if "likeCount" in statistics and "dislikeCount" in statistics:
        likes = statistics["likeCount"]
        dislikes = statistics["dislikeCount"]
    else:
        ratings_disabled = True
        likes = 0
        dislikes = 0

    comments_disabled = False
    if "commentCount" in statistics:
        comment_count = statistics["commentCount"]
    else:
        comments_disabled = True
        comment_count = 0
    channel_stats_features = [
        channel_statistics.get(feature, 0) for feature in channel_stats_atts
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
    data = pd.DataFrame(data=np.reshape(line, (1, -1)), columns=column_names)
    return data
