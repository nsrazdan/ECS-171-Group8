import requests, sys, time, os
import datetime
import dateutil.parser as parser
import numpy as np
import pandas as pd

# Inspired by Mitchell J's trending video script (https://github.com/mitchelljy/Trending-YouTube-Scraper)
# modified to gather nontrending videos from the same time interval

# List of simple to collect features
snippet_features = ["title", "publishedAt", "channelId", "channelTitle", "categoryId"]

# Any characters to exclude, generally these are things that become problematic in CSV files
unsafe_characters = ["\n", '"']
country_codes = ["US"]

# source file for trending videos. Used to determine time range.
trending_fname = "20.06.11.15.22.06_US_videos-augmented.csv"
key_path = "allkeys.txt"
output_dir = "output/"

# status codes
switching_key = -1
no_more_keys = -2

# Used to identify columns, currently hardcoded order
header = (
    ["video_id"]
    + snippet_features
    + [
        "trending_date",
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
)

# get time range from trending video file
def get_time_range():
    df = pd.read_csv(f"./{output_dir}{trending_fname}")
    start_time = parser.isoparse(df["publishedAt"].min())
    end_time = parser.isoparse(df["publishedAt"].max())
    return (start_time, end_time)


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


def api_request(page_token, country_code, start_time, end_time):
    # Builds the URL and requests the JSON from it
    global api_key
    start = f"{start_time.date()}T{start_time.time()}Z"
    end = f"{end_time.date()}T{end_time.time()}Z"
    print(start)
    print(end)
    request_url = f"https://www.googleapis.com/youtube/v3/search?part=id,snippet{page_token}&type=video&publishedAfter={start}&publishedBefore={end}&order=date&regionCode=US&relevanceLanguage=en&maxResults=50&key={api_key}"
    request = requests.get(request_url)
    print(request.status_code)
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


def get_videos(items):
    lines = []
    for video in items:

        # We can assume something is wrong with the video if it has no statistics, often this means it has been deleted
        # so we can just skip it

        # A full explanation of all of these features can be found on the GitHub page for this project
        video_id = prepare_feature(video["id"]["videoId"])

        # Snippet and statistics are sub-dicts of video, containing the most useful info
        snippet = video["snippet"]

        # This list contains all of the features in snippet that are 1 deep and require no special processing
        features = [
            prepare_feature(snippet.get(feature, "")) for feature in snippet_features
        ]

        # The following are special case features which require unique processing, or are not within the snippet dict
        description = snippet.get("description", "")
        thumbnail_link = (
            snippet.get("thumbnails", dict()).get("default", dict()).get("url", "")
        )
        trending_date = np.NaN
        tags = get_tags(snippet.get("tags", ["[none]"]))

        view_count = np.NaN
        likes = np.NaN
        dislikes = np.NaN
        ratings_disabled = np.NaN
        comments_disabled = np.NaN
        comment_count = np.NaN

        # Compiles all of the various bits of info into one consistently formatted line
        line = (
            [video_id]
            + features
            + [
                prepare_feature(x)
                for x in [
                    trending_date,
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
            ]
        )
        lines.append(",".join(line))
    return lines


def get_pages(country_code, next_page_token="&"):
    country_data = []
    start_time = true_start
    end_time = start_time + delta
    # Because the API uses page tokens (which are literally just the same function of numbers everywhere) it is much
    # more inconvenient to iterate over pages, but that is what is done here.
    while end_time < true_end:
        while next_page_token is not None:
            # A page of data i.e. a list of videos and all needed data
            video_data_page = api_request(
                next_page_token, country_code, start_time, end_time
            )
            if video_data_page == switching_key:
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
            country_data += get_videos(items)
            if len(items) < 50:
                break
            print(len(country_data))
        next_page_token = "&"
        start_time, end_time = advance_time(start_time, end_time)
    return country_data


def write_to_file(country_code, country_data):

    print(f"Writing {country_code} data to file...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(
        f"{output_dir}/{time.strftime('%M.%S')}_videos.csv", "w+", encoding="utf-8",
    ) as file:
        for row in country_data:
            file.write(f"{row}\n")


def get_data():
    for country_code in country_codes:
        country_data = [",".join(header)] + get_pages(country_code)
        write_to_file(country_code, country_data)


allkeys = setup(key_path)
api_key = allkeys.pop(0)

true_start, true_end = get_time_range()
delta = datetime.timedelta(hours=2)

get_data()

