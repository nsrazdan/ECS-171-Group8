import requests
import pandas as pd

with open("allkeys.txt") as file:
    allkeys = [x.rstrip() for x in file]
    for key in allkeys:
        response = requests.get(
            f"https://www.googleapis.com/youtube/v3/videos?part=id&chart=mostPopular&regionCode=US&key={key}"
        )
        print(response.status_code)
        if response.status_code != 200:
            print(response.json())

