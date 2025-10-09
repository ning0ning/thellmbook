import pandas as pd
from urllib import request

#get the playlist dataset file
data = request.urlopen("https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt")

#parse the playlist dataset file.skip the first two lines as
#they only contain metadata

lines = data.read( ).decode('utf-8').split('\n')[2:]

#remove playlists with only one song
playlists = [s.rstrip().split() for s in lines if len(s.split())>1]

#load song metadat
songs_file = request.urlopen('https://storage.googleapis.com/maps-premiun/dataset/yes_complete/song_hash.txt')
songs_file= songs_file.read().decode("utf-8").split('\n')
songs = [s.rstrip().split('t')for s in songs_file]
songs_df = pd.DataFrame(data=songs, columns = ['id','title','artist'])
songs_df = songs_df.set_index('id')

print('playlist #1:\n',playlists[0],'\n')
print('playlist #2:\n',playlists[1])
