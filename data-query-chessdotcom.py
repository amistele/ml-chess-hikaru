# IMPORTS
import requests
import json
import chess
import chess.pgn
import numpy as np
import pandas as pd
import chessfun # my function library

# SET UP API CALL HEADERS
headers = {
    'User-Agent':'andrew@mistele.com'
}


# ASSEMBLE URLS to QUERY
api_url     = 'https://api.chess.com'
api_local   = '/pub/player/'
api_games   = '/games/archives'
players     =['hikaru']

# GET A LIST OF GAME ARCHIVES FOR THE PLAYER
urls_to_hit = [api_url+api_local+player+api_games for player in players]
response = requests.get( urls_to_hit[0], headers=headers)

# GET THE LATEST MONTH'S GAME ARCHIVES
archive_urls = response.json()['archives']
print(str(len(archive_urls))+' archive urls found')

# GET THE NUMBER OF GAMES
archives = []
played_games = []
won_games = []
won_games_pgns  = []
won_games_colors = []
total_games_played = 0
total_games_won = 0
archive_url_number = 1
for archive_url in archive_urls:
    print('Checking archive #',archive_url_number,'/',len(archive_urls))

    archive = requests.get( archive_url , headers=headers)
    archives.append( archive )

    games = archive.json()['games']

    games_in_archive = 0
    games_won_in_archive = 0
    for game in games:
        # SKIP CHESS VARIANTS
        if game['rules'] != 'chess':
            continue

        # SKIP BULLET GAMES
        if game['time_class'] == 'bullet':
            continue

        # SAVE THE GAME
        played_games.append(game)
        games_in_archive += 1
        total_games_played += 1

        # DETERMINE IF PLAYER WON
        current_player = api_url+api_local+players[0]

        # DETERMINE WHICH COLOR OUR CURRENT PLAYER WAS IN THIS GAME
        if (game['white']['@id'] == current_player.lower()):
            player_color_string = 'white'
        elif (game['black']['@id'] == current_player.lower()):
            player_color_string = 'black'
        else:
            print('Uhhhhhhh cannot find the player in their own game??')
            quit()

        # MAKE SURE THIS PLAYER WON
        if (game[ player_color_string ]['result'] == 'win'):
            won_games.append( game )
            won_games_pgns.append( game['pgn'])
            won_games_colors.append( player_color_string.upper())
            games_won_in_archive += 1
    total_games_won += games_won_in_archive
    print('#----------#')
    print('Games found in archive:  ',games_in_archive)
    print('Games won in archive:    ',games_won_in_archive)
    print('Total games played:      ',total_games_played)
    print('Total games won:         ',total_games_won)
    print('#----------#')
    print(' ' )
    archive_url_number += 1
            
wins_data = pd.DataFrame()
wins_data['MY_COLOR'] = won_games_colors
wins_data['PGN_RAW'] = won_games_pgns
wins_data.to_csv('TRAINING_GAMES_'+players[0]+'_'+str(total_games_won)+'.csv',index_label='INDEX')


"""
# OK, IF WE'VE GOTTEN HERE, THEN A WINNING GAME WAS FOUND!
pgn_string = games[idx_win]['pgn']
fen_string = games[idx_win]['fen']

# GET USEFUL PGN STRING
(pgn_fix_full_string , useful_pgn_string) = chessfun.rewrite_api_pgn( pgn_string )


# CONVERT PGN TO DICT OF MOVES BY TURN
#moves_dict = chessfun.pgn_to_dict( useful_pgn_string)
moves_list = chessfun.pgn_to_list( useful_pgn_string)


# INITIALIZE CHESS BOARD
cb = chess.Board() # initial setup, white up front




# GET WHITE PAWN BITBOARD
"""