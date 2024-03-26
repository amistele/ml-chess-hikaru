import numpy as np
import pandas as pd
import chessfun

# LOAD THE RAW TRAINING DATA I ALREADY SCRAPED FROM CHESS.com
hikaru_games = pd.read_csv('TRAINING_GAMES_hikaru_29397.csv')

# CLEAN THE PGN in THE DATA SET
pgns = []
colors = []
for ind in hikaru_games['INDEX']:
    print('Cleaning '+str(ind+1)+'/'+str(len(hikaru_games['INDEX'])))

    color = hikaru_games['MY_COLOR'][ind]
    pgn = chessfun.rewrite_api_pgn( hikaru_games['PGN_RAW'][ind] )[1]

    if (len(pgn) > 0) and (pgn[0] == '1'): # sometimes for balanced this isn't the case! don't want these
        pgns.append(pgn)
        colors.append(color)

# WRITE NEW DATA INCLUDING CLEANED DATA OUT TO CSV
hikaru_cleaned = pd.DataFrame()
hikaru_cleaned['MY_COLOR'] = colors
hikaru_cleaned['PGN_CLEAN'] = pgns
hikaru_cleaned.to_csv('TRAINING_GAMES_hikaru_'+str(len(pgns))+'_CLEANED.csv',index_label='INDEX')