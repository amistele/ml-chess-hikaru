import pandas as pd
import chessfun

# READ PGN AND COLORS
hikaru_games = pd.read_csv('TRAINING_GAMES_hikaru_29389_CLEANED.csv')

# GENERATE FEN INPUTS AND MOVE OUTPUTS
in_fen = []
out_fmt = []
out_std = []
colors = []

for ind in range( len(hikaru_games['INDEX']) ):
    #print('#-----#')
    #print('Game '+str(ind+1)+'/'+str(len(hikaru_games['INDEX'])))
    (fen,fmt,std) = chessfun.pgn_to_move_solutions( hikaru_games['PGN_CLEAN'][ind] , hikaru_games['MY_COLOR'][ind])
    in_fen += fen
    out_fmt += fmt
    out_std += std
    if len( fen ) > 0: # invalid standard openings return zero length lists
        colors += [hikaru_games['MY_COLOR'][ind] for x in fen]
    #print('New solutions:   '+str(len(fen)))
    #print('Total solutions: '+str(len(in_fen)))
    #print('#-----#')
    #print(' ')

    print('Game '+str(ind+1)+'/'+str(len(hikaru_games['INDEX']))+' --&&-- New Solutions: '+str(len(fen))+' --&&-- Total Solutions: '+str(len(in_fen)))

# STORE TO DATAFRAME AGAIN, YAY
n_solutions = len(in_fen)
hikaru_solutions = pd.DataFrame()
hikaru_solutions['MY_COLOR'] = colors
hikaru_solutions['IN_FEN'] = in_fen
hikaru_solutions['OUT_FORMAT'] = out_fmt
hikaru_solutions['OUT_STANDARD'] = out_std

hikaru_solutions.to_csv('TRAINING_SOLUTIONS_hikaru_'+str(n_solutions)+'.csv',index=False)