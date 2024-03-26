import chess
import numpy as np
import re
import pandas as pd

# GENERATE EMPTY BOARD
eb  = chess.Board()
eb.clear()

starting_squares = chess.SQUARE_NAMES


# GENERATE A LIST OF QUEEN MOVES
# FOR EACH SQUARE ON THE BOARD AS A STARTING POSITION
queen_moves = []
for start_pos in starting_squares:
    # DUPLICATE EMPTY BOARD AND PLACE A QUEEN AT THIS STARTING POSITION
    new_board = eb.copy()
    new_board.set_piece_at( getattr(chess,start_pos.upper()) , chess.Piece(getattr(chess,'QUEEN'),chess.WHITE) )

    # GET THE LEGAL MOVES FOR THIS PIECE FROM THE CURRENT SQUARE
    legal_moves = new_board.legal_moves
    for move in legal_moves:
        queen_moves.append(move.uci())

# KNIGHT MOVES
knight_moves = []
for start_pos in starting_squares:
    # DUPLICATE EMPTY BOARD AND PLACE A KNIGHT
    new_board = eb.copy()
    new_board.set_piece_at( getattr(chess,start_pos.upper()) , chess.Piece(getattr(chess,'KNIGHT'),chess.WHITE) )

    # GET THE LEGAL MOVES FOR THIS PIECE FROM THE CURRENT SQUARE
    legal_moves = new_board.legal_moves
    for move in legal_moves:
        knight_moves.append(move.uci())

# CASTLING MOVES
castle_moves = ['O-O-O','O-O']

# PAWN PROMOTION MOVES
promotion_moves = []
files = [file.upper() for file in chess.FILE_NAMES]
starting_squares = [file+'7' for file in files]
for start_pos in starting_squares:
    # DUPLICATE EMPTY BOARD AND PLACE THE INITIAL PAWN
    new_board = eb.copy()
    new_board.set_piece_at( getattr(chess,start_pos.upper()) , chess.Piece(getattr(chess,'PAWN'),chess.WHITE) )

    # GET VALID FILES TO LEFT AND RIGHT OF PAWN
    adjacent_files  = [chr(ord(start_pos[0])-1) , chr(ord(start_pos[0])+1)]
    valid_files = [file for file in adjacent_files if file[0] in files] # MAKE SURE WE'RE NOT OFF THE BOARD
    
    # GET VALID POSITIONS FOR OPPONENT PIECES, TO OFFER PROMOTION BY ADVANCEMENT OR CAPTURE
    valid_opponent_squares = [file+'8' for file in valid_files]

    # PLACE OPPONENT ROOKS ON VALID OPPONENT SQUARES, TO MAXIMIZE NUMBER OF WAYS THE CURRENT PAWN CAN PROMOTE
    for square in valid_opponent_squares:
        new_board.set_piece_at( getattr(chess,square.upper()), chess.Piece(getattr(chess,'ROOK'),chess.BLACK) )
    
    # GET LEGAL MOVES
    legal_moves = new_board.legal_moves
    for move in legal_moves:
        promotion_moves.append(move.uci())

n_total_moves = len(queen_moves) + len(knight_moves) + len(promotion_moves) + len(castle_moves)
print('**Total number of allowed moves: '+str(n_total_moves)+'**')

# DO SOME WRITE-OUT TO CSV
castle_indicator = ['O' for move in castle_moves]
queen_indicator = ['Q' for move in queen_moves]
knight_indicator = ['N' for move in knight_moves]
promotion_indicator = ['P' for move in promotion_moves]


all_valid_moves = castle_moves + queen_moves + knight_moves + promotion_moves
all_indicators = castle_indicator + queen_indicator + knight_indicator + promotion_indicator
n_moves_check = len(all_valid_moves)

key_frame = pd.DataFrame()
key_frame['MOVES'] = all_valid_moves
key_frame['PIECE'] = all_indicators
key_frame.to_csv('Output_Node_Key.csv',index_label='INDEX')
print('Successfully wrote key data to CSV!')
