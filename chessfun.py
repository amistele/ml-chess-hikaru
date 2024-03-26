import re
import numpy as np
import io
import chess
import chess.pgn

# fix chess.com API distorted output of pgn string
def rewrite_api_pgn( pgn_string ):

    # SPLIT HEADER AND BODY
    pgn_parts = re.split( '\n\n' , pgn_string)
    head = pgn_parts[0]
    pgn = pgn_parts[1]

    # REMOVE THE TIME STAMPS AND CRAP WITH THE CURLY BRACES FROM THE PGN STRING
    pgn_nice = re.sub(r'\{[^{}]*\}', '', pgn)

    # REMOVE THE ADDITIONAL MOVE NUMBER INDICATORS
    pgn_final = re.sub(r' \d{1,3}\.\.\. ','',pgn_nice)

    # REMOVE RECORD (e.g. 1-0) AND NEWLINE AT END OF PGN
    pgn_stripped = re.sub(r'\d{1,4}-\d{1,4}','',pgn_final).strip()

    recomposed = head + '\n\n' + pgn_final
    return recomposed , pgn_stripped

# CONVERT STRIPPED PGN STRING FROM 'rewrite_api_gen' TO A DICT OF MOVES
"""
def pgn_to_dict( pgn_stripped):
    # GET ELEMENT OF MOVE NUMBER, AND CORRESPONDING MOVES
    elements = [element for element in pgn_stripped.split(' ') if element not in ' ']

    moves = {}
    idx = 0
    while idx < len( elements ):
        id = int( elements[idx].strip('.') )
        moves[id] = {}
        
        # ALLOW DICT ACCES BY INT OR STRING
        moves[id].update( dict.fromkeys(['white',0],elements[idx+1]))
        moves[id].update( dict.fromkeys(['black',1],elements[idx+2]))

        idx += 3

    return moves
"""

# COVERT STRIPPED PGN STRING FROM 'rewrite_api_gen' TO A LIST OF MOVES
def pgn_to_list( pgn_stripped):
    # GE ELEMENT OF MOVE NUMBERAND CORRESPONDING MOVES
    elements = [element for element in pgn_stripped.split(' ') if element not in ' ']
    moves = [move for move in elements if '.' not in move]
    return moves


# CONVERT CHESS BOARD (chess.Board() object) TO NUMERICAL MATRIX
# MATRIX IS ALWAYS ORIENTED SO THAT "MY" PIECES ARE AT BOTTOM
def board_to_matrix( chess_board ):
    #   1 = pawn
    #   2 = knight
    #   3 = bishop
    #   4 = rook
    #   5 = queen
    #   6 = king
    
    # GET BOARD STRING AND USE REGEXES TO REPLACE CHARACTERS WITH NUMBERS
    #   WHITE > 0, BLACK < 0
    board_string = chess_board.__str__()
    #   WHITE PIECES
    board_nums = re.sub(r'P','1',board_string)
    board_nums = re.sub(r'N','2',board_nums)
    board_nums = re.sub(r'B','3',board_nums)
    board_nums = re.sub(r'R','4',board_nums)
    board_nums = re.sub(r'Q','5',board_nums)
    board_nums = re.sub(r'K','6',board_nums)
    #   BLACK PIECES
    board_nums = re.sub(r'p','-1',board_nums)
    board_nums = re.sub(r'n','-2',board_nums)
    board_nums = re.sub(r'b','-3',board_nums)
    board_nums = re.sub(r'r','-4',board_nums)
    board_nums = re.sub(r'q','-5',board_nums)
    board_nums = re.sub(r'k','-6',board_nums)
    #   EMPTY SPACES
    board_nums = re.sub(r'\.','0',board_nums)
    #print(board_nums)

    # SPLIT BOARD INTO ROWS
    board_ints = [int(x) for x in re.split(r'\n| ',board_nums)]
    #print(board_ints)

    # PARSE THE STRING INTO A MATRIX
    # MATRIX IS ORIENTED SAME AS CHESSBOARD (7,0) = A1
    matrix = np.array( board_ints ).reshape([8,8])
    #print(matrix)
    return matrix

# TAKE OUTPUT of 'board_to_matrix' AND ENSURE "MY" PIECES ARE AT THE BOTTOM "TOWARDS ME" AND THAT THEIR INTEGERS ARE POSITIVE
def matrix_color_adjust( matrix , my_color ):
    # int > 0 --> my piece
    # int < 0 --> opponent piece
    if (my_color.upper() == 'BLACK') or (my_color == 1):
        matrix = -1 * np.rot90(matrix,2) # rotate twice = flip board; negative sign flips to my pieces
    
    elif( my_color.upper() != 'WHITE') and (my_color != 0):
        print('ERROR! Unrecognized color.')
        quit()
    
    return  matrix

# TAKE MATRIX REPRESENTATION OF CHESS BOARD AND CONVERT TO BITBOARDS (STORED TO DICT)
def matrix_to_bitboards( matrix ):
    #   1 = pawn
    #   2 = knight
    #   3 = bishop
    #   4 = rook
    #   5 = queen
    #   6 = king
    # # > 0 --> mine , # < 0 --> opponent's
    bitboards = np.empty([12,8,8])
    for index,piece in enumerate([-6,-5,-4,-3,-2,-1,1,2,3,4,5,6]):
        bitboards[index,:,:] = matrix == piece

    return bitboards

# CONVERT 3D BITBOARD MATRIX TO COLUMN VECTOR FOR INPUT INTO FIRST LAYER!
def bitboards_to_input( bitboards ):
    input = np.reshape( bitboards , [1, np.size(bitboards) ])
    return input


# CONVERT GAME OBJECT TO LISTS OF BOARD FEN AND "CORRECT" MOVES
def pgn_to_move_solutions( pgn , color):

    # CONVERT pgn TO GAME OBJECT, CREATE INITIAL BOARD
    pgnio = io.StringIO( pgn )
    chessgame = chess.pgn.read_game(pgnio)
    board = chessgame.board()
    
    # PARSE OUT MOVES FROM THE GAME OBJECT
    moves = []
    for move in chessgame.mainline_moves():
        moves.append(move)

    if len(moves) == 0: # SOMETIMES HAPPENS WHEN TOURNAMENT / BALANCE RULES RESULT IN NON-STANDARD OPENING, WHICH I DON'T HAVE PGN / FEN FOR
        print('Non-standard opening! '+pgn[1:14])
        return ([],[],[])
    
    # ITERATE THROUGH THE MOVES IN THE GAME AND STORE THE ANSWERS BASED ON CURRENT COLOR
    board_fen = [] # fen representation of board state, convert to embedding later
    correct_moves = [] # standard notation representation of "correct" move
    correct_moves_fmt = [] # notation with modified strings for castling, which match my output layer key
    move_count = 0

    if color == 'WHITE': 
        incr = 0 # first half-turn is first solution (zero-idx'd)
    elif color == 'BLACK':
        incr = 1 # second half-turn is first solution
        board.push(moves[0]) # white makes the initial move
    else:
        print('Unrecognized color!')
        quit()
    
    while incr <= len(moves)-1:
        # STORE BOARD DATA AS FEN
        board_fen.append( board.fen() )

        # STORE THE UPCOMING (CURRENT) MOVE AS SOLUTION
        correct_moves.append( moves[incr].__str__())

        # STORE MOVE AS SOLUTION, WITH MODIFIED CASTLE STRING
        if board.is_kingside_castling( moves[incr] ):
            new_move = 'O-O'
        elif board.is_queenside_castling( moves[incr] ):
            new_move = 'O-O-O'
        else:
            new_move = moves[incr].__str__()
        correct_moves_fmt.append( new_move )

        # MAKE THE MOVE THAT WAS JUST SAVED
        board.push( moves[incr] )
        #   now it's the other player's turn

        # MAKE THE OTHER PLAYER'S MOVE - I DON'T CARE WHAT IT IS
        #   OBVIOUSLY, ONLY IF THERE ARE MORE MOVES LEFT
        if incr+1 <= len(moves)-1:
            board.push( moves[incr+1] )

            move_count += 1 # number of moves saved
            incr += 2 # 2 moves per turn, on to my next turn
        else:
            break

    return (board_fen,correct_moves_fmt,correct_moves)

def fen_to_embedding( fen , color):
    # GENERATE BOARD FROM fen
    board = chess.Board( fen )

    # CONVERT BOARD TO NUMERIC MATRIX
    matrix_raw = board_to_matrix( board )

    # ADJUST THE MATRIX TO MATCH THE CURRENT COLOR
    #   ACCOUNTS FOR BOARD ROTATION SO MY PIECES ARE ON THE BOTTOM, AND FLIPPING SIGNS ON PIECES
    #   SO MY PIECES ARE POSITION
    matrix_to_me = matrix_color_adjust( matrix_raw , color )

    # CONVERT THE MATRIX TO BITBOARDS REPRESENTING MY POSITION
    bitboards = matrix_to_bitboards( matrix_to_me )

    # CONVERT TO VECTOR EMBEDDING
    embedding_bool =  bitboards_to_input( bitboards )
    embedding = embedding_bool.astype(int)[0]
    return embedding

def update_notation_orientation( move_str , color ):
    # OUTPUT LAYER REPRESENTS POSITIONAL MOVES ON THE CHESSBOARD
    # INPUT EMBEDDING IS ALWAYS "BOARD TOWARDS ME" ('a1' = my bottom left)
    #   WHICH MEANS OUTPUT WILL ALWAYS BE "BOARD TOWARDS ME" ('a1' = my bottom left)
    # PROBLEM:
    #   NEED TO CONVERT THIS TO ACTUAL BOARD NOTATION ('a1' = white's bottom left) FOR READING OUTPUT LAYER FOR BLACK
    #   ALSO, NEED TO CONVERT BOARD NOTATION TO THIS "RELATIVE" NOTATION WHEN TRAINING FOR BLACK MOVES
    #       FORTUNATELY, THE SAME OPERATION WORKS BOTH DIRECTION

    # DON'T MODIFY CASTLING STRINGS
    if (move_str == 'O-O') or (move_str == 'O-O-O'):
        return move_str

    # IF WHITE, RELATIVE NOTATION IS CHESS NOTATION
    if color == 'WHITE':
        return move_str
    
    # ELSE, CONVERT RELATIVE NOTATION TO CHESS NOTATION OR VICE VERSA -- THIS WORKS BOTH WAYS
    elif color == 'BLACK':
        # GET LIST OF SQUARES
        squares = chess.SQUARE_NAMES
        white_board = np.flipud(np.reshape( np.array(squares) , (8,8))) # NOW, a1 IS IN BOTTOM LEFT: RELATIVE = CHESS

        # TURN THE BOARD TWICE TO GET THE BOARD FROM BLACK'S POSITION
        black_board = np.rot90( white_board , 2)

        # GET THE "FROM" AND "TOWARD" SQUARES
        frm = move_str[0:2]
        twd = move_str[2:4]
        the_rest = move_str[4:]

        new_frm = black_board[ white_board == frm ][0] # output node gives move as if white, need corresponding actual black move
        new_twd = black_board[ white_board == twd ][0]

        # ASSEMBLE NEW MOVE
        new_move = new_frm+new_twd+the_rest
        return new_move


    