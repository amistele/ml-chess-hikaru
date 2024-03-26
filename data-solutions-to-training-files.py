import pandas as pd
import chessfun
import numpy as np

solutions = pd.read_csv('TRAINING_SOLUTIONS_hikaru_10k.csv')
output_key = pd.read_csv('Output_Node_Key.csv')

embeddings = [] # VECTORS OF FLATTENED BITBOARDS
colors = []
outputs_raw = [] # MOVES FROM SOLUTIONS SET (BOARD NOTATION), NO CASTLE STRINGS
outputs_fmt = [] # MOVES FROM SOLUTIONS SET (BBOARD NOTATION),  CASTLE STRINGS
outputs_rel = [] # MOVES IN NOTATION RELATIVE TO PLAYER (FLIPPED BOARD FOR BLACK), WITH CASTLE STRINGS
outputs_ind = [] # INDEX IN OUTPUT NODE THAT MOVE CORRESPONDS TO


for ind in range(len(solutions)):

    print('Parsing Solution '+str(ind+1)+'/'+str(len(solutions)))
    # STORE COLOR
    colors.append( solutions['MY_COLOR'][ind] )

    # GET INPUT EMBEDDING BASED ON BOARD LAYOUT AND COLOR
    embeddings.append( chessfun.fen_to_embedding( solutions['IN_FEN'][ind] , solutions['MY_COLOR'][ind] ))

    # STORE THE UN-TRANSLATED OUTPUTS
    outputs_raw.append( solutions['OUT_STANDARD'][ind] ) # NO CASTLE STRINGS
    outputs_fmt.append( solutions['OUT_FORMAT'][ind] ) # CASTLE STRINGS
    outputs_rel.append( chessfun.update_notation_orientation( solutions['OUT_FORMAT'][ind] , solutions['MY_COLOR'][ind] )) # RELATIVE TO COLOR
    
    # FIND INDICES WHERE KEY VECTOR MATCHES CURRENT MOVE
    indices = np.where( output_key['MOVES'] == outputs_rel[-1] )[0]
    n_ind = len(indices)
    if n_ind != 1:
        print('Error! More than one move found in key that matches, how is this possible?')
        quit()
    outputs_ind.append( indices[0] )


# CONSTRUCT INPUTS DATAFRAME
print('Generating input layers dataframe...')
col_names = ['IN_'+str(x) for x in range(len(embeddings[0]))]
input_layers = pd.DataFrame( embeddings , columns=col_names)
input_layers.insert( 0, 'COLOR',colors)

# SAVE INPUTS TO CSV
print('Saving input layer dataframe to csv...')
input_layers.to_csv('10k_INPUT_LAYERS.csv',index_label='INDEX')

# CONSTRUCT OUTPUTS DATAFRAME
#   ACTUAL OUTPUT LAYER VECTORS WILL BE CREATED AS TRAINING SINCE IT'S EASY, AND THEY'D TAKE TOO MUCH SPACE TO STORE WITH 1882 OUTPUT NODES
print('Generating output node index dataframe')
output_data = pd.DataFrame()
output_data['COLOR'] = colors
output_data['OUT_INDEX'] = outputs_ind # THE RELEVANT VALUE FOR TRAINING
output_data['OUTPUT_DATA_REL'] = outputs_rel # WHAT THE RELEVANT VALUE WAS DERIVED FROM
output_data['OUTPUT_DATA_FMT'] = outputs_fmt # WHAT THE FORMATTED, ABSOLUTE POSITION WAS (NOT RELATIVE TO COLOR)
output_data['OUTPUT_DATA_RAW'] = outputs_raw # ABSOLUTE POSITION WITHOUT CASTLE STRINGS

print('Saving output node index dataframe to to csv...')
output_data.to_csv('10k_OUTPUT_NODE_INDICES.csv',index_label='INDEX')
print('Done!')