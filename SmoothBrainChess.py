import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import chessfun
import chess


class SBCEngine:
    def __init__(self, n_nodes, weights_file, node_key_file, engine_color):

        # INITIALIZE THE ENGINE, SET UP TO RUN (NOT TRAIN)
        self.model = SmoothBrainChess(n_nodes, True)
        self.model.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu')))
        self.model.eval()
        print('Engine initialized using n='+str(n_nodes)+' and weights @ '+weights_file)

        # KEY FILE FOR TRANSLATING OUTPUTS
        self.key_frame = pd.read_csv(node_key_file)

        # STORE ENGINE COLOR
        self.engine_color = engine_color.upper() # WHITE or BLACK
        print('Engine is playing as '+engine_color+', initialization complete!')

    def run(self, board):

        # EVALUATE THE MODEL GIVEN THE CURRENT BOARD STATE
        matrix = chessfun.board_to_matrix(board)
        my_board = chessfun.matrix_color_adjust(matrix,self.engine_color)
        bitboards = chessfun.matrix_to_bitboards(my_board)
        embedding = torch.from_numpy(chessfun.bitboards_to_input(bitboards).astype(np.float32))

        model_output = self.model(embedding)
        return model_output

    def get_move(self, board):
        moves,values,output = self.topk(board,1,True)
        return moves[0]

    def topk(self, board, n, legal_flag):

        # EVALUATE THE BOARD
        model_output = self.run(board)
        model_output = model_output[0].detach()

        # ZERO-OUT ILLEGAL MOVES IF 'legal_flag=True'
        if legal_flag:
            for x in range(len(model_output)):
                # GET AND UPDATE THE MOVE STRING
                node_str = self.key_frame['MOVES'][x]

                corr_node_str = chessfun.update_notation_orientation(node_str, self.engine_color)

                # CONVERT CASTLING MOVES TO UCI NOTATION
                if corr_node_str == 'O-O':
                    if self.engine_color == 'WHITE':
                        corr_node_str = 'e1g1'
                    if self.engine_color == 'BLACK':
                        corr_node_str = 'e8g8'
                elif corr_node_str == 'O-O-O':
                    if self.engine_color == 'WHITE':
                        corr_node_str = 'e1c1'
                    if self.engine_color == 'BLACK':
                        corr_node_str = 'e8c8'

                # IF MOVE IS ILLEGAL, THE REWRITE NODE OUTPUT
                legal_moves = [mv.uci() for mv in board.legal_moves]
                n_legal_moves = len(legal_moves)
                if corr_node_str not in legal_moves:
                    #print(corr_node_str, moves_uci)
                    model_output[x] = -999.0

                # IF THERE ARE FEWER LEGAL MOVES THAN MOVES REQUESTED, REDUCE NUMBER OF MOVES RETURNED
                if n_legal_moves < n:
                    n = n_legal_moves

        # GET TOP K MOST LIKELY
        values, indices = torch.topk(model_output, n)
        #print('values',values[0].detach().numpy())
        #print('indices',indices[0].numpy())

        # CONVERT TO USEABLE TYPES FROM TENSOR
        vals = values.numpy()
        inds = indices.numpy()

        moves = [chessfun.update_notation_orientation(self.key_frame['MOVES'][x], self.engine_color) for x in inds]

        #print('values',vals)
        #print('indices',inds)
        #print('moves',moves)

        return moves, vals, model_output.detach().numpy()


class SmoothBrainChess(nn.Module):
    def __init__(self, n_per_hidden, biases):
        super().__init__()
        self.biases = biases
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, n_per_hidden, bias=self.biases),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_per_hidden, n_per_hidden, bias=self.biases),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_per_hidden, 1882, bias=self.biases),
        )

    # FORWARD PROPAGATE TO GET OUTPUTS
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class TestChessDataSet_SingleFile(Dataset):
    def __init__(self, device, train, infile, keyfile):
        # LOAD INPUTS
        self.device = device
        self.train = train
        df_input = pd.read_csv(infile)
        if self.train:
            df_input = df_input[0:int(np.floor(len(df_input)/2))]
        else:
            df_input = df_input[int(np.ceil(len(df_input)/2)):]

        # LOAD OUTPUTS
        df_output = pd.read_csv(keyfile)
        if self.train:
            df_output = df_output[0:int(np.floor(len(df_output)/2))]
        else:
            df_output = df_output[int(np.ceil(len(df_output)/2)):]

        # EACH ELEMENT IN self.embeddings IS A NUMPY ARRAY OF THE INPUT LAYER EMBEDDING
        self.embeddings = torch.from_numpy(df_input.to_numpy()[:,2:].astype(np.float32))

        # EACH ELEMENT IN self.labels IS THE INDEX OF THE CORRECT OUTPUT NODE
        self.label_indices = df_output['OUT_INDEX'].to_numpy()
        #transform = Lambda(lambda y: torch.zeros(1882, dtype=torch.float).scatter(0, torch.tensor(y), value=1))

    def __len__(self):
        return len(self.label_indices)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        if self.train:
            label = torch.zeros(1882, dtype=torch.float).scatter(0, torch.tensor(self.label_indices[idx]), value=1).to(self.device)
        else:
            label = torch.tensor(self.label_indices[idx])
        return embedding.to(self.device), label.to(self.device)


class TestChessDataSet_MultiFile(Dataset):
    def __init__(self, device, train, infile_pattern, keyfile_pattern,ndiv):
        # LOAD INPUTS
        self.device = device
        self.train = train

        print([infile_pattern+str(x+1)+'.csv' for x in range(ndiv)])
        dfs_input = [pd.read_csv(infile_pattern+str(x+1)+'.csv') for x in range(ndiv)]
        dfs_output = [pd.read_csv(keyfile_pattern+str(x+1)+'.csv') for x in range(ndiv)]

        print('INPUT LENS: ', [len(x) for x in dfs_input])
        print('OUTPUT LENS: ', [len(x) for x in dfs_output])

        df_input = pd.concat(dfs_input)
        df_output = pd.concat(dfs_output)

        print('FINAL INPUT LEN: ',len(df_input))
        print('FINAL OUTPUT LEN: ',len(df_output))

        # HANDLE INPUTS
        if self.train:
            df_input = df_input[0:int(np.floor(len(df_input)/2))]
        else:
            df_input = df_input[int(np.ceil(len(df_input)/2)):]

        # HANDLE OUPUTS OUTPUTS
        if self.train:
            df_output = df_output[0:int(np.floor(len(df_output)/2))]
        else:
            df_output = df_output[int(np.ceil(len(df_output)/2)):]

        # EACH ELEMENT IN self.embeddings IS A NUMPY ARRAY OF THE INPUT LAYER EMBEDDING
        self.embeddings = torch.from_numpy(df_input.to_numpy()[:,2:].astype(np.float32))

        # EACH ELEMENT IN self.labels IS THE INDEX OF THE CORRECT OUTPUT NODE
        self.label_indices = df_output['OUT_INDEX'].to_numpy()
        #transform = Lambda(lambda y: torch.zeros(1882, dtype=torch.float).scatter(0, torch.tensor(y), value=1))

    def __len__(self):
        return len(self.label_indices)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        if self.train:
            label = torch.zeros(1882, dtype=torch.float).scatter(0, torch.tensor(self.label_indices[idx]), value=1).to(self.device)
        else:
            label = torch.tensor(self.label_indices[idx])
        return embedding.to(self.device), label.to(self.device)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")