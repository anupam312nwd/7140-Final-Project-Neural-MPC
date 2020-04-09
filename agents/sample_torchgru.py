import os
import torch
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser('GRU demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

class GRU(nn.Module):

    def __init__(self, input_size= , hidden_size= , out_size= ):

        super(GRU).__init__()

        # Add an GRU layer
        self.gru = nn.GRU(input_size, hidden_size)

        # Add a fully-connected layer
        self.linear = nn.Linear(hidden_size,out_size)

        # Initialize h0
        self.hidden = torch.zeros(1, 1, hidden_size)

    def forward(self, seq):
        gru_out, self.hidden = self.gru(seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(gru_out.view(len(seq),-1))
        return pred[-1]   # we only care about the last prediction


if __name__ == '__main__':

    ii = 0

    model = LSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    end = time.time()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = model.forward()
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = model.forward()
                loss = criterion(pred_y, true_y)
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()

