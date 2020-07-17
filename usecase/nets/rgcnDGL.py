import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from functools import partial

from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv


class RGCN(nn.Module):
    def __init__(self, g, num_gnn_layers, in_dim, hidden_dimensions, num_rels, activation, feat_drop, num_bases=-1):
        super(RGCN, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.hidden_dimensions = hidden_dimensions
        self.num_rels = num_rels
        self.feat_drop = feat_drop
        self.num_bases = num_bases
        self.activation = activation
        self.num_gnn_layers = num_gnn_layers
        self.layers = nn.ModuleList()
        # create RGCN layers
        self.build_model()
        
    def set_g(self, g):
        self.g = g

    def build_model(self):
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for i in range(self.num_gnn_layers-2):
            h2h = self.build_hidden_layer(i)
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    def build_input_layer(self):
        print('Building an INPUT  layer of {}x{}'.format(self.in_dim, self.hidden_dimensions[0]))
        return RelGraphConv(self.in_dim, self.hidden_dimensions[0], self.num_rels, regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=F.leaky_relu)

    def build_hidden_layer(self, i):
        print('Building an HIDDEN  layer of {}x{}'.format(self.hidden_dimensions[i], self.hidden_dimensions[i+1]))
        return RelGraphConv(self.hidden_dimensions[i], self.hidden_dimensions[i+1],  self.num_rels, regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=F.leaky_relu)

    def build_output_layer(self):
        print('Building an OUTPUT  layer of {}x{}'.format(self.hidden_dimensions[-2], self.hidden_dimensions[-1]))
        return RelGraphConv(self.hidden_dimensions[-2], self.hidden_dimensions[-1], self.num_rels, regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=torch.tanh)

    def forward(self, features, etypes):
        h = features
        self.g.edata['norm'] = self.g.edata['norm'].to(device=features.device)

        #print("H",h.size())
        # self.g.ndataz['h'] = features
        for layer in self.layers:
            h = layer(self.g, h, etypes)
            #print("Size:", h.size())
        logits = h

        base_index = 0
        batch_number = 0
        unbatched = dgl.unbatch(self.g)
        output = torch.Tensor(size=(len(unbatched), 3))
        for g in unbatched:
            num_nodes = g.number_of_nodes()
            output[batch_number, :] = logits[base_index, :] # Output is just the room's node
            # output[batch_number, :] = logits[base_index:base_index+num_nodes, :].mean(dim=0) # Output is the average of all nodes
            base_index += num_nodes
            batch_number += 1
        return output

