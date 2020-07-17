from torch.utils.data import DataLoader
import dgl
import torch
import numpy as np
import sys
import os
import socnavData
import pickle
import torch.nn.functional as F

def activation_functions(activation_tuple_src):
    ret = []
    for x in activation_tuple_src:
        if x == 'relu':
            ret.append(F.relu)
        elif x == 'elu':
            ret.append(F.elu)
        elif x == 'tanh':
            ret.append(torch.tanh)
        elif x == 'leaky_relu':
            ret.append(F.leaky_relu)
        else:
            print('Unknown activation function {}.'.format(x))
            sys.exit(-1)
    return tuple(ret)

sys.path.append(os.path.join(os.path.dirname(__file__), 'nets'))

from rgcnDGL import RGCN


def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels


class SocNavAPI(object):
    def __init__(self, base, dataset, device='cpu'):
        self.device = torch.device(device)  # For gpu change it to cuda
        self.device2 = torch.device('cpu')
        print(base)
        self.params = pickle.load(open(os.path.dirname(__file__)+'/SOCNAV.prms', 'rb'), fix_imports=True)
        self.params['net'] = self.params['net'].lower()
        print(self.params)
        print(self.params['net'])
        if self.params['net'] in ['rgcn']:
            self.GNNmodel = RGCN(g=None,
                                 num_gnn_layers=self.params['gnn_layers'],
                                 in_dim=self.params['num_feats'],
                                 hidden_dimensions=self.params['num_hidden'],  # feats hidden
                                 num_rels=self.params['num_rels'],
                                 activation=torch.tanh,
                                 feat_drop=self.params['in_drop'],
                                 num_bases=self.params['num_rels'])
        else:
            print('Unknown/unsupported model in the parameters file')
            sys.exit(0)


        self.GNNmodel.load_state_dict(torch.load(os.path.dirname(__file__)+'/SOCNAV.tch', map_location = device))
        self.GNNmodel.to(self.device)
        self.GNNmodel.eval()

        if dataset is not None:
            self.test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate)
        

    def predictOneGraph(self, g):
        self.test_dataloader = DataLoader(g, batch_size=1, collate_fn=collate)
        logits = self.predict()
        return logits
        
    def predict(self):
        result = []
        for batch, data in enumerate(self.test_dataloader):
            subgraph, feats, labels = data
            feats = feats.to(self.device)
            self.GNNmodel.set_g(subgraph)
            if self.params['net'] in ['rgcn']:
                logits = self.GNNmodel(feats.float(), subgraph.edata['rel_type'].squeeze().to(self.device))
            else:
                logits = self.GNNmodel(feats.float())
            result.append(logits[0])
        return result
        
