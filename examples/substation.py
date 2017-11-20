#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:40:22 2017

@author: santi
"""

import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
        
        


if __name__ == "__main__":
    
    # load data    
    conn_df = pd.read_excel('substation.xlsx', 'Connectivity', index_col=0).fillna(0)
    stat_df = pd.read_excel('substation.xlsx', 'States', index_col=0)
    pos_df = pd.read_excel('substation.xlsx', 'Pos', index_col=0)
    
    node_names = conn_df.columns.values

    
    G = nx.Graph()
    pos = dict()
    lpos = dict()
    
    # add nodes to the graph
    for i in range(len(node_names)):
        G.add_node(node_names[i])
        x = pos_df.values[i, 0]
        y = pos_df.values[i, 1] 
        pos[node_names[i]] = [x, y]
        lpos[node_names[i]] = [x, y]
    
    # add branches to the graph    
    for i, line in enumerate(conn_df.values):
        if stat_df.values[i] > 0:
            x, y = np.where(line > 0)[0]  # works because there are only 2 values per line with a 1 in the excel file
            n1 = node_names[x]
            n2 = node_names[y]
            G.add_edge(n1, n2)
                
    # get the islands
    islands = list(nx.connected_components(G))
    sub_grids = list()
    print('Islands:\n', islands, '\n\n')
    for island in islands:
        g = nx.subgraph(G, island)
        sub_grids.append(g)
    
    # plot
    nx.draw(G, pos=pos, node_size=100, node_color='black')
    for name in node_names:
        x, y = lpos[name]
        plt.text(x+1.5,y+1,s=name, bbox=dict(facecolor='white', alpha=0.5), horizontalalignment='center')
    plt.show()