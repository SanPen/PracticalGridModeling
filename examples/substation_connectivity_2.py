import numpy as np
from scipy.sparse import csc_matrix


class Terminal:

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class Switch:

    def __init__(self, name, t1, t2, state=True):
        self.name = name
        self.t1 = t1
        self.t2 = t2
        self.state = state

    def __str__(self):
        return self.name


class ConnectivityNode:

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class Branch:

    def __init__(self, name, t1, t2):
        self.name = name
        self.t1 = t1
        self.t2 = t2

    def __str__(self):
        return self.name


class Circuit:

    def __init__(self):

        self.nodes = list()
        self.terminals = list()
        self.switches = list()
        self.branches = list()

        self.nodes_idx = dict()  # dictiona
        self.terminals_idx = dict()

        # relations between connectivity nodes and terminals
        # node_terminal[some_node] = list of terminals
        self.node_terminal = dict()

    def add_node_terminal_relation(self, connectivity_node, terminal):
        """
        Add the relation between a Connectivity Node and a Terminal
        :param terminal:
        :param connectivity_node:
        :return:
        """
        if connectivity_node in self.node_terminal.keys():
            self.node_terminal[connectivity_node].append(terminal)
        else:
            self.node_terminal[connectivity_node] = [terminal]

    def add_connectivity_node(self, node):
        """
        add a Connectivity node
        :param node:
        :return:
        """
        self.nodes.append(node)

    def add_terminal(self, terminal):

        self.terminals.append(terminal)

    def add_switch(self, switch):
        """
        Add a switch
        :param switch:
        :return:
        """
        self.switches.append(switch)

    def add_branch(self, branch):
        """
        Add a branch
        :param branch:
        :return:
        """
        self.branches.append(branch)

    def compile(self):
        """
        Compile the circuit
        :return:
        """
        n_nodes = len(self.nodes)
        n_terminals = len(self.terminals)
        n_br = len(self.branches)
        n_sw = len(self.switches)
        self.nodes_idx = dict()  # dictionary of node object -> node index
        self.terminals_idx = dict()  # dictionary of terminals -> terminal index

        CN_T = csc_matrix((n_nodes, n_terminals), dtype=int)
        BR_T = csc_matrix((n_br, n_terminals), dtype=int)
        SW_T = csc_matrix((n_sw, n_terminals), dtype=int)
        SW_states = np.zeros((n_sw, 1), dtype=int)

        # Terminals
        for i, terminal in enumerate(self.terminals):
            self.terminals_idx[terminal] = i

        # Connectivity Nodes
        for i, node in enumerate(self.nodes):

            self.nodes_idx[node] = i

            terminals = self.node_terminal[node]
            for terminal in terminals:
                j = self.terminals_idx[terminal]
                CN_T[i, j] = 1

        # Switches
        for i, switch in enumerate(self.switches):
            j = self.terminals_idx[switch.t1]
            SW_T[i, j] = 1

            j = self.terminals_idx[switch.t2]
            SW_T[i, j] = 1

            SW_states[i, 0] = int(switch.state)

        # Branches
        for i, branch in enumerate(self.branches):
            j = self.terminals_idx[branch.t1]
            BR_T[i, j] = 1

            j = self.terminals_idx[branch.t2]
            BR_T[i, j] = 1

        print('CN_T\n', CN_T)
        print('CN_BR\n', BR_T)
        print('CN_SW\n', SW_T)
        print('SW_states\n', SW_states)

        # Compute the cross connectivity matrices to determine the Node-Branch connectivity
        SW_T_state = SW_T.multiply(SW_states)  # switches connectivity matrix with the switches state applied
        BR_SW = BR_T.dot(SW_T_state.transpose())  # Branch-Switch connectivity matrix
        CN_SW = CN_T.dot(SW_T_state.transpose())  # Node-Switch connectivity matrix
        CN_BR = CN_SW.dot(BR_SW.transpose())  # node-branch connectivity matrix

        pass


if __name__ == '__main__':

    circuit = Circuit()

    cn1 = ConnectivityNode('CN1')
    cn2 = ConnectivityNode('CN2')
    cn3 = ConnectivityNode('CN3')

    t1 = Terminal('T1')
    t2 = Terminal('T2')
    t3 = Terminal('T3')
    t4 = Terminal('T4')
    t5 = Terminal('T5')
    t6 = Terminal('T6')
    t7 = Terminal('T7')
    t8 = Terminal('T8')

    sw1 = Switch('SW1', t1, t2)
    sw2 = Switch('SW2', t3, t4)
    sw3 = Switch('SW3', t5, t6)
    sw4 = Switch('SW4', t7, t8)

    br1 = Branch('BR1', t2, t3)
    br2 = Branch('BR2', t6, t7)

    circuit.add_connectivity_node(cn1)
    circuit.add_connectivity_node(cn2)
    circuit.add_connectivity_node(cn3)

    circuit.add_terminal(t1)
    circuit.add_terminal(t2)
    circuit.add_terminal(t3)
    circuit.add_terminal(t4)
    circuit.add_terminal(t5)
    circuit.add_terminal(t6)
    circuit.add_terminal(t7)
    circuit.add_terminal(t8)

    circuit.add_node_terminal_relation(cn1, t1)
    circuit.add_node_terminal_relation(cn1, t5)
    circuit.add_node_terminal_relation(cn2, t4)
    circuit.add_node_terminal_relation(cn3, t8)

    circuit.add_branch(br1)
    circuit.add_branch(br2)

    circuit.add_switch(sw1)
    circuit.add_switch(sw2)
    circuit.add_switch(sw3)
    circuit.add_switch(sw4)

    circuit.compile()
