import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, lil_matrix, diags

from JacobianBased import IwamotoNR

np.set_printoptions(linewidth=10000, precision=3)

# pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Graph:
    """
    Program to count islands in boolean 2D matrix
    """

    def __init__(self, row, col, g):
        """
        :param row: number of columns
        :param col: number of rows
        :param g: adjacency matrix
        """
        self.ROW = row
        self.COL = col
        self.graph = g

    def is_safe(self, i, j, visited):
        """
        A function to check if a given cell (row, col) can be included in DFS
        :param i: row index
        :param j: column index
        :param visited: 2D array of visited elements
        :return: if it is safe or not
        """
        # row number is in range, column number is in range and value is 1 and not yet visited
        return 0 >= i < self.ROW and 0 >= j < self.COL and not visited[i][j] and self.graph[i][j]

    def dfs(self, i, j, visited):
        """
        A utility function to do DFS for a 2D boolean matrix.
        It only considers the 8 neighbours as adjacent vertices
        :param i: row index
        :param j: column index
        :param visited: 2D array of visited elements
        """

        # TODO: Use a proper DFS with sparsity considerations

        # These arrays are used to get row and column numbers of 8 neighbours of a given cell
        rowNbr = [-1, -1, -1, 0, 0, 1, 1, 1]
        colNbr = [-1, 0, 1, -1, 1, -1, 0, 1]

        # Mark this cell as visited
        visited[i][j] = True

        # Recur for all connected neighbours
        for k in range(8):
            if self.is_safe(i + rowNbr[k], j + colNbr[k], visited):
                self.dfs(i + rowNbr[k], j + colNbr[k], visited)

    def count_islands(self):
        """
        The main function that returns count of islands in a given boolean 2D matrix
        :return: count of islands
        """

        # Make a bool array to mark visited cells. Initially all cells are unvisited
        # TODO: Replace with sparse matrix
        visited = [[False for j in range(self.COL)] for i in range(self.ROW)]

        # Initialize count as 0 and traverse through the all cells of given matrix
        count = 0
        # TODO: replace with sparse version
        for i in range(self.ROW):
            for j in range(self.COL):
                # If a cell with value 1 is not visited yet, then new island found
                if not visited[i][j] and self.graph[i][j] == 1:
                    # Visit all cells in this island and increment island count
                    self.dfs(i, j, visited)
                    count += 1

        return count


class Terminal:

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class ConnectivityNode:

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class ShuntDevice:

    def __init__(self, name, terminal: Terminal):
        self.name = name
        self.terminal = terminal

    def __str__(self):
        return self.name


class Load(ShuntDevice):

    def __init__(self, name, terminal: Terminal, P=0, Q=0):

        ShuntDevice.__init__(self, name, terminal)

        self.P = P
        self.Q = Q


class Shunt(ShuntDevice):

    def __init__(self, name, terminal: Terminal, G=0, B=0):

        ShuntDevice.__init__(self, name, terminal)

        self.G = G
        self.B = B


class Generator(ShuntDevice):

    def __init__(self, name, terminal: Terminal, P=0, Vset=0):

        ShuntDevice.__init__(self, name, terminal)

        self.P = P
        self.Vset = Vset


class Branch:

    def __init__(self, name, t1, t2):
        self.name = name
        self.t1 = t1
        self.t2 = t2

    def get_y(self):

        return 100.0, 0.0, 0.0, 100.0

    def __str__(self):
        return self.name


class Jumper(Branch):

    def __init__(self, name, t1, t2):

        Branch.__init__(self, name, t1, t2)


class Switch(Branch):

    def __init__(self, name, t1, t2, state=True):

        Branch.__init__(self, name, t1, t2)

        self.state = state


class Line(Branch):

    def __init__(self, name, t1, t2, r=0, x=0, r0=0, x0=0, g=0, b=0, g0=0, b0=0, length=1, tap_module=1.0, tap_angle=0):

        Branch.__init__(self, name, t1, t2)

        self.r = r
        self.x = x
        self.r0 = r0
        self.x0 = x0

        self.g = g
        self.b = b
        self.g0 = g0
        self.b0 = b0

        self.length = length

        self.tap_module = tap_module
        self.tap_angle = tap_angle

    def get_y(self):

        tap = self.tap_module * np.exp(-1j * self.tap_angle)

        Ysh = complex(self.g * self.length, self.b * self.length) / 2

        if self.r > 0 or self.x:
            Ys = 1 / complex(self.r * self.length, self.x * self.length)
        else:
            raise ValueError("The impedance at " + self.name + " is zero")

        Ytt = Ys + Ysh
        Yff = Ytt / (tap * np.conj(tap))
        Yft = - Ys / np.conj(tap)
        Ytf = - Ys / tap

        return Yff, Yft, Ytf, Ytt


class Connectivity:

    def __init__(self, n_terminals, n_nodes, n_br, n_sw, n_ld, n_gen, n_sh, Sbase):
        """
        Constructor
        :param n_terminals: number of terminals
        :param n_nodes: number of nodes
        :param n_br: number of branches
        :param n_sw: number of switches
        :param n_ld: number of loads
        :param n_gen: number of generators
        :param n_sh: number of shunts
        """

        self.Sbase = Sbase

        # connectivity nodes - terminals matrix
        self.CN_T = lil_matrix((n_nodes, n_terminals), dtype=int)

        # lines, transformers and jumpers to terminals matrix
        self.BR_T_f = lil_matrix((n_br, n_terminals), dtype=int)
        self.BR_T_t = lil_matrix((n_br, n_terminals), dtype=int)

        # switches - terminals matrix
        self.SW_T = lil_matrix((n_sw, n_terminals), dtype=int)
        self.SW_states = np.zeros(n_sw, dtype=int)

        # shunt elements (loads, shunts, generators)
        self.LD_T = lil_matrix((n_ld, n_terminals), dtype=int)
        self.GEN_T = lil_matrix((n_gen, n_terminals), dtype=int)
        self.SH_T = lil_matrix((n_sh, n_terminals), dtype=int)

        # admittance components vectors
        self.BR_yff = np.zeros(n_br, dtype=complex)
        self.BR_yft = np.zeros(n_br, dtype=complex)
        self.BR_ytf = np.zeros(n_br, dtype=complex)
        self.BR_ytt = np.zeros(n_br, dtype=complex)

        # load generation and shunts
        self.LD_Power = np.zeros(n_ld, dtype=complex)
        self.Gen_Power = np.zeros(n_gen, dtype=float)
        self.Gen_voltage = np.zeros(n_gen, dtype=float)
        self.SH_Power = np.zeros(n_sh, dtype=complex)

        # names
        self.T_names = [None] * n_terminals
        self.CN_names = [None] * n_nodes
        self.BR_names = [None] * n_br
        self.SW_names = [None] * n_sw
        self.LD_names = [None] * n_ld
        self.GEN_names = [None] * n_gen
        self.SH_names = [None] * n_sh

        # resulting matrices
        self.BR_CN = None  # nodes - branch
        self.CN_CN = None  # node - node
        self.SW_T_state = None  # switch - terminal with the switches state applied
        self.BR_SW_f = None  # branch - switch
        self.BR_SW_t = None  # branch - switch
        self.CN_SW = None  # node - switch
        self.LD_CN = None  # load - node
        self.GEN_CN = None  # generator - node
        self.SH_CN = None  # shunt - node

        # resulting matrices
        self.Cf = None
        self.Ct = None
        self.Yf = None
        self.Yt = None
        self.Ybus = None
        self.Ysh = None
        self.Sbus = None
        self.Ibus = None
        self.Vbus = None
        self.types = None
        self.pq = None
        self.pv = None
        self.ref = None

    def compute(self):
        """
        Compute the cross connectivity matrices to determine the circuit connectivity towards the calculation
        Additionally, compute the calculation matrices
        """

        # --------------------------------------------------------------------------------------------------------------
        # Connectivity matrices
        # --------------------------------------------------------------------------------------------------------------

        # switches connectivity matrix with the switches state applied
        self.SW_T_state = diags(self.SW_states) * self.SW_T

        # Branch-Switch connectivity matrix
        self.BR_SW_f = self.BR_T_f * self.SW_T_state.transpose()
        self.BR_SW_t = self.BR_T_t * self.SW_T_state.transpose()

        # Node-Switch connectivity matrix
        self.CN_SW = self.CN_T * self.SW_T_state.transpose()

        # load-Connectivity Node matrix
        self.LD_CN = self.LD_T * self.CN_T.transpose()

        # generator-Connectivity Node matrix
        self.GEN_CN = self.GEN_T * self.CN_T.transpose()

        # shunt-Connectivity Node matrix
        self.SH_CN = self.SH_T * self.CN_T.transpose()

        # branch-node connectivity matrix (Equals A^t)
        # A branch and a node can be connected via a switch or directly
        self.Cf = self.CN_SW * self.BR_SW_f.transpose() + self.CN_T * self.BR_T_f.transpose()
        self.Ct = self.CN_SW * self.BR_SW_t.transpose() + self.CN_T * self.BR_T_t.transpose()
        self.BR_CN = (self.Cf - self.Ct).transpose()

        # node-node connectivity matrix
        self.CN_CN = self.BR_CN.transpose() * self.BR_CN
        self.CN_CN = self.CN_CN.astype(bool).astype(int)

        # --------------------------------------------------------------------------------------------------------------
        # Calculation matrices
        # --------------------------------------------------------------------------------------------------------------

        # form the power injections vector
        PD = self.LD_CN.transpose() * self.LD_Power  # demand (complex)
        PG = self.GEN_CN.transpose() * self.Gen_Power  # generation (real)
        self.Sbus = (PG - PD) / self.Sbase
        self.Ibus = np.zeros_like(self.Sbus)

        # types logic:
        # if the number is < 10 -> PQ
        # if the number is >= 10 -> PV
        # later, choose a PV gen as Slack
        self.types = (self.LD_CN.sum(axis=0).A1 + self.GEN_CN.sum(axis=0).A1 * 10).reshape(-1)

        # Voltage vector
        # self.Vbus = self.GEN_CN.transpose() * self.Gen_voltage
        self.Vbus = np.ones_like(self.Sbus)

        # form the shunt vector
        self.Ysh = self.SH_CN.transpose() * self.SH_Power

        # form the admittance matrix
        self.Yf = diags(self.BR_yff) * self.Cf.transpose() + diags(self.BR_yft) * self.Ct.transpose()
        self.Yt = diags(self.BR_ytf) * self.Cf.transpose() + diags(self.BR_ytt) * self.Ct.transpose()
        self.Ybus = self.Cf * self.Yf + self.Ct * self.Yt + diags(self.Ysh)

        self.pq = np.where(self.types < 10)[0]
        self.pv = np.where(self.types >= 10)[0]
        if self.ref is None:
            self.ref = self.pv[0]
            self.pv = self.pv[:-1]  # pick all bu the first, which is not a ref

    def print(self):
        """
        print the connectivity matrices
        :return:
        """
        print('\nCN_T\n', pd.DataFrame(self.CN_T.todense(), index=self.CN_names, columns=self.T_names).to_latex())
        print('\nBR_T_f\n', pd.DataFrame(self.BR_T_f.todense(), index=self.BR_names, columns=self.T_names).to_latex())
        print('\nBR_T_t\n', pd.DataFrame(self.BR_T_t.todense(), index=self.BR_names, columns=self.T_names).to_latex())
        print('\nSW_T\n', pd.DataFrame(self.SW_T.todense(), index=self.SW_names, columns=self.T_names).to_latex())
        print('\nSW_states\n', pd.DataFrame(self.SW_states, index=self.SW_names, columns=['States']).to_latex())

        # resulting
        print('\n\n' + '-' * 40 + ' RESULTS ' + '-' * 40 + '\n')
        print('\nLD_CN\n', pd.DataFrame(self.LD_CN.todense(), index=self.LD_names, columns=self.CN_names).to_latex())
        print('\nSH_CN\n', pd.DataFrame(self.SH_CN.todense(), index=self.SH_names, columns=self.CN_names).to_latex())
        print('\nGEN_CN\n', pd.DataFrame(self.GEN_CN.todense(), index=self.GEN_names, columns=self.CN_names).to_latex())
        print('\nBR_CN\n', pd.DataFrame(self.BR_CN.astype(int).todense(), index=self.BR_names, columns=self.CN_names).to_latex())
        print('\nCN_CN\n', pd.DataFrame(self.CN_CN.todense(), index=self.CN_names, columns=self.CN_names).to_latex())

        print('\ntypes\n', self.types)
        print('\nSbus\n', self.Sbus)
        print('\nVbus\n', self.Vbus)
        print('\nYsh\n', self.Ysh)
        print('\nYbus\n', self.Ybus.todense())


class Circuit:

    def __init__(self, Sbase=100):
        """
        Circuit constructor
        """

        self.Sbase = Sbase

        self.connectivity_nodes = list()
        self.terminals = list()

        self.switches = list()
        self.branches = list()
        self.jumpers = list()

        self.loads = list()
        self.shunts = list()
        self.generators = list()

        self.nodes_idx = dict()
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
        self.connectivity_nodes.append(node)

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

    def add_jumper(self, jumper):
        """

        :param jumper:
        """
        self.jumpers.append(jumper)

    def add_load(self, load):
        """

        :param load:
        """
        self.loads.append(load)

    def add_shunt(self, shunt):
        """

        :param shunt:
        """
        self.shunts.append(shunt)

    def add_generator(self, generator):
        """

        :param generator:
        """
        self.generators.append(generator)

    def load_file(self, fname):
        """
        Load file
        :param fname: file name
        """
        xls = pd.ExcelFile(fname)

        # Terminals
        T_dict = dict()
        df = pd.read_excel(xls, 'Terminals')
        for i in range(df.shape[0]):
            val = df.values[i, 0]
            T = Terminal(val)
            T_dict[val] = T
            self.add_terminal(T)

        # ConnectivityNodes
        CN_dict = dict()
        df = pd.read_excel(xls, 'ConnectivityNodes')
        for i in range(df.shape[0]):
            val = df.values[i, 0]
            CN = ConnectivityNode(val)
            CN_dict[val] = CN
            self.add_connectivity_node(CN)

        # Branches
        df = pd.read_excel(xls, 'Branches')
        for i in range(df.shape[0]):
            T1 = T_dict[df.values[i, 1]]
            T2 = T_dict[df.values[i, 2]]

            r = df.values[i, 3]
            x = df.values[i, 4]
            r0 = df.values[i, 5]
            x0 = df.values[i, 6]

            g = df.values[i, 7]
            b = df.values[i, 8]
            g0 = df.values[i, 9]
            b0 = df.values[i, 10]

            l = df.values[i, 11]

            self.add_branch(Line(df.values[i, 0], T1, T2, r, x, r0, x0, g, b, g0, b0, l))

        df = pd.read_excel(xls, 'Jumpers')
        for i in range(df.shape[0]):
            T1 = T_dict[df.values[i, 1]]
            T2 = T_dict[df.values[i, 2]]
            self.add_branch(Jumper(df.values[i, 0], T1, T2))

        # Switches
        df = pd.read_excel(xls, 'Switches')
        for i in range(df.shape[0]):
            T1 = T_dict[df.values[i, 1]]
            T2 = T_dict[df.values[i, 2]]
            state = bool(df.values[i, 3])
            self.add_switch(Switch(df.values[i, 0], T1, T2, state))

        # Loads
        df = pd.read_excel(xls, 'Loads')
        for i in range(df.shape[0]):
            T1 = T_dict[df.values[i, 1]]
            p = df.values[i, 2]
            q = df.values[i, 3]
            self.add_load(Load(df.values[i, 0], T1, p, q))

        # shunts
        df = pd.read_excel(xls, 'Shunts')
        for i in range(df.shape[0]):
            T1 = T_dict[df.values[i, 1]]
            g = df.values[i, 2]
            b = df.values[i, 3]
            self.add_shunt(Shunt(df.values[i, 0], T1, g, b))

        # Generators
        df = pd.read_excel(xls, 'Generators')
        for i in range(df.shape[0]):
            T1 = T_dict[df.values[i, 1]]
            p = df.values[i, 2]
            vset = df.values[i, 3]
            self.add_generator(Generator(df.values[i, 0], T1, p, vset))

        # CN_T
        df = pd.read_excel(xls, 'CN_T')
        for i in range(df.shape[0]):
            CN = CN_dict[df.values[i, 0]]
            T = T_dict[df.values[i, 1]]
            self.add_node_terminal_relation(CN, T)

    def compile(self):
        """
        Compile the circuit
        """

        n_nodes = len(self.connectivity_nodes)
        n_terminals = len(self.terminals)
        n_br = len(self.branches) + len(self.jumpers)
        n_sw = len(self.switches)
        n_ld = len(self.loads)
        n_gen = len(self.generators)
        n_sh = len(self.shunts)

        self.nodes_idx = dict()  # dictionary of node object -> node index
        self.terminals_idx = dict()  # dictionary of terminals -> terminal index

        conn = Connectivity(n_terminals=n_terminals,
                            n_nodes=n_nodes,
                            n_br=n_br,
                            n_sw=n_sw,
                            n_ld=n_ld,
                            n_gen=n_gen,
                            n_sh=n_sh,
                            Sbase=self.Sbase)

        # Terminals
        for i, terminal in enumerate(self.terminals):

            self.terminals_idx[terminal] = i
            conn.T_names[i] = terminal.name

        # Connectivity Nodes
        for i, node in enumerate(self.connectivity_nodes):

            self.nodes_idx[node] = i
            conn.CN_names[i] = node.name
            terminals = self.node_terminal[node]

            for terminal in terminals:
                j = self.terminals_idx[terminal]
                conn.CN_T[i, j] = 1

        # Switches
        for i, switch in enumerate(self.switches):

            j = self.terminals_idx[switch.t1]
            conn.SW_T[i, j] = 1

            j = self.terminals_idx[switch.t2]
            conn.SW_T[i, j] = 1

            conn.SW_states[i] = int(switch.state)
            conn.SW_names[i] = switch.name

        # Branches (lines, transformers and jumpers)
        for i, branch in enumerate(self.branches):

            # from
            f = self.terminals_idx[branch.t1]
            conn.BR_T_f[i, f] = 1

            # to
            t = self.terminals_idx[branch.t2]
            conn.BR_T_t[i, t] = 1

            # name
            conn.BR_names[i] = branch.name

            # branch admittances
            yff, yft, ytf, ytt = branch.get_y()
            conn.BR_yff[i] = yff
            conn.BR_yft[i] = yft
            conn.BR_ytf[i] = ytf
            conn.BR_ytt[i] = ytt

        # Loads
        for i, load in enumerate(self.loads):
            j = self.terminals_idx[load.terminal]
            conn.LD_T[i, j] = 1
            conn.LD_names[i] = load.name
            conn.LD_Power[i] = complex(load.P, load.Q)

        # Generators
        for i, generator in enumerate(self.generators):
            j = self.terminals_idx[generator.terminal]
            conn.GEN_T[i, j] = 1
            conn.GEN_names[i] = generator.name
            conn.Gen_Power[i] = generator.P
            conn.Gen_voltage[i] = generator.Vset

        # Shunts
        for i, shunt in enumerate(self.shunts):
            j = self.terminals_idx[shunt.terminal]
            conn.SH_T[i, j] = 1
            conn.SH_names[i] = shunt.name
            conn.SH_Power[i] = complex(shunt.G, shunt.B)

        # compute topology
        conn.compute()

        return conn


class PowerFlow:

    def __init__(self, circuit: Circuit):

        self.circuit = circuit

    def run(self):
        """
        Run power flow
        :return:
        """

        # compile circuit
        conn = self.circuit.compile()

        # run power flow
        V, converged, normF, Scalc, iter_, elapsed = IwamotoNR(Ybus=conn.Ybus,
                                                               Sbus=conn.Sbus,
                                                               V0=conn.Vbus,
                                                               Ibus=conn.Ibus,
                                                               pv=conn.pv,
                                                               pq=conn.pq,
                                                               tol=conn.ref,
                                                               max_it=15,
                                                               robust=False)
        return V


if __name__ == '__main__':

    circuit = Circuit()
    # circuit.load_file('substation_data.xlsx')
    circuit.load_file('lynn5.xlsx')

    conn_ = circuit.compile()
    conn_.print()

    pf = PowerFlow(circuit)
    Vsol = pf.run()
    print('\nVsol:', np.abs(Vsol))
