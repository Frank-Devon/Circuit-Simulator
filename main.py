import numpy as np
from matplotlib import pyplot as plt
import sys
from collections import deque
import copy
import time
import code

comp_t_start = time.perf_counter()

# read in circuit description and create incidence matrix
def FirstEqualIndices( nums):
    index_map = {}
    for index, num in enumerate(nums):
        if num in index_map:
            return (index_map[num], index)
        index_map[num] = index
    return None  # Return None if no equal integers are found


class Branch:
    def __init__(self, node_p, node_m, comp_type, value, name, state=0):
        self.node_p = node_p
        self.node_m = node_m
        self.comp_type = comp_type 
        self.value = value
        self.name = name
        self.state = state # initial voltage or cur of cap or inductor

    def __str__(self):
        return f"Branch {self.name}: {self.comp_type}, value: {self.value}, {self.node_p}, {self.node_m}, {self.state}" 
np.set_printoptions(precision=4, suppress=False, floatmode='fixed')

t = 0.0 # time
t_delta = 0.01# time step
A = np.zeros((1, 1)) # Incidence matrix, 
U = [] # list of 2-tuple, first element of tuple refers to a node variable, 2nd delta V
C = np.zeros((1, 1)) # conductance matrix
W = np.zeros((1, 1)) # branch currents
b = np.zeros(1) # battery matrix
Branches = []
Branch_names = []
Nodes_set = set()

GND = 0 # ground node, init to 0 #TODO currently only allowed to be last node, fix
#checking for valid circuit file
GND_found = False
Error_found = False

if len(sys.argv) > 1:
    circuit_name = sys.argv[1]
else:
    circuit_name = 'circuit.dspice'

# Make Branches list
with open(circuit_name, 'r') as file:
    for line in file:
        # s contains words in a line of the file
        s = line.strip().split()
        char_start = s[0][0]
        if char_start == '*':
            continue
        elif char_start == 'G':
            if GND_found == False:
                GND_found = True
                GND = int(s[1])
            else :
                Error_found = True # can't have 2 ground nodes
                break
        # good enough, but could make this more "elegant"
        elif char_start == 'V':
            Branches.append(Branch(int(s[1]), int(s[2]), 'V', float(s[3]), s[0]))
        elif char_start == 'I':
            Branches.append(Branch(int(s[1]), int(s[2]), 'I', float(s[3]), s[0]))
        elif char_start == 'R':
            Branches.append(Branch(int(s[1]), int(s[2]), 'R', float(s[3]), s[0]))
        elif char_start == 'C': 
            if len(s) <= 4: # find initial voltage
                init_cond = 0.0 
            if len(s) > 4:
                init_cond = float(s[4])
            Branches.append(Branch(int(s[1]),int(s[2]),'C', float(s[3]),s[0], init_cond))
        elif char_start == 'L':
            if len(s) <= 4: # find initial current
                init_cond = 0.0
            if len(s) > 4:
                init_cond = float(s[4])
            Branches.append(Branch(int(s[1]),int(s[2]),'L', float(s[3]),s[0], init_cond))

        else:
            Error_found = True
            break
        print (line.strip() + "*** word count = " + str(len(s)))
    
    print(f"was error found: {Error_found}")
    
if Error_found:
    print ("ERROR FOUND: exiting program")
    exit()

#placing branch voltage sources at the end of list
#Branches = sorted(Branches, key=lambda b: (b.comp_type == 'V', b.comp_type))
Branches = sorted(Branches, key=lambda b: (b.comp_type == 'V', b.comp_type == 'I', b.comp_type == 'C'))
print("hello")

for branch in Branches:
    Nodes_set.add(branch.node_p)
    Nodes_set.add(branch.node_m)
    Branch_names.append(branch.name)

# make sure nodes are consecutive integers that start at 0
#if max(Nodes_set) - min(Nodes_set) != len(Nodes_set) - 1:
if min(Nodes_set) != 0 or max(Nodes_set) + 1 != len(Nodes_set):
    print ("ERROR FOUND: node numbers must be consecutive AND start at 0")
    exit()

# make sure last node is ground
if max(Nodes_set) != GND:
    print ("ERROR GND node must be last numbered node")
    exit()


# construct U vector (must match the number of nodes)
# also, node numbers are guarenteed to be consecutive
U.extend([0] * len(Nodes_set))
for i in range(len(U)):
    U[i] = [i, 0] 

# creating incidence matrix
A = np.zeros((len(Branches), len(Nodes_set))) # correct size
for i in range(len(Branches)):
    A[i, Branches[i].node_p] = 1   
    A[i, Branches[i].node_m] = -1   
    #TODO keep track of rows here?



#remove row (branch) associated with voltage sources (and capacitors, inductors)
rows_to_delete = []
for i in range(len(Branches)):
    if Branches[i].comp_type == 'V':
        #A = np.delete(A, i, axis=0)
        rows_to_delete.append(i)
    if Branches[i].comp_type == 'I':
        rows_to_delete.append(i)
    if Branches[i].comp_type == 'C':
        # capacitors are treated like cur dependent v-sources
        rows_to_delete.append(i)
    if Branches[i].comp_type == 'L':
        rows_to_delete.append(i)
     
A_untrimmed = A # useful for calculating branch currents in voltage sources? maybe delete
A = np.delete(A, rows_to_delete, axis=0)

#TODO ! this is where iteration starts
# this will simulate one time step
def func_main():
    global b
    global W_full
    global U_volts
    # call everytime a V-source is accomodated for
    # this way, all dependent nodes will be expressed as functions of fundemental nodes.
    # also call at the end of 'add voltage sources' routine
    def FixU(u_changed):
        #u_node was just adjusted, see if any other nodes depend on u_node
        for i, (u_, delta) in enumerate(U):
            if u_ == u_changed:
                U[i][0] = U[u_changed][0]
                U[i][1] = U[u_changed][1] + U[i][1]
    
    for branch in Branches:
        if branch.comp_type == 'V':
            U[branch.node_p] = [branch.node_m, branch.value]
            FixU(branch.node_p)
        elif branch.comp_type == 'C': # treat capacitors like v-sources
            U[branch.node_p] = [branch.node_m, branch.state]
            FixU(branch.node_p)
    
    FixU(branch.node_p) # final fix-up
   
    
    #
    # find b "battery" matrix, containing independent voltage sources
    #
    b = np.zeros(np.size(A, 0)) # get row count of A
    for i, u_node in enumerate(U):
        if u_node[1] != 0:
            u_temp = np.zeros(np.size(A, 1))
            u_temp[i] = u_node[1]
            b = b + A @ u_temp 
    
    # construct C, conductance matrix
    # C has same size as A
    C = np.zeros((np.size(A, 0), np.size(A, 0 ))) # 0 means row size
    for i in range(len(Branches)):
        if Branches[i].comp_type == 'R':
            C[i, i] = 1.0 / Branches[i].value
            #TODO put branch names here?
    
    #
    # Trim A
    #
    
    # must also have a trimmed U vector
    U_trimmed = copy.deepcopy(U)
    
    
    # remove ground node. (A and U must be edited)
    del U_trimmed[GND] 
    # find # of "essential nodes"
    e_node_num = 0
    for u in U:
        if u[1] == 0:
            e_node_num += 1
    e_node_num -= 1 # don't count ground node
    #print(f"num essential nodes: {e_node_num}")
    A_trimmed = np.zeros((np.size(A,0), e_node_num))
    
    # fix up A
    avail_index = 0
    A_dict = {} # key: node variable (represented by a #), value is associated column 
                # in A_trimmed
    
    for i, u_ in enumerate(U_trimmed):
        if u_[0] == GND:
            continue
        A_index = A_dict.get(u_[0], None)
        if A_index == None:
            A_index = avail_index
            avail_index += 1 # could bounds check
            A_dict.update({u_[0] : A_index})
        A_trimmed[:, A_index] += A[:, i] 
    
    
    # construct vector f (from current sources)
    f = np.zeros(e_node_num)
    f_index = 0
    for i, u_ in enumerate(U_trimmed):
        if u_[1] == 0: # no v-delta, so 'non-dependent' node
            for branch in Branches:
                if not (branch.comp_type == 'I' or branch.comp_type == 'L'):
                    continue
                cur = 0.0
                if branch.comp_type == 'I':
                    cur = branch.value
                elif branch.comp_type == 'L':
                    cur = branch.state
                if branch.node_p == u_[0]:
                    f[f_index] = f[f_index] +  cur # branch.value
                elif branch.node_m == u_[0]:
                    #sub branch.value from f @ some index
                    f[f_index] = f[f_index] - cur # branch.value

            f_index += 1        
    
    
    # finally calculate node voltages (okay)
    U_volts_trimmed = - np.linalg.pinv(A_trimmed.T @ C @ A_trimmed) @ (A_trimmed.T @ C @ b + f)
    
    # calculate more node voltages (include nodes adjacent to sources)
    U_volts = []  # !!!!!!!!!! ONE OF THE MOST IMPORTANT VARIABLES !!!!!!!!!!!!
    for node, delta in U_trimmed:
        if node != GND:
            U_volts.append( delta + U_volts_trimmed[A_dict.get(node, None)])
        else:
            U_volts.append( delta)
    
    # W is currents
    W = C @ (b + A_trimmed @ U_volts_trimmed)
    
    # W represents branch currents of non source elements
    # W_full includes all branch currents
    W_full_size = np.shape(A_untrimmed.T)[1]
    W_full = np.zeros(W_full_size) # !!! ONE OF THE MOST IMPORTANT VARIABLES!!!!
    W_full[:np.size(W)] = W
    Z = - A_untrimmed.T @ W_full
    Y = (A_untrimmed.T)[:, -(W_full_size - np.size(W)):]
    W_source = np.linalg.pinv(Y) @ Z
    W_full[-np.size(W_source):] = W_source
    #W_full now has all the branch currents
    
    # calculating all voltage drops of each element/branch
    A_untrimmed_no_gnd = A_untrimmed[:, :-1]
    Branch_Voltages = A_untrimmed_no_gnd @ U_volts
    
#    print ("********************************************************************************")
#    print ("Branch Name, Component Type, Component Value, Voltage Difference, Current, Power")
#    for i in range(len(Branches)):
#        watts = Branch_Voltages[i] * W_full[i]
#        if Branches[i].comp_type == 'V':
#            comp_type_str = "Volt Source"
#        elif Branches[i].comp_type == 'R':
#            comp_type_str = "Resistor"
#        else :
#            comp_type_str =  "Unknown"
#        print (f"{Branch_names[i]}, {comp_type_str}, {Branches[i].value}, {Branch_Voltages[i]} Volts, {W_full[i]} Amps, {watts} Watts")
#    # prints node voltages
#    for i, v in enumerate(U_volts):
#        print (f"Node {i} = {v} Volts")

    # now must calculate new states (voltages) of capacitors
    for i, current in enumerate (W_full):
        if Branches[i].comp_type == 'C':
            Branches[i].state += (1.0/Branches[i].value) * t_delta * current
       
    for branch in Branches:
        if branch.comp_type == 'L':
            voltage = U_volts[branch.node_p] - U_volts[branch.node_m]
            branch.state += (1.0/branch.value) * t_delta * voltage

    #code.interact(local={**globals(), **locals()})
   
# start iterating
t = 0.0
# simulation setup variables
t_start = 0.0
t_end = 40.0
t = t_start
W_full_list = []
U_volts_list = []
t_list = []
while t <= t_end: 
    # working euler method below
    func_main()
    W_full_list.append(W_full)
    U_volts_list.append(U_volts)
    t_list.append(t)
    t += t_delta 

bogus_time = np.linspace(t_start, t_end, int((t_end - t_start) / t_delta + 1.0))

#
#Below are tests and plots for different circuits
#
#volt_cap = [expression for item in iterable if condition == True]
# for cap0.dspice 
#volt_cap = [u_vec[1] - u_vec[3] for u_vec in U_volts_list if  True]
# for cap1.dspice no cap in this circuit
#volt_cap = [u_vec[1] for u_vec in U_volts_list if  True]
# cap2.dspice
#volt_cap = [u_vec[1] for u_vec in U_volts_list if  True]
# cap3.dspice
# volt_cap = [u_vec[1] - u_vec[3] for u_vec in U_volts_list if  True]
# cap4.dspice
#volt_cap = [u_vec[1] - u_vec[2] for u_vec in U_volts_list if  True]
# opencir.dspice
#volt_cap = [u_vec[1] for u_vec in U_volts_list if  True]
# ind0.dspice
#volt_cap = [u_vec[1] - u_vec[2] for u_vec in U_volts_list if  True]
# rlc0.dspice
volt_cap = [u_vec[1] - u_vec[2] for u_vec in U_volts_list if  True]
## rlc1.dspice
#volt_cap = [u_vec[2] - u_vec[3] for u_vec in U_volts_list if  True]
#volt_ind = [u_vec[1] - u_vec[2] for u_vec in U_volts_list if  True]

comp_t_end = time.perf_counter()
print(f"Solved in: {comp_t_end - comp_t_start} seconds")

show_plot = True
if show_plot == True:
    plt.plot(t_list, volt_cap, label='Capacitor Voltage', color='blue')
    #plt.plot(t_list, volt_ind, label='Inductor Voltage', color='red')
    
    # Add titles and labels
    plt.title('Parallel RLC')
    plt.xlabel('Time (sec)')
    plt.ylabel('Voltage')
    plt.legend()
    
    # Display the graph
    plt.show()

# write results to plain text file
# open file
with open(circuit_name + ".out.txt", 'w+') as f:
    # write elements of list
    for i in range(len(t_list)):
        f.write(str(t_list[i]) + "   " + str(volt_cap[i]) + '\n')

    print("File written successfully")

f.close()

print("bye")
