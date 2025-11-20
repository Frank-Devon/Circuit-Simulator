
To run program type: python3 -i main.py my_netlist

netlist has similiar syntax to spice, except, nodes are to be numbered 0 to N (where N is an actual number), in sequential order, and the last node is ground. Also type GND N needs to be typed into the netlist.

RLC components mu
example of inductor syntax
L1 2 3 0.1 1.5
L - means inductor, the rest of the word is just a name. 2 and 3 are the nodes. 0.1 is the value of the inductor, 1.5 is the inital state of the inductor (1.5 Amps). If initial state is 0, the last number can 

Independent voltage and current sources actually act like step functions at t = 0. They go from 0 to their "stated" value at t = 0

t_delta and t_end are adjustable in the script.

