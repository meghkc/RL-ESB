There is one directory for each subset of instances. In each directory, there are three directories : one for the test instances, one for the training instances and one for the validation instances. An instance is described by 4 files. 


** File description (all times are in minutes) **

* depots.txt : each line describes a depot with the following format : {depot node};{default number of available buses};

* recharge.txt : each line describes a recharging station with the following format : {station node};{number of available chargers};

* voyages.txt : each line describes a trip with the following format : {trip id}; {departure node};{departure time};{arrival node};{arrival time};{line number};. For a real bus line numbered n, there are two bus lines 2n and 2n + 1, one in each direction. 

* hlp.txt : each line describes the deadhead time necessary to travel from one node of the network to another with the following format : {node from};{node to};{travel time};0;. The last 0 is not useful. 


** Computing the cost of a trip  **

* For a scheduled trip : zero (because all scheduled trips are supposed to be covered and yield a constant total cost)

* For a deadhead trip : unit travel cost (4 per minute) multiplied by travel time (found in the hlp.txt file)

Don't forget to add waiting cost, recharge cost, return to the depot cost and fixed cost.  


** Computing the energy consumption of a trip **

* For a scheduled trip : compute first the travel distance by multiplying travel time (found in the hlp.txt file) by speed (18/60 km per minute). Then compute the energy consumption by multiplying the travel distance by the energy consumption per travel distance (1050 Wh per km). Then compute the idle time during the scheduled trip by substracting the sum of the departure time and travel time (found in hlp.txt file) from the arrival time. This idle time is multiplied by the energy consumption per idle time (11000/60 Wh per minute).

* For a deadhead trip : compute first the travel distance by multiplying travel time (found in the hlp.txt file) by speed (18/60 km per minute). Then compute the energy consumption by multiplying travel distance by energy consumption per travel distance (1050 Wh per km). When a bus is waiting without passengers, we assume that no energy is consumed. 
