from Mark4 import Mk4SaveDataToCSV, simulation
import pterasoftware as ps
import numpy as np
import matplotlib.pyplot as plt
from SelfFunctions import extract_forces, extract_second_cycle
import os 
import csv

def Mark4Simulation(FlappingPeriods, Angles_of_Attacks, Air_Speeds, mw_root_chords, mw_wingspans, mw_tip_chords, 
                    tail_bpositions, tail_root_chords, tail_tip_chords, tail_wingspans):
    n = 0
    
    # Function call
    for i in FlappingPeriods:
        for j in Air_Speeds:
            for k in Angles_of_Attacks:
                
                for iterate_mw_root_chords in mw_root_chords:
                    lift, induced_drag, pitching_moment = simulation(va= j, aoa = k, fp = i, mw_airfoil = "naca8304", mw_root_chord = iterate_mw_root_chords, mw_wingspan = 1.4,
                                                                    mw_tip_chord = 0.2, tail_bposition = 0.45, tail_type = "V-Tail", tail_root_chord =0.2 , 
                                                                    tail_tip_chord = 0.01, tail_wingspan = 0.4, tail_airfoil = "naca0004")
                    # Flatten the arrays
                    lift = lift.flatten()
                    induced_drag = induced_drag.flatten()
                    pitching_moment = pitching_moment.flatten()
                    print("Before Second Cycle Extraction :  ", lift.shape, induced_drag.shape, pitching_moment.shape)
                    print("Before Second Cycle Extraction value:  ", lift, induced_drag, pitching_moment)

                    # Extract the second cycle from the data
                    lift, induced_drag, pitching_moment = extract_second_cycle(lift, induced_drag, pitching_moment)
                    print("After Second Cycle Extraction :  ",lift.shape, induced_drag.shape, pitching_moment.shape)
                    print("After Second Cycle Extraction value:  ", lift, induced_drag, pitching_moment)

                    # Save data to CSV with a unique filename for each combination
                    file_name = f"Data/Data_Instance{n}.csv"
                    Mk4SaveDataToCSV(i, j, k, lift, pitching_moment, induced_drag, file_name,iterate_mw_root_chords, 1.4, 
                                        0.2, 0.45, 0.2 , 0.01, 0.4)
                    n += 1



                

                for iterate_mw_wingspans in mw_wingspans:
                    lift, induced_drag, pitching_moment = simulation(va= j, aoa = k, fp = i, mw_airfoil = "naca8304", mw_root_chord = 0.3, mw_wingspan = 1.4,
                                                                    mw_tip_chord = 0.2, tail_bposition = 0.45, tail_type = "V-Tail", tail_root_chord =0.2 , 
                                                                    tail_tip_chord = 0.01, tail_wingspan = 0.4, tail_airfoil = "naca0004")
                    # Flatten the arrays
                    lift = lift.flatten()
                    induced_drag = induced_drag.flatten()
                    pitching_moment = pitching_moment.flatten()
                    print("Before Second Cycle Extraction :  ", lift.shape, induced_drag.shape, pitching_moment.shape)
                    print("Before Second Cycle Extraction value:  ", lift, induced_drag, pitching_moment)

                    # Extract the second cycle from the data
                    lift, induced_drag, pitching_moment = extract_second_cycle(lift, induced_drag, pitching_moment)
                    print("After Second Cycle Extraction :  ",lift.shape, induced_drag.shape, pitching_moment.shape)
                    print("After Second Cycle Extraction value:  ", lift, induced_drag, pitching_moment)

                    # Save data to CSV with a unique filename for each combination
                    file_name = f"Data/Data_Instance{n}.csv"
                    Mk4SaveDataToCSV(i, j, k, lift, pitching_moment, induced_drag, file_name,0.3, iterate_mw_wingspans, 
                                        0.2, 0.45, 0.2 , 0.01, 0.4)
                    n += 1

                
                for iterate_mw_tip_chords in mw_tip_chords:
                    lift, induced_drag, pitching_moment = simulation(va= j, aoa = k, fp = i, mw_airfoil = "naca8304", mw_root_chord = 0.3, mw_wingspan = iterate_mw_wingspans,
                                                                    mw_tip_chord = iterate_mw_tip_chords, tail_bposition = 0.45, tail_type = "V-Tail", tail_root_chord =0.2 , 
                                                                    tail_tip_chord = 0.01, tail_wingspan = 0.4, tail_airfoil = "naca0004")
                    # Flatten the arrays
                    lift = lift.flatten()
                    induced_drag = induced_drag.flatten()
                    pitching_moment = pitching_moment.flatten()
                    print("Before Second Cycle Extraction :  ", lift.shape, induced_drag.shape, pitching_moment.shape)
                    print("Before Second Cycle Extraction value:  ", lift, induced_drag, pitching_moment)

                    # Extract the second cycle from the data
                    lift, induced_drag, pitching_moment = extract_second_cycle(lift, induced_drag, pitching_moment)
                    print("After Second Cycle Extraction :  ",lift.shape, induced_drag.shape, pitching_moment.shape)
                    print("After Second Cycle Extraction value:  ", lift, induced_drag, pitching_moment)

                    # Save data to CSV with a unique filename for each combination
                    file_name = f"Data/Data_Instance{n}.csv"
                    Mk4SaveDataToCSV(i, j, k, lift, pitching_moment, induced_drag, file_name,0.3, 1.4, 
                                        iterate_mw_tip_chords, 0.45, 0.2 , 0.01, 0.4)
                    n += 1


                for iterate_tail_bpositions in tail_bpositions:
                    lift, induced_drag, pitching_moment = simulation(va= j, aoa = k, fp = i, mw_airfoil = "naca8304", mw_root_chord = 0.3, mw_wingspan = iterate_mw_wingspans,
                                                                    mw_tip_chord = 0.2, tail_bposition = iterate_tail_bpositions, tail_type = "V-Tail", tail_root_chord =0.2 , 
                                                                    tail_tip_chord = 0.01, tail_wingspan = 0.4, tail_airfoil = "naca0004")
                    # Flatten the arrays
                    lift = lift.flatten()
                    induced_drag = induced_drag.flatten()
                    pitching_moment = pitching_moment.flatten()
                    print("Before Second Cycle Extraction :  ", lift.shape, induced_drag.shape, pitching_moment.shape)
                    print("Before Second Cycle Extraction value:  ", lift, induced_drag, pitching_moment)

                    # Extract the second cycle from the data
                    lift, induced_drag, pitching_moment = extract_second_cycle(lift, induced_drag, pitching_moment)
                    print("After Second Cycle Extraction :  ",lift.shape, induced_drag.shape, pitching_moment.shape)
                    print("After Second Cycle Extraction value:  ", lift, induced_drag, pitching_moment)

                    # Save data to CSV with a unique filename for each combination
                    file_name = f"Data/Data_Instance{n}.csv"
                    Mk4SaveDataToCSV(i, j, k, lift, pitching_moment, induced_drag, file_name,0.3, 1.4, 
                                        0.2, iterate_tail_bpositions, 0.2 , 0.01, 0.4)
                    n += 1

                
                for iterate_tail_root_chords in tail_root_chords:
                    lift, induced_drag, pitching_moment = simulation(va= j, aoa = k, fp = i, mw_airfoil = "naca8304", mw_root_chord = 0.3, mw_wingspan = iterate_mw_wingspans,
                                                                    mw_tip_chord = 0.2, tail_bposition = 0.45, tail_type = "V-Tail", tail_root_chord =iterate_tail_root_chords , 
                                                                    tail_tip_chord = 0.01, tail_wingspan = 0.4, tail_airfoil = "naca0004")
                    # Flatten the arrays
                    lift = lift.flatten()
                    induced_drag = induced_drag.flatten()
                    pitching_moment = pitching_moment.flatten()
                    print("Before Second Cycle Extraction :  ", lift.shape, induced_drag.shape, pitching_moment.shape)
                    print("Before Second Cycle Extraction value:  ", lift, induced_drag, pitching_moment)

                    # Extract the second cycle from the data
                    lift, induced_drag, pitching_moment = extract_second_cycle(lift, induced_drag, pitching_moment)
                    print("After Second Cycle Extraction :  ",lift.shape, induced_drag.shape, pitching_moment.shape)
                    print("After Second Cycle Extraction value:  ", lift, induced_drag, pitching_moment)

                    # Save data to CSV with a unique filename for each combination
                    file_name = f"Data/Data_Instance{n}.csv"
                    Mk4SaveDataToCSV(i, j, k, lift, pitching_moment, induced_drag, file_name,0.3, 1.4, 
                                        0.2, 0.45, iterate_tail_root_chords, 0.01, 0.4)
                    n += 1



                for iterate_tail_tip_chords in tail_tip_chords:
                    lift, induced_drag, pitching_moment = simulation(va= j, aoa = k, fp = i, mw_airfoil = "naca8304", mw_root_chord = 0.3, mw_wingspan = iterate_mw_wingspans,
                                                                    mw_tip_chord = 0.2, tail_bposition = 0.45, tail_type = "V-Tail", tail_root_chord =0.2 , 
                                                                    tail_tip_chord = iterate_tail_tip_chords, tail_wingspan = 0.4, tail_airfoil = "naca0004")
                    # Flatten the arrays
                    lift = lift.flatten()
                    induced_drag = induced_drag.flatten()
                    pitching_moment = pitching_moment.flatten()
                    print("Before Second Cycle Extraction :  ", lift.shape, induced_drag.shape, pitching_moment.shape)
                    print("Before Second Cycle Extraction value:  ", lift, induced_drag, pitching_moment)

                    # Extract the second cycle from the data
                    lift, induced_drag, pitching_moment = extract_second_cycle(lift, induced_drag, pitching_moment)
                    print("After Second Cycle Extraction :  ",lift.shape, induced_drag.shape, pitching_moment.shape)
                    print("After Second Cycle Extraction value:  ", lift, induced_drag, pitching_moment)

                    # Save data to CSV with a unique filename for each combination
                    file_name = f"Data/Data_Instance{n}.csv"
                    Mk4SaveDataToCSV(i, j, k, lift, pitching_moment, induced_drag, file_name,0.3, 1.4, 
                                        0.2, 0.45, 0.2, iterate_tail_tip_chords, 0.4)
                    n += 1

                
                for iterate_tail_wingspans in tail_wingspans:
                    lift, induced_drag, pitching_moment = simulation(va= j, aoa = k, fp = i, mw_airfoil = "naca8304", mw_root_chord = 0.3, mw_wingspan = iterate_mw_wingspans,
                                                                    mw_tip_chord = 0.2, tail_bposition = 0.45, tail_type = "V-Tail", tail_root_chord =0.2 , 
                                                                    tail_tip_chord = 0.01, tail_wingspan = iterate_tail_wingspans, tail_airfoil = "naca0004")
                    # Flatten the arrays
                    lift = lift.flatten()
                    induced_drag = induced_drag.flatten()
                    pitching_moment = pitching_moment.flatten()
                    print("Before Second Cycle Extraction :  ", lift.shape, induced_drag.shape, pitching_moment.shape)
                    print("Before Second Cycle Extraction value:  ", lift, induced_drag, pitching_moment)

                    # Extract the second cycle from the data
                    lift, induced_drag, pitching_moment = extract_second_cycle(lift, induced_drag, pitching_moment)
                    print("After Second Cycle Extraction :  ",lift.shape, induced_drag.shape, pitching_moment.shape)
                    print("After Second Cycle Extraction value:  ", lift, induced_drag, pitching_moment)

                    # Save data to CSV with a unique filename for each combination
                    file_name = f"Data/Data_Instance{n}.csv"
                    Mk4SaveDataToCSV(i, j, k, lift, pitching_moment, induced_drag, file_name,0.3, 1.4, 
                                        0.2, 0.45, 0.2, 0.01, iterate_tail_wingspans)
                    n += 1
                

    print("Data Collection is Over")



mw_root_chords_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mw_wingspans_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
mw_tip_chords_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
tail_bpositions_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
tail_root_chords_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
tail_tip_chords_list = [0.001, 0.051, 0.101, 0.151, 0.201]
tail_wingspans_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Example call with lists for iteration
Mark4Simulation(FlappingPeriods = [0.5], Angles_of_Attacks = [5], Air_Speeds = [5], mw_root_chords = mw_root_chords_list, mw_wingspans = mw_wingspans_list, mw_tip_chords = mw_tip_chords_list, 
                    tail_bpositions = tail_bpositions_list, tail_root_chords = tail_root_chords_list, tail_tip_chords = tail_tip_chords_list, tail_wingspans = tail_wingspans_list)
