#Self defined functions for the unsteady ring vortex lattice method solver

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd

def extract_forces(solver):
    """
    Extracts lift, induced drag, side forces, and pitching moments from the solver for all time steps.

    :param solver: UnsteadyRingVortexLatticeMethodSolver
        The solver object containing the results.

    :return: tuple of four numpy arrays
        lift, induced_drag, side_force, pitching_moment
    """
    num_steps = solver.num_steps
    first_results_step = solver.first_results_step
    num_steps_to_average = num_steps - first_results_step
    num_airplanes = solver.num_airplanes

    # Initialize arrays to store forces and moments
    lift = np.zeros((num_airplanes, num_steps_to_average))
    induced_drag = np.zeros((num_airplanes, num_steps_to_average))
    side_force = np.zeros((num_airplanes, num_steps_to_average))
    pitching_moment = np.zeros((num_airplanes, num_steps_to_average))

    # Iterate through time steps and extract forces and moments
    for step in range(first_results_step, num_steps):
        results_step = step - first_results_step
        airplanes = solver.steady_problems[step].airplanes

        for airplane_id, airplane in enumerate(airplanes):
            forces = airplane.total_near_field_force_wind_axes
            moments = airplane.total_near_field_moment_wind_axes

            # Extract forces
            induced_drag[airplane_id, results_step] = forces[0]
            side_force[airplane_id, results_step] = forces[1]
            lift[airplane_id, results_step] = forces[2]

            # Extract pitching moment
            pitching_moment[airplane_id, results_step] = moments[1]  # Y-axis is pitching moment

    return lift, induced_drag, side_force, pitching_moment



def plot_forces(lift, drag, side_force, 
                         lift_style='-', drag_style='-', side_force_style='-', 
                         lift_color='blue', drag_color='red', side_force_color='green'):
    
    lift = lift.flatten()
    drag = drag.flatten()
    side_force = side_force.flatten()
    
    # Check if all input arrays have the same length
    if not (len(lift) == len(drag) == len(side_force)):
        raise ValueError("All input arrays must have the same length.")

    # Create an array for the x-axis (assuming it's the same length as the input arrays)
    x_values = np.arange(len(lift))

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot Lift
    axs[0].plot(x_values, lift, linestyle=lift_style, color=lift_color, linewidth=2)
    axs[0].set_title('Lift')
    axs[0].set_xlabel('Sample Index')
    axs[0].set_ylabel('Lift Values')
    axs[0].grid(True)

    # Plot Drag
    axs[1].plot(x_values, drag, linestyle=drag_style, color=drag_color, linewidth=2)
    axs[1].set_title('Drag')
    axs[1].set_xlabel('Sample Index')
    axs[1].set_ylabel('Drag Values')
    axs[1].grid(True)

    # Plot Side Force
    axs[2].plot(x_values, side_force, linestyle=side_force_style, color=side_force_color, linewidth=2)
    axs[2].set_title('Side Force')
    axs[2].set_xlabel('Sample Index')
    axs[2].set_ylabel('Side Force Values')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()



def save_flapping_data_to_csv(flapping_freqs, aoas, Vas, angles_of_flapping, lifts, pitching_moments, induced_drags, filename="flapping_data.csv"):
    """
    Save the flapping data into a CSV file with columns for Flapping Frequency, Angle of Attack, Va, 
    Angle of Flapping, Lift, Pitching Moment, and Induced Drag.

    :param flapping_freqs: List of flapping frequencies
    :param aoas: List of angles of attack
    :param Vas: List of Vas
    :param angles_of_flapping: Numpy array of angles of flapping (2D, one per time step and configuration)
    :param lifts: Numpy array of lift values (2D, one per time step and configuration)
    :param pitching_moments: Numpy array of pitching moments (2D, one per time step and configuration)
    :param induced_drags: Numpy array of induced drag values (2D, one per time step and configuration)
    :param filename: The name of the CSV file to save the data in
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(["Flapping Frequency", "Angle of Attack", "Va", "Angle of Flapping", "Lift", "Pitching Moment", "Induced Drag"])

        # Loop through the combinations of inputs and corresponding data arrays
        for i, freq in enumerate(flapping_freqs):
            for j, aoa in enumerate(aoas):
                for k, Va in enumerate(Vas):
                    for t in range(len(angles_of_flapping[i][j][k])):
                        writer.writerow([freq, aoa, Va, angles_of_flapping[i][j][k][t], lifts[i][j][k][t], pitching_moments[i][j][k][t], induced_drags[i][j][k][t]])

    print(f"Data successfully saved to {filename}")








# def extract_second_cycle(data1, data2, data3, flapping_period):
#     total_points = len(data1)  # Get the number of points from the data
#     points_per_second = 150  # Updated to 150 points per second
#     points_per_cycle = int(points_per_second * flapping_period)
#     num_cycles = total_points // points_per_cycle

#     if num_cycles != 3:
#         print(f"Warning: Expected 3 cycles, but found {num_cycles} cycles.")

#     # Determine the start and end index for the second cycle
#     start_index = points_per_cycle
#     end_index = 2 * points_per_cycle

#     # Extract data for the second cycle
#     data1_second_cycle = data1[start_index:end_index]
#     data2_second_cycle = data2[start_index:end_index]
#     data3_second_cycle = data3[start_index:end_index]

#     # If data lengths are less than expected, pad with zeros
#     if len(data1_second_cycle) < points_per_cycle:
#         padding_length = points_per_cycle - len(data1_second_cycle)
#         data1_second_cycle = np.pad(data1_second_cycle, (0, padding_length), 'constant')

#     if len(data2_second_cycle) < points_per_cycle:
#         padding_length = points_per_cycle - len(data2_second_cycle)
#         data2_second_cycle = np.pad(data2_second_cycle, (0, padding_length), 'constant')

#     if len(data3_second_cycle) < points_per_cycle:
#         padding_length = points_per_cycle - len(data3_second_cycle)
#         data3_second_cycle = np.pad(data3_second_cycle, (0, padding_length), 'constant')

#     return data1_second_cycle, data2_second_cycle, data3_second_cycle

import numpy as np

def extract_second_cycle(data1, data2, data3):
    # Check if inputs are valid
    if not (isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray) and isinstance(data3, np.ndarray)):
        raise TypeError("Input data must be NumPy arrays")
    
    if not (data1.size == data2.size == data3.size):
        raise ValueError("All input arrays must have the same size")

    # Compute points per cycle based on the length of the input arrays
    total_points = data1.size  # Use .size for NumPy arrays
    points_per_cycle = total_points // 3  # Divide by 3 to get points per cycle

    # The second cycle starts at points_per_cycle and ends at 2 * points_per_cycle
    start_index = points_per_cycle
    end_index = 2 * points_per_cycle

    # Check if indices are within bounds
    if end_index > total_points:
        raise IndexError("End index exceeds the length of the input arrays.")

    # Slice the arrays to extract only the second cycle
    data1_second_cycle = data1[start_index:end_index]
    data2_second_cycle = data2[start_index:end_index]
    data3_second_cycle = data3[start_index:end_index]

    return data1_second_cycle, data2_second_cycle, data3_second_cycle


def save_flight_data_to_csv(FlappingPeriod, Va, AoA, Lift, PitchingMoment, InducedDrag, file_name):
    # Ensure that Lift, PitchingMoment, and InducedDrag are numpy arrays
    Lift = np.asarray(Lift)
    PitchingMoment = np.asarray(PitchingMoment)
    InducedDrag = np.asarray(InducedDrag)

    # Total number of points for one flapping period
    num_points = Lift.size
    
    # Ensure that all arrays have the same length
    if not all(arr.size == num_points for arr in [PitchingMoment, InducedDrag]):
        raise ValueError(f"Lift, PitchingMoment, and InducedDrag must all have the same length of {num_points} points.")
    
    # Create a normalized time array (range between 0 and 1)
    normalized_time = np.linspace(0, 1, num_points, endpoint=False)
    
    # Create the 'data' directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Construct the full file path
    file_path = os.path.join('data', file_name)
    
    # Open the CSV file to write the data
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the data for each time step (excluding column names)
        for i in range(num_points):
            writer.writerow([FlappingPeriod, Va, AoA, normalized_time[i], Lift[i], PitchingMoment[i], InducedDrag[i]])

    print(f"Data has been successfully saved to {file_path}")


# Example usage:
# save_flight_data_to_csv(FlappingPeriod, Va, AoA, Lift, PitchingMoment, InducedDrag, 'flight_data.csv')





def stack_csv_vertically(directory, output_file):
    # Initialize an empty list to store dataframes
    dataframes = []
    
    # Loop through files and read them into dataframes
    for i in range(len(os.listdir(directory))):  # assuming the CSV files are sequentially named
        file_path = os.path.join(directory, f'flapping{i}.csv')
        if os.path.isfile(file_path):
            if i == 0:
                # For the first file, read normally (including headers)
                df = pd.read_csv(file_path)
                columns = df.columns  # Save the column names
            else:
                # For subsequent files, skip the header
                df = pd.read_csv(file_path, header=None)
                df.columns = columns  # Assign the column names from the first file
            dataframes.append(df)
    
    # Concatenate all dataframes vertically (axis=0)
    result_df = pd.concat(dataframes, axis=0, ignore_index=True)
    
    # Save the result to the output CSV file
    result_df.to_csv(output_file, index=False, header=True)
    
    print(f"Stacked CSV saved to {output_file}")

# Example usage:
# stack_csv_vertically('/path/to/your/csv/files', 'stacked_output.csv')



# Example usage:
# stack_csv_horizontally('/path/to/your/csv/files', 'stacked_output.csv')


#_______________________________________________________________

# Tail Part

#____________________________________________________________________

