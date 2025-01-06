import pterasoftware as ps
import numpy as np
import matplotlib.pyplot as plt
from SelfFunctions import extract_forces, extract_second_cycle
import os 
import csv


def Mk4SaveDataToCSV(FlappingPeriod, Va, AoA, Lift, PitchingMoment, InducedDrag, file_name, 
     mw_root_chord, mw_wingspan, mw_tip_chord, tail_bposition, tail_root_chord, tail_tip_chord, tail_wingspan):
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
    
    # Write the CSV file in the current directory
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # # Write the header (column names)
        # writer.writerow(['Flapping Frequency', 'AirSpeed', 'Angle of Attack', 'Normalised Time', 'Lift', 'Induced Drag', 'Pitching Moment', 
        #                  'MW Root Airfoil', 'MW Root Chord', 'MW Wingspan', 'MW Tip Chord', 'Tail BPosition', 'Tail Type', 'Tail Root Chord', 
        #                  'Tail Tip Chord', 'Tail Wingspan', 'Tail Airfoil'])
        
        # Write the data for each time step
        for i in range(num_points):
            writer.writerow([FlappingPeriod, Va, AoA, normalized_time[i], Lift[i], InducedDrag[i], PitchingMoment[i], 
                             mw_root_chord, mw_wingspan, mw_tip_chord, tail_bposition, tail_root_chord, 
                             tail_tip_chord, tail_wingspan])

    print(f"Data has been successfully saved to {file_name}")



    
# Function call for the unsteady aerodynamic model with the NACA 8304 airfoil
def simulation(va= 5, aoa = 5, fp = 0.2, mw_airfoil = "naca8304", mw_root_chord = 0.3, mw_wingspan = 1.4, mw_tip_chord = 0.2, tail_bposition = 0.45, tail_type = "V-Tail", tail_root_chord =0.2 , tail_tip_chord = 0.01, tail_wingspan = 0.4, tail_airfoil = "naca0004" ):
    example_airplane=ps.geometry.Airplane(
        name="naca8304",
        x_ref=0.11,
        y_ref=0.0,
        z_ref=0.0,
        s_ref=None,
        b_ref=None,
        c_ref=None,
        wings=[
            ps.geometry.Wing(
                name="Main Wing",
                x_le=0.0,
                y_le=0.0,
                z_le=0.0,
                symmetric=True,
                num_chordwise_panels=6,
                chordwise_spacing="cosine",
                wing_cross_sections=[
                    ps.geometry.WingCrossSection(
                        x_le=0.0,
                        y_le=0.0,
                        z_le=0.0,
                        twist=0.0,
                        control_surface_type="symmetric",
                        control_surface_hinge_point=0.0,
                        control_surface_deflection=0.0,
                        num_spanwise_panels=8,
                        spanwise_spacing="cosine",
                        chord=mw_root_chord,
                        airfoil=ps.geometry.Airfoil(
                            name= mw_airfoil,
                            coordinates=None,
                            repanel=True,
                            n_points_per_side=400,
                        ),
                    ),
                    ps.geometry.WingCrossSection(
                        x_le=0.0,
                        y_le=mw_wingspan/2,
                        z_le=0.0,
                        chord=mw_tip_chord,
                        twist=0.0,
                        airfoil=ps.geometry.Airfoil(
                            name="naca8304",
                        ),
                    ),
                ],
            ),
            ps.geometry.Wing(
                name=tail_type,
                x_le=tail_bposition,
                y_le=0.0,
                z_le=0.0,
                num_chordwise_panels=6,
                chordwise_spacing="cosine",
                symmetric=True,
                wing_cross_sections=[
                    ps.geometry.WingCrossSection(
                        chord=tail_root_chord,
                        control_surface_type="symmetric",
                        control_surface_hinge_point=0.1,
                        control_surface_deflection=0.0,
                        airfoil=ps.geometry.Airfoil(
                            name=tail_airfoil,
                        ),
                        twist=0.0,
                    ),
                    ps.geometry.WingCrossSection(
                        x_le=0.19,
                        y_le=tail_wingspan/2,
                        z_le=0.003,
                        chord=tail_root_chord,
                        control_surface_type="symmetric",
                        control_surface_hinge_point=0.1,
                        control_surface_deflection=0.0,
                        twist=0.0,
                        airfoil=ps.geometry.Airfoil(
                            name=tail_airfoil,
                        ),
                    ),
                ],
            ),
        ],
    )
    main_wing_root_wing_cross_section_movement=ps.movement.WingCrossSectionMovement(
        base_wing_cross_section=example_airplane.wings[0].wing_cross_sections[0],
    )
    main_wing_tip_wing_cross_section_movement=ps.movement.WingCrossSectionMovement(
        base_wing_cross_section=example_airplane.wings[0].wing_cross_sections[1],
        sweeping_amplitude=30.0,
        #____________________________________________________________________
        sweeping_period=fp,
        #____________________________________________________________________
        sweeping_spacing="sine",
        pitching_amplitude=0.0,
        pitching_period=0.0,
        pitching_spacing="sine",
        heaving_amplitude=0.0,
        heaving_period=0.0,
        heaving_spacing="sine",
    )
    v_tail_root_wing_cross_section_movement=ps.movement.WingCrossSectionMovement(
        base_wing_cross_section=example_airplane.wings[1].wing_cross_sections[0],
    )
    v_tail_tip_wing_cross_section_movement=ps.movement.WingCrossSectionMovement(
        base_wing_cross_section=example_airplane.wings[1].wing_cross_sections[1],
    )
    main_wing_movement=ps.movement.WingMovement(
        base_wing=example_airplane.wings[0],
        wing_cross_sections_movements=[
            main_wing_root_wing_cross_section_movement,
            main_wing_tip_wing_cross_section_movement,
        ],
    )
    del main_wing_root_wing_cross_section_movement
    del main_wing_tip_wing_cross_section_movement
    v_tail_movement=ps.movement.WingMovement(
        base_wing=example_airplane.wings[1],
        wing_cross_sections_movements=[
            v_tail_root_wing_cross_section_movement,
            v_tail_tip_wing_cross_section_movement,
        ],
    )
    del v_tail_root_wing_cross_section_movement
    del v_tail_tip_wing_cross_section_movement
    airplane_movement=ps.movement.AirplaneMovement(
        base_airplane=example_airplane,
        wing_movements=[main_wing_movement,v_tail_movement],
    )
    del main_wing_movement
    del v_tail_movement
    example_operating_point=ps.operating_point.OperatingPoint(

        #____________________________________________________________________
        density=1.225,
        beta=0.0,
        velocity=va,
        alpha=aoa,
        nu=15.06e-6,
        external_thrust=0.0,
        #____________________________________________________________________

    )
    operating_point_movement=ps.movement.OperatingPointMovement(
        base_operating_point=example_operating_point,
        velocity_amplitude=0.0,
        velocity_period=0.0,
        velocity_spacing="sine",
    )
    movement=ps.movement.Movement(
        airplane_movements=[airplane_movement],
        operating_point_movement=operating_point_movement,
        num_steps=None,
        delta_time=None,
    )
    del airplane_movement
    del operating_point_movement
    example_problem=ps.problems.UnsteadyProblem(
        movement=movement,
    )
    example_solver=ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
        unsteady_problem=example_problem,
    )
    del example_problem
    example_solver.run(
        logging_level="Warning",
        prescribed_wake=True,
    )

    lift, induced_drag, side_force, pitching_moment = extract_forces(example_solver)  

    return lift, induced_drag, pitching_moment









