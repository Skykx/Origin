import sys
from flowPy import *
from parameters_htm_cfd import rowPoints, colPoints, breadth, length, rho, mu, v_wall, p_out, time, file_flag, CFL_number, \
    interval


def runCFD(u_in, dir_path):
    # Create an object of the class Space called cavity
    cavity = Space()
    cavity.createMesh(rowPoints, colPoints)
    cavity.setDeltas(breadth, length)

    # Create an object of the class Fluid called water
    water = Fluid(rho, mu)

    # Create objects of the class Boundary having either Dirichlet ("D") or Neumann ("N") type boundaries
    flow = Boundary("D", u_in)
    noSlip = Boundary("D", v_wall)
    zeroFlux = Boundary("N", 0)
    pressureAtm = Boundary("D", p_out)

    # RUN SIMULATION
    # Print general simulation information
    print("######## Beginning FlowPy Simulation ########")
    print("#############################################")
    print("# Simulation time: {0:.2f}".format(time))
    print("# Mesh: {0} x {1}".format(colPoints, rowPoints))
    print("# Re/u: {0:.2f}\tRe/v:{1:.2f}".format(rho * length / mu, rho * breadth / mu))
    print("# Save outputs to text file: {0}".format(bool(file_flag)))

    # Initialization
    makeResultDirectory(wipe=False, u_in=u_in)
    # Initialize counters
    t = 0
    i = 0

    # Run
    while t < time:
        # Print time left
        sys.stdout.write("\rSimulation time left: {0:.2f}".format(time - t))
        sys.stdout.flush()
        # Set the time-step
        setTimeStep(CFL_number, cavity, water)
        timestep = cavity.dt

        # Set boundary conditions
        setUBoundary(cavity, noSlip, noSlip, flow, noSlip)
        setVBoundary(cavity, noSlip, noSlip, noSlip, noSlip)
        setPBoundary(cavity, zeroFlux, zeroFlux, pressureAtm, zeroFlux)

        # Calculate starred velocities
        getStarredVelocities(cavity, water)

        # Solve the pressure Poisson equation
        solvePressurePoisson(cavity, water, zeroFlux, zeroFlux,
                             pressureAtm, zeroFlux)
        # Solve the momentum equation
        solveMomentumEquation(cavity, water)
        # Save variables and write to file
        setCentrePUV(cavity)
        if file_flag == 1:
            writeToFile(cavity, i, interval, dir_path)
        # Advance time-step and counter
        t += timestep
        i += 1
