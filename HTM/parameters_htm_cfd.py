"""
Parameters for CFD
"""

u_in = 1.0  # inlet velocity - (.1, .2, .3, .4, .5, .6, .7, .8, .9)

# SPATIAL AND TEMPORAL INPUTS
length = 4  # Length of computational domain in the x-direction
breadth = 4  # Breadth of computational domain in the y-direction
colPoints = 129  # Number of grid points in the x-direction #KEEP ODD
rowPoints = 129  # Number of grid points in the y-direction #KEEP ODD

# FLUID PROPERTIES
rho = 1  # Density of fluid
mu = 0.01  # Viscosity of fluid

# BOUNDARY SPECIFICATIONS
uin = 1  # Lid velocity
v_wall = 0  # Velocity of fluid at the walls
p_out = 0  # Gauge pressure at the boundaries

# SIMULATION PARAMETERS
time = 150  # Simulation time
CFL_number = 0.8  # Reduce this if solution diverges
file_flag = 1  # Keep 1 to print results to file
interval = 100  # Record values in file per interval number of iterations

"""
Parameters for HTM
"""
columnCount = 1024
htm_parameters = {

    # region ENCODER PARAMETERS
    'enc': {
        "p": {'minimum': 0,
              'maximum': 2,
              'size': 1000,
              'sparsity': 0.02
              },
        "u": {'minimum': 0,
              'maximum': 8,
              'size': 1000,
              'sparsity': 0.02
              },
        "v": {'minimum': 0,
              'maximum': 8,
              'size': 1000,
              'sparsity': 0.02
              },
        "uin": {'minimum': 0,
                'maximum': 10,
                'size': 1000,
                'sparsity': 0.02
                },
        "rho": {'minimum': 0,
                'maximum': 10,
                'size': 1000,
                'sparsity': 0.02
                },
        "mu": {'minimum': 0,
               'maximum': 1,
               'size': 1000,
               'sparsity': 0.02
               },
        "x": {'minimum': 0,
              'maximum': colPoints,
              'size': 500,
              'sparsity': 0.03
              },
        "y": {'minimum': 0,
              'maximum': rowPoints,
              'size': 500,
              'sparsity': 0.03
              },
    },
    # endregion

    # region SP, TM, PREDICTOR PARAMETERS
    'sp_l4': {'boostStrength': 3.0,
              'columnCount': columnCount,
              'localAreaDensity': 0.02,
              'potentialPct': 0.85,
              'synPermActiveInc': 0.04,
              'synPermConnected': 0.14,
              'synPermInactiveDec': 0.006
              },
    'sp_l23': {'boostStrength': 3.0,
               'columnCount': columnCount,
               'localAreaDensity': 0.02,
               'potentialPct': 0.85,
               'synPermActiveInc': 0.04,
               'synPermConnected': 0.14,
               'synPermInactiveDec': 0.006
               },
    'tm_l4': {'activationThreshold': 7,
              'columnCount': columnCount,
              'cellsPerColumn': 50,  # reduce to 50
              'initialPerm': 0.21,
              'synPermConnected': 0.14,
              'maxSegmentsPerCell': 64,
              'maxSynapsesPerSegment': 32,
              'minThreshold': 5,
              'newSynapseCount': 16,
              'permanenceDec': 0.1,
              'permanenceInc': 0.1,
              'externalPredictiveInputs': columnCount * 50},
    'tm_l23': {'activationThreshold': 7,
               'columnCount': columnCount,
               'cellsPerColumn': 50,  # reduce to 50
               'initialPerm': 0.21,
               'synPermConnected': 0.14,
               'maxSegmentsPerCell': 64,
               'maxSynapsesPerSegment': 32,
               'minThreshold': 5,
               'newSynapseCount': 16,
               'permanenceDec': 0.1,
               'permanenceInc': 0.1,
               'externalPredictiveInputs': columnCount * 50},
    'predictor': {'sdrc_alpha': 0.5}

    # endregion
}
