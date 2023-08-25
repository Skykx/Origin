import os.path
import math
import time
import gc

import numpy as np
from matplotlib import pyplot as plt
from flowPy import *
from htm.bindings.encoders import *
from htm.bindings.algorithms import SpatialPooler, TemporalMemory, Predictor
from htm.bindings.sdr import SDR
from parameters_htm_cfd import htm_parameters, u_in

seed = 0
predictor_res = 0.001


def train(coordinates_dict, num_cycles, sensory_type, predictor_on, learning, parameters=None):
    global sp_l4, tm_l4, sp_l23, tm_l23, predictor
    np.set_printoptions(threshold=np.inf)

    # region Preparation of the simulation data
    Ps, Us, Vs = getData(os.path.join(os.getcwd(), "result\\u_in_{}".format(u_in)))  # get every single cfd file data

    """  
    HTM has problems working with negative numbers, so here all negative numbers from the simulation are converted to 
    positive numbers. For example, if Ps=[-2,1,3,-5,7], then the smallest number is taken and subtracted from all 
    numbers (here -5), so that Ps=[3,6,8,0,12]. This must be subtracted again later when interpreting the data.
    """

    if np.amin(Ps) < 0:
        Ps = Ps - np.amin(Ps)
    if np.amin(Us) < 0:
        Us = Us - np.amin(Us)
    if np.amin(Vs) < 0:
        Vs = Vs - np.amin(Vs)

    data_dict = {"p": Ps, "u": Us, "v": Vs}
    data = data_dict[sensory_type]  # Create a new variable for efficient access to the cfd data

    '''  
    The idea now is to create dictionaries for each column and it's number of sensors. This way you can
    easily access each sensor's data from the cfd simulation.  
    First Loop: get access to the coordinates through the key-parameter of the dictionaries and create a variable that 
    containing all the coordinates.
    Second Loop: get access to the x and y coordinates. senor_value_List=[] is required to separate each sensor data in
    the dictionary.
    Third Loop: get access to every single data from the cfd simulation that is stored in "data". Read all data for the
    coordinate for the specified column_, create a dictionary for each column (sensor_dict_) and save the sensor_x_y
    as a key-parameter with the simulation data at the given coordinate (sensor_value_list).  
    '''

    # Initialize sensor data dictionaries for each column
    sensor_data_dict = {'column_1': {}, 'column_2': {}, 'column_3': {}, 'column_4': {}, 'column_5': {}, 'column_6': {},
                        'column_7': {}, 'column_8': {}, 'column_9': {}}

    # Loop over the columns and generate sensor data for each column
    for column_num in coordinates_dict:
        column_coordinates = coordinates_dict[column_num]

        for x, y in column_coordinates:
            sensor_value_list = []
            for sensoryData in data:
                sensor_value_list.append(sensoryData[x][y])

            sensor_data_dict[column_num][f"sensor_{x}_{y}"] = sensor_value_list

    del x, y, Ps, Us, Vs, data  # clear the memory
    gc.collect()  # Remove Ps, Us, Vs, data completely from memory (just to be sure)

    # endregion

    # region Preparation for the Encoder

    # create Encoder with Parameters and set it up for use
    enc_sensory = create_encoder(parameters["enc"], sensory_type)
    enc_x = create_encoder(parameters["enc"], "x")
    enc_y = create_encoder(parameters["enc"], "y")
    encoding_width = enc_sensory.size + enc_x.size + enc_y.size

    # endregion

    # region

    '''
    Try to load or initialize Spatial Pooler, Temporal Memory and Predictor for each Sensor via function. 

    Use "predictions = {}" dictionary to safe all the predictions for each column with its sensors.  
    sdr_dict = {} will be used to save the SDR output of the Temporal Memory computation. This way we can use (and 
    combine) different SDRs as an external input for the Temporal Memory computation.

    We run the loops once for all columns, their sensors and all of their cfd data. This is because we can then
    use the initialised data for lateral communication between different columns. 
    '''

    predictions_dict = {}
    sdr_active_cells_dict = {}
    sdr_winner_cells_dict = {}

    sdr_active_cells_temporal_dict = {}
    sdr_winner_cells_temporal_dict = {}

    accuracy_dict = {}
    accuracy_samples_dict = {}
    accuracy_total_dict = {}

    runtime = []
    for cycle in range(num_cycles + 1):
        for column_num in coordinates_dict:
            if column_num == 'column_5':
                start = time.time()
            column_coordinates = coordinates_dict[column_num]
            for x, y in column_coordinates:

                '''
                Prepare the dictionary "predictions = {}" with a key parameter + another dictionary. The key parameter 
                is the column number and the value is another dictionary. That dictionaries key parameter is 
                the sensor_x_y coordinate of the column and the value is an empty list. The result will be like this:
                {'column_1': {'sensor_65_65': []}, 'column_2': {'sensor_64_65': []}, 'column_3'....}. This gives us easy 
                access to any prediction data from each sensor later on. 
                '''
                # Prüfe, ob der gegebene Column in der Vorhersage-Datenstruktur vorhanden ist
                # Prüfe, ob die aktuelle Sensorposition in der Vorhersage-Datenstruktur vorhanden ist
                if column_num not in predictions_dict:
                    predictions_dict[column_num] = {}
                predictions_dict[column_num][f'sensor_{x}_{y}'] = []

                # Get the current cfd data from the dictionary at coordinate x,y and safe it in current_sensor_data
                # Hole die aktuellen Sensordaten an der gegebenen Position aus der Sensor-Datenstruktur
                current_sensor_data = sensor_data_dict.get(column_num, {}).get(f'sensor_{x}_{y}', [])
                print(column_num, f'sensor_{x}_{y}', current_sensor_data)

                # Encode x and y Coordinates
                x_bits = enc_x.encode(x)
                y_bits = enc_y.encode(y)

                if cycle == 0 or learning == False:
                    print("#### Lateral communication off #### for: ", sensory_type, "#####")
                    print("#### Cyle Nr.", cycle, " #####")

                    # Try to load or initialize Spatial Pooler, Temporal Memory and Predictor via function
                    sp_l4, tm_l4, predictor = initialize_sensory_layer(sensory_type, column_num, x, y, parameters,
                                                                       encoding_width, seed, predictor_on)

                    # Create SDR object with dimension of the SP. The initial value is all zeros.
                    active_columns_l4 = SDR(sp_l4.getColumnDimensions())

                    for record_num, sensory_input in enumerate(current_sensor_data):
                        #  encode data
                        sensory_bits = enc_sensory.encode(sensory_input)
                        encoding = SDR(encoding_width).concatenate([sensory_bits, x_bits, y_bits])  # = SDR(2000) 51200

                        # compute spatial pooler and temporal memory
                        sp_l4, tm_l4 = sensory_layer(column_num,
                                                     sdr_active_cells_temporal_dict,
                                                     sdr_winner_cells_temporal_dict,
                                                     sp_l4, tm_l4,
                                                     active_columns_l4,
                                                     encoding, x, y)

                        # Predict what will happen, and then train the predictor based on what just happened.
                        predictor, predictions_dict = prediction(tm_l4,
                                                                 predictions_dict,
                                                                 column_num, x, y,
                                                                 sensory_input,
                                                                 predictor, predictor_res,
                                                                 record_num)

                elif cycle > 0 and learning == True:
                    print("#### Lateral communication on #### for: ", sensory_type, "#####")
                    print("#### Cyle Nr.", cycle, " #####")

                    # Try to load or initialize Spatial Pooler, Temporal Memory and Predictor via function
                    sp_l4, tm_l4, predictor = initialize_sensory_layer(sensory_type, column_num, x, y, parameters,
                                                                       encoding_width, seed, predictor_on)

                    sp_l23, tm_l23 = initialize_object_layer(sensory_type, column_num, x, y, parameters, seed)

                    # Create SDR object with dimension of the SP. The initial value is all zeros.
                    active_columns_l4 = SDR(sp_l4.getColumnDimensions())
                    active_columns_l23 = SDR(sp_l23.getColumnDimensions())

                    for record_num, sensory_input in enumerate(current_sensor_data):
                        # encode data
                        sensory_bits = enc_sensory.encode(sensory_input)
                        encoding = SDR(encoding_width).concatenate([sensory_bits, x_bits, y_bits])

                        # compute spatial pooler and then temporal memory with lateral connection

                        sp_l4, tm_l4 = sensory_layer(column_num,
                                                     sdr_active_cells_temporal_dict,
                                                     sdr_winner_cells_temporal_dict,
                                                     sp_l4, tm_l4,
                                                     active_columns_l4,
                                                     encoding, x, y)

                        sp_l23, tm_l23, tm_l4 = object_layer(column_num,
                                                             sdr_active_cells_dict,
                                                             sdr_winner_cells_dict,
                                                             sdr_active_cells_temporal_dict,
                                                             sdr_winner_cells_temporal_dict,
                                                             sp_l23, tm_l23, tm_l4,
                                                             active_columns_l23, active_columns_l4)

                        predictor, predictions_dict = prediction(tm_l4,
                                                                 predictions_dict,
                                                                 column_num, x, y,
                                                                 sensory_input,
                                                                 predictor, predictor_res,
                                                                 record_num)

                # Save every file for sensory_layer (l4)
                sp_l4.saveToFile("state/sp_l4_{}_{}_{}_{}_save.temp".format(column_num, sensory_type, x, y))
                print("{}_{}_{}_{}_SP_l4 saved".format(column_num, sensory_type, x, y))
                tm_l4.saveToFile("state/tm_l4_{}_{}_{}_{}_save.temp".format(column_num, sensory_type, x, y))
                print("{}_{}_{}_{}_TM_l4 saved".format(column_num, sensory_type, x, y))

                if cycle > 0 and learning == True:
                    # Save every file for object_layer (l23)
                    sp_l23.saveToFile("state/sp_l23_{}_{}_{}_{}_save.temp".format(column_num, sensory_type, x, y))
                    print("{}_{}_{}_{}_SP_l23 saved".format(column_num, sensory_type, x, y))
                    tm_l23.saveToFile("state/tm_l23_{}_{}_{}_{}_save.temp".format(column_num, sensory_type, x, y))
                    print("{}_{}_{}_{}_TM_l23 saved".format(column_num, sensory_type, x, y))

                if predictor_on:
                    # save predictors
                    predictor.saveToFile("state/predictor/{}_{}_{}_{}_save.temp".format(column_num, sensory_type, x, y),
                                         fmt="BINARY")
                    print("{}_{}_{}_{}_predictor saved".format(column_num, sensory_type, x, y))

                # region Prediction and accuracy stuff
                # Shift the predictions so that they are aligned with the input they predict.
                for n_steps, pred_list in predictions_dict[column_num].items():
                    for i in range(1):
                        pred_list.insert(0, float('nan'))
                        pred_list.pop()

                # Calculate the predictive accuracy, Root-Mean-Squared
                if column_num not in accuracy_dict:
                    accuracy_dict[column_num] = {}
                accuracy_dict[column_num][f'sensor_{x}_{y}'] = 0
                if column_num not in accuracy_samples_dict:
                    accuracy_samples_dict[column_num] = {}
                accuracy_samples_dict[column_num][f'sensor_{x}_{y}'] = 0

                sensor_data_list = sensor_data_dict.get(column_num, [])
                for sensor, sensor_data in sensor_data_list.items():
                    sensor_predictions = predictions_dict[column_num][sensor]
                    for idx, inp in enumerate(sensor_data):
                        val = sensor_predictions[idx]
                        if not math.isnan(val):
                            try:
                                accuracy_dict[column_num][sensor] += (inp - val) ** 2
                                accuracy_samples_dict[column_num][sensor] += 1
                            except:
                                print(idx)
                                print(inp)

                var_1 = accuracy_dict[column_num][f"sensor_{x}_{y}"]
                var_2 = accuracy_samples_dict[column_num][f"sensor_{x}_{y}"]
                var_3 = (var_1 / var_2) ** .5

                if column_num not in accuracy_total_dict:
                    accuracy_total_dict[column_num] = {}
                if f"sensor_{x}_{y}" not in accuracy_total_dict[column_num]:
                    accuracy_total_dict[column_num][f"sensor_{x}_{y}"] = []
                accuracy_total_dict[column_num][f"sensor_{x}_{y}"].append(var_3)

                predictor.reset()
                # endregion
                if column_num == 'column_5':
                    end = time.time()
                    runtime.append(end - start)

                if cycle % 1 == 0:
                    plt.plot(sensor_data_list[f'sensor_{x}_{y}'], 'red', label='actual')
                    plt.plot(predictions_dict[column_num][f'sensor_{x}_{y}'], 'blue', label='predicted')
                    plt.title("Cycle Number {}".format(cycle))

                    sim_handle, = plt.plot([], [], color='red', label='actual')
                    pred_handle, = plt.plot([], [], color='blue', label='predicted')
                    plt.legend(handles=[sim_handle, pred_handle])

                    plt.xlabel("Timestep")
                    if sensory_type == "p":
                        plt.ylabel("pressure (p)")
                    if sensory_type == "u":
                        plt.ylabel("horizontal velocity (u)")
                    if sensory_type == "v":
                        plt.ylabel("vertical velocity (v)")
                    plt.savefig("plot/sim_pred_{}_{}_{}_{}_{}.svg".format(column_num, sensory_type, x, y, cycle))
                    plt.close()

                    # Plotting SDRs
                    sdr_dense = tm_l4.getActiveCells().flatten()
                    sdr_dense = sdr_dense.reshape((200, 256))
                    sdr_dense = sdr_dense.dense.transpose()
                    print("tm_l4: ", tm_l4.getActiveCells())
                    fig = plt.figure(figsize=(10, 10), facecolor='white')
                    ax = fig.add_subplot(1, 1, 1)
                    ax.imshow(sdr_dense.T, cmap='binary', aspect='auto',
                              extent=[0, sdr_dense.shape[1], 0, sdr_dense.shape[0]])
                    ax.tick_params(axis='both', labelsize=20)
                    plt.savefig("plot/SDR_l4_{}_{}_{}_{}_{}.svg".format(column_num, sensory_type, x, y, cycle))
                    plt.close()

                    if cycle > 0 and learning == True:
                        sdr_dense = tm_l23.getActiveCells().flatten()
                        sdr_dense = sdr_dense.reshape((200, 256))
                        sdr_dense = sdr_dense.dense.transpose()
                        print("tm_l23: ", tm_l23.getActiveCells())
                        fig = plt.figure(figsize=(10, 10), facecolor='white')
                        ax = fig.add_subplot(1, 1, 1)
                        ax.imshow(sdr_dense.T, cmap='binary', aspect='auto',
                                  extent=[0, sdr_dense.shape[1], 0, sdr_dense.shape[0]])
                        ax.tick_params(axis='both', labelsize=20)
                        plt.savefig("plot/SDR_l23_{}_{}_{}_{}_{}.svg".format(column_num, sensory_type, x, y, cycle))
                        plt.close()

        """
        merge the cells of all the columns and their temporal dictionary into the static dictionary, so that we can use 
        it in the next cycle for our lateral communication, without overwriting it with new active and new winning cells 
        of the current cycle.
        """
        sdr_active_cells_dict = sdr_active_cells_temporal_dict
        sdr_winner_cells_dict = sdr_winner_cells_temporal_dict


    print("accuracy_total_dict", accuracy_total_dict)
    for column_num in coordinates_dict:
        column_coordinates = coordinates_dict[column_num]
        for x, y in column_coordinates:
            # plot accuracy_total_dict
            plt.plot(accuracy_total_dict[column_num][f'sensor_{x}_{y}'], 'blue')
            plt.legend("{}_pred_1".format(sensory_type))
            plt.xlabel("Cycle Number")
            plt.ylabel("Predictive Error (RMS)")

            # set the y-ticks
            yticks = np.linspace(0, 0.14, 15)
            plt.yticks(yticks)

            # customize the plot
            if sensory_type == "p":
                plt.ylim(0, 0.10)
            if sensory_type == "u":
                plt.ylim(0, 0.07)
            if sensory_type == "v":
                plt.ylim(0, 0.12)
            plt.grid(True)

            plt.savefig("plot/{}_RMS_cycleNo_{}_{}_{}_{}.svg".format(column_num, sensory_type, x, y, num_cycles))
            plt.close()

            plt.plot(runtime, 'blue')
            plt.xlabel("Cycle Number")
            plt.ylabel("Runtime")
            plt.grid(True)
            plt.savefig("plot/{}_Runtime_cycleNo_{}_{}_{}_{}.svg".format(column_num, sensory_type, x, y, num_cycles))
            plt.close()


def initialize_sensory_layer(sensory_type, column_num, x, y, parameters, encoding_width, seed, predictor_on):
    sp_l4 = SpatialPooler()
    tm_l4 = TemporalMemory()
    predictor = None

    try:
        sp_l4.loadFromFile("state/sp_l4_{}_{}_{}_{}_save.temp".format(column_num, sensory_type, x, y))
        print("Spatial Pooler l4_{}_{}_{}_{} state loaded successfully".format(column_num, sensory_type, x, y))
        tm_l4.loadFromFile("state/tm_l4_{}_{}_{}_{}_save.temp".format(column_num, sensory_type, x, y))
        print("Temporal Memory l4_{}_{}_{}_{} state loaded successfully".format(column_num, sensory_type, x, y))
    except RuntimeError:
        print("initialize l4_{}_{}_{}_{} Spatial Pooler and Temporal Memory".format(column_num, sensory_type, x, y))
        sp_l4 = initSpatialPooler_l4(params=parameters["sp_l4"], encoding_width=encoding_width, seed=seed)
        tm_l4 = initTemporalMemory_l4(params=parameters["tm_l4"], seed=seed)

    if predictor_on:
        predictor = Predictor(steps=[1], alpha=parameters["predictor"]["sdrc_alpha"])
        try:
            predictor.loadFromFile("state/predictor/{}_{}_{}_{}_save.temp".format(column_num, sensory_type, x, y))
            print("{}_{}_{}_{} predictor loaded successfully".format(column_num, sensory_type, x, y))
            predictor.reset()
        except RuntimeError:
            print("initialize {}_{}_{}_{} predictor".format(column_num, sensory_type, x, y))

    return sp_l4, tm_l4, predictor


def initialize_object_layer(sensory_type, column_num, x, y, parameters, seed):
    sp_l23 = SpatialPooler()
    tm_l23 = TemporalMemory()

    try:
        sp_l23.loadFromFile("state/sp_l23_{}_{}_{}_{}_save.temp".format(column_num, sensory_type, x, y))
        print("Spatial Pooler l23_{}_{}_{}_{} state loaded successfully".format(column_num, sensory_type, x, y))
        tm_l23.loadFromFile("state/tm_l23_{}_{}_{}_{}_save.temp".format(column_num, sensory_type, x, y))
        print("Temporal Memory l23_{}_{}_{}_{} state loaded successfully".format(column_num, sensory_type, x, y))
    except RuntimeError:
        print("initialize l23_{}_{}_{}_{} Spatial Pooler and Temporal Memory".format(column_num, sensory_type, x, y))
        sp_l23 = initSpatialPooler_l23(params=parameters["sp_l23"], seed=seed)
        tm_l23 = initTemporalMemory_l23(params=parameters["tm_l23"], seed=seed)

    return sp_l23, tm_l23


def sensory_layer(column_num, sdr_active_cells_temporal_dict, sdr_winner_cells_temporal_dict, sp_l4, tm_l4,
                  active_columns_l4, encoding, x, y):

    # compute spatial pooler
    sp_l4.compute(input=encoding, learn=True, output=active_columns_l4)

    # compute temporal memory
    tm_l4.compute(activeColumns=active_columns_l4, learn=True)

    # Save each active/winner cells of each temporal memory in a Dictionary for lateral communication
    if column_num not in sdr_active_cells_temporal_dict:
        sdr_active_cells_temporal_dict[column_num] = {}
    sdr_active_cells_temporal_dict[column_num][f"sensor_{x}_{y}"] = tm_l4.getActiveCells()
    if column_num not in sdr_winner_cells_temporal_dict:
        sdr_winner_cells_temporal_dict[column_num] = {}
    sdr_winner_cells_temporal_dict[column_num][f"sensor_{x}_{y}"] = tm_l4.getWinnerCells()

    return sp_l4, tm_l4


def object_layer(column_num, sdr_active_cells_dict, sdr_winner_cells_dict, sdr_active_cells_temporal_dict,
                 sdr_winner_cells_temporal_dict, sp_l23, tm_l23, tm_l4, active_columns_l23, active_columns_l4):
    '''
    The object_layer function basically collects all the external data from the columns, computes the spatial pooler,
    computes the temporal memory of its layer 2/3, and then it computes the temporal memory sensory layer 4 with its
    collected data as external input.
    The active cells of the sensory layer are used as input for the object_layer and its spatial pool calculation.
    '''

    sensor_layer_active_cells = tm_l4.getActiveCells().flatten()

    # get every 10th bit out of the 51200 active cells SDR. Create a new SDR with 1/10th of the size. And then put
    # all the 10th bits inside the new sdr.

    active_indices = sensor_layer_active_cells.sparse
    selected_indices = active_indices[::10]
    small_sdr = SDR(2048)
    small_sdr.sparse = list({idx % 2048 for idx in selected_indices})

    '''
    {idx % 1024 for idx in selected_indices}" is a Python set comprehension that creates a set of modulo-1024 indices 
    by taking each index in "selected_indices" and computing the result of the modulo operator "%" with 1024. 
    The result is a set of unique indices within the range of 0 to 1023, which are representative of the selected 
    indices in the original SDR.
    "list()" converts the set into a list. This is necessary because the "sparse" property of the "small_sdr" object 
    expects a list.
    '''

    sp_l23.compute(input=small_sdr, learn=True, output=active_columns_l23)

    # initialize 2 SDRs for using the .union() method
    cells_per_column = htm_parameters['tm_l23']['cellsPerColumn']
    column_count = htm_parameters['tm_l23']['columnCount']
    sdr_input_ini_1 = SDR(column_count * cells_per_column)
    sdr_input_ini_2 = SDR(column_count * cells_per_column)

    if column_num.startswith('column_'):
        columns = ['column_' + str(i) for i in range(1, 10) if f'column_{i}' != column_num]

        # Loop through columns and get active and winner cells for each column
        for column_i in columns:
            # Get sensor coordinates for the current column
            sensor = list(sdr_active_cells_dict[column_i].keys())[0]
            x_i, y_i = map(int, sensor.split("_")[1:])

            # Get active and winner cells for the current column
            sdr_active_cells = sdr_active_cells_dict[column_i][f"sensor_{x_i}_{y_i}"].flatten()
            sdr_winner_cells = sdr_winner_cells_dict[column_i][f"sensor_{x_i}_{y_i}"].flatten()

            # Combine all active and winner cells into two arrays
            external_active_cells = sdr_input_ini_1.union([sdr_input_ini_1, sdr_active_cells])
            external_winner_cells = sdr_input_ini_2.union([sdr_input_ini_2, sdr_winner_cells])

        # Reshape arrays
        external_active_cells = external_active_cells.reshape((1024, 50))
        external_winner_cells = external_winner_cells.reshape((1024, 50))

        # Compute with new inputs
        tm_l23.compute(activeColumns=active_columns_l23,
                       learn=True,
                       externalPredictiveInputsActive=external_active_cells,
                       externalPredictiveInputsWinners=external_winner_cells)

        tm_l4.compute(activeColumns=active_columns_l4,
                      learn=True,
                      externalPredictiveInputsActive=tm_l23.getActiveCells(),
                      externalPredictiveInputsWinners=tm_l23.getWinnerCells())

        # Update each active/winner cells from sensory layer l4_tm in a temporal dictionary for lateral communication
        if column_num not in sdr_active_cells_temporal_dict:
            sdr_active_cells_temporal_dict[column_num] = {}
        sdr_active_cells_temporal_dict[column_num][f"sensor_{x}_{y}"] = tm_l4.getActiveCells()
        if column_num not in sdr_winner_cells_temporal_dict:
            sdr_winner_cells_temporal_dict[column_num] = {}
        sdr_winner_cells_temporal_dict[column_num][f"sensor_{x}_{y}"] = tm_l4.getWinnerCells()

        return sp_l23, tm_l23, tm_l4


def prediction(tm, predictions_dict, column_num, x, y, sensory_input, predictor, predictor_res, record_num):
    # Predict what will happen, and then train the predictor based on what just happened.
    pdf = predictor.infer(tm.getActiveCells())
    if pdf[1]:
        predictions_dict[column_num][f"sensor_{x}_{y}"].append(np.argmax(pdf[1]) * predictor_res)
    else:
        predictions_dict[column_num][f"sensor_{x}_{y}"].append(float('nan'))
    predictor.learn(record_num, tm.getActiveCells(), int(sensory_input / predictor_res))

    return predictor, predictions_dict


def create_encoder(enc_params, name):
    """
    encParams = parameters["enc"] -> you access the dictionary in parameters_htm_cfd.py
    name = sensory_type -> select the correct (p,u or v) dictionary/parameters in the enc-dictionary

    Since you cant access class-ScalarEncoders directly with parameters,you have use to class-ScalarsEncodersParameters
    methods, so we define a new variable to access the ScalarEncoderParameters. And with the methods within the
    ScalarsEncodersParameters class we can define the parameters from our own parameter file.

    After that we can create the ScalarEncoder by returning ScalarEncoder(encoder_params). Now we can use all
    methods inside the ScalarEncoder class.
    """

    encoder_params = ScalarEncoderParameters()
    encoder_params.minimum = enc_params[name]["minimum"]
    encoder_params.maximum = enc_params[name]["maximum"]
    encoder_params.size = enc_params[name]["size"]
    encoder_params.sparsity = enc_params[name]["sparsity"]

    return ScalarEncoder(encoder_params)


def initSpatialPooler_l4(params, encoding_width, seed):
    sp_l4 = SpatialPooler(
        inputDimensions=(encoding_width,),
        columnDimensions=(params["columnCount"],),
        potentialPct=params["potentialPct"],
        potentialRadius=encoding_width,
        globalInhibition=True,
        localAreaDensity=params["localAreaDensity"],
        synPermInactiveDec=params["synPermInactiveDec"],
        synPermActiveInc=params["synPermActiveInc"],
        synPermConnected=params["synPermConnected"],
        boostStrength=params["boostStrength"],
        wrapAround=True,
        seed=seed
    )
    return sp_l4


def initSpatialPooler_l23(params, seed):
    sp_l23 = SpatialPooler(
        inputDimensions=(2048,),
        columnDimensions=(params["columnCount"],),
        potentialPct=params["potentialPct"],
        potentialRadius=2048,
        globalInhibition=True,
        localAreaDensity=params["localAreaDensity"],
        synPermInactiveDec=params["synPermInactiveDec"],
        synPermActiveInc=params["synPermActiveInc"],
        synPermConnected=params["synPermConnected"],
        boostStrength=params["boostStrength"],
        wrapAround=True,
        seed=seed
    )
    return sp_l23


def initTemporalMemory_l4(params, seed):
    tm_l4 = TemporalMemory(
        columnDimensions=(params["columnCount"],),
        cellsPerColumn=params["cellsPerColumn"],
        activationThreshold=params["activationThreshold"],
        initialPermanence=params["initialPerm"],
        connectedPermanence=params["synPermConnected"],
        minThreshold=params["minThreshold"],
        maxNewSynapseCount=params["newSynapseCount"],
        permanenceIncrement=params["permanenceInc"],
        permanenceDecrement=params["permanenceDec"],
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=params["maxSegmentsPerCell"],
        maxSynapsesPerSegment=params["maxSynapsesPerSegment"],
        seed=seed,
        externalPredictiveInputs=params["externalPredictiveInputs"],
    )
    return tm_l4


def initTemporalMemory_l23(params, seed):
    tm_23 = TemporalMemory(
        columnDimensions=(params["columnCount"],),
        cellsPerColumn=params["cellsPerColumn"],
        activationThreshold=params["activationThreshold"],
        initialPermanence=params["initialPerm"],
        connectedPermanence=params["synPermConnected"],
        minThreshold=params["minThreshold"],
        maxNewSynapseCount=params["newSynapseCount"],
        permanenceIncrement=params["permanenceInc"],
        permanenceDecrement=params["permanenceDec"],
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=params["maxSegmentsPerCell"],
        maxSynapsesPerSegment=params["maxSynapsesPerSegment"],
        seed=seed,
        externalPredictiveInputs=params["externalPredictiveInputs"],
    )
    return tm_23


def getData(path):  # Function to read out all simulation data from the Excel file
    iterations = []
    p = []
    u = []
    v = []
    for root, dirs, files in os.walk(path):  # files = List of all files in current directory
        for datafile in files:
            if "PUV" in datafile:
                no_ext_file = datafile.replace(".csv", "").strip()  # delete .csv and "space" from file name
                iter_no = int(no_ext_file.split("V")[-1])  # PUV1000 -> .split the name and safe the number 1000
                iterations.append(iter_no)  # create a list of iteration numbers

    sorted_iterations = np.sort(iterations)  # sort the list

    for i in sorted_iterations:
        p_p, u_p, v_p = read_datafile(i, path)  # save p,u,v data from Excel as list in these variables

        p.append(p_p)
        u.append(u_p)
        v.append(v_p)

    return p, u, v


if __name__ == "__main__":

    # Define the coordinates for each column as integers
    column_coordinates = [(64, 66), (65, 66),  (66, 66),
                          (64, 65),  (65, 65), (66, 65),
                          (64, 64),  (65, 64), (66, 64)]

    # Create a dictionary to store the coordinates for each column
    x_y_coordinates = {}

    # Loop over the columns and generate coordinates for each column
    for column in range(9):
        x, y = column_coordinates[column]
        x_y_coordinates[f"column_{column + 1}"] = [(x, y)]

    print("x_y_coordinates", x_y_coordinates)

    learning = int(input(f"Enter '0' for off or '1' for on for lateral communication between Columns: "))
    if learning == 1:
        learning = True
    else:
        learning = False

    for sensory_type in ("v"):
        train(coordinates_dict=x_y_coordinates,
              num_cycles=100,
              sensory_type=sensory_type,
              predictor_on=True,
              learning=learning,
              parameters=htm_parameters)
