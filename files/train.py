import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense

def create_model(trainX):

    #############################################
    ######### Creating the model ################

    # Now we define and train the model using the Keras API ##
    ## The model will be a LSTM model ##
    # TODO Investigar capas dropout y batch normalization, dice Andoni que funkan bien
    model = Sequential()
    model.add(LSTM(256, return_sequences=False,
            input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    print(model.summary())

    #############################################
    #############################################

    return model

def train_model(model, trainX, trainY, PATH, MODEL_NAME):
    
    # Callback es una funcion a la que llamaremos en el entrenamiento para hacer el Early Stopping
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=50, min_delta=0.01)

    # Train the model
    model.fit(trainX, trainY, epochs=1, batch_size=512, verbose=1,
            shuffle=False, callbacks=[earlystop], validation_split=0.1)
    
    # Save the model
    model.save(PATH + MODEL_NAME)

# Check Pangu CPU
try:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except (ValueError, RuntimeError):
    print("Unable to reach GPU")
    pass

# Read the CSV_NAME'd dataset and return a Pandas DataFrame.
def read_csv(CSV_NAME):
    
    df = pd.read_csv("./" + CSV_NAME)

    return df

# Function to split datasets into TrainX, TrainY, TestX and TestY according to parameters
# and generate suitable input for LSTM models
def splitting(df, column_names, time_steps, window_size, forward_predict):

    # Formatting the dataset for LSTM.
    # Shape is (elements, time_steps, features)
    # Windowing is performed on window_size windows
    # column_name is an array of names to be fed to the LSTM as input
    # column_name[0] should be the feature to be predicted

    num_features = len(column_names)

    # Create empty arrays with the right dimensions and then fill them
    feature = np.empty(
        [int(len(df)/window_size), time_steps, num_features], dtype="float64")
    tag = np.empty(
        [int(len(df)/window_size), 1], dtype="float64")

    # Apply window_size and time_steps considerations and fill the numpy array
    LOOP_TIMES = int(len(df)/window_size) - int((time_steps)/window_size)

    for i in range(LOOP_TIMES):
        # Allocate values into features
        # Take the num_features
        dfportion = df[column_names][i * (window_size): i*(window_size) + time_steps].to_numpy()
        feature[i] = dfportion.reshape(len(dfportion), num_features)
        # Allocate values into tags
        tag[i] = df[column_names[0]][i *
                                     (window_size) + time_steps + forward_predict - 1]

    print("Dataset splitted!")
    return feature, tag

# Calls splitting() and saves the splitted train and test datasets into csv files
def make_dataset(PATH, dataset, time_steps, window_size, forward_predict, num_features, column_names):

    # Split dataset in features
    # Splitting gives numpy array already shaped for LSTM

    Xtbuc, Ytbuc = splitting(dataset, column_names=column_names,
                            time_steps=time_steps, window_size=window_size, forward_predict=forward_predict)

    # Divide into train and test, 75% for train, 25% for test
    trainX, trainY = Xtbuc[:int(len(Xtbuc)*0.75), :,
                        :].copy(), Ytbuc[:int(len(Ytbuc)*0.75)].copy()
    testX, testY = Xtbuc[int(len(Xtbuc)*0.75):, :,
                        :].copy(), Ytbuc[int(len(Ytbuc)*0.75):].copy()

    pd.DataFrame(trainX.reshape(len(trainX)*num_features,time_steps)).to_csv(PATH + "trainX.csv")
    pd.DataFrame(trainY.reshape(len(trainY),1)).to_csv(PATH + "trainY.csv")
    pd.DataFrame(testX.reshape(len(testX)*num_features,time_steps)).to_csv(PATH + "testX.csv")
    pd.DataFrame(testY.reshape(len(testY),1)).to_csv(PATH + "testY.csv")

    print("Datasets done!")

# Reads the saves datasets with make_dataset() with the correct format ready to be fed into LSTMs
def read_datasets(PATH, time_steps, num_features):
    TRAINX = "trainX.csv"
    TRAINY = "trainY.csv"
    TESTX = "testX.csv"
    TESTY = "testY.csv"

    trainX = pd.read_csv(PATH + TRAINX, index_col=0)
    trainY = pd.read_csv(PATH + TRAINY, index_col=0)
    testX = pd.read_csv(PATH + TESTX, index_col=0)
    testY = pd.read_csv(PATH + TESTY, index_col=0)

    trainX = trainX.to_numpy()
    trainY = trainY.to_numpy()
    testX = testX.to_numpy()
    testY = testY.to_numpy()

    trainX = trainX.reshape(int(len(trainX)/num_features), time_steps, num_features)
    trainY = trainY.reshape(len(trainY), 1)
    testX = testX.reshape(int(len(testX)/num_features), time_steps, num_features)
    testY = testY.reshape(len(testY), 1)

    print("Datasets read!")
    return trainX, trainY, testX, testY


df = read_csv(CSV_NAME="dataset.csv")

df.timestamp = pd.to_datetime(df.timestamp)

# Pasar temperatura del bucle a float
df.bucle_temp = df.bucle_temp.replace("\,", ".", regex=True)
df.bucle_temp = pd.to_numeric(df.bucle_temp)

dataset = pd.DataFrame()
dataset["date"] = df.timestamp
dataset["tbuc"] = df.bucle_temp


# Drop NaN values
dataset = dataset.dropna()
dataset = dataset.reset_index()
dataset = dataset.drop(["index"], axis=1)

# Make datasets
time_steps = 200
window_size = 8
forward_predict = 8
num_features = 1
column_names = ["tbuc"]
PATH = "./"

# Make and read the datasets

make_dataset(PATH=PATH, dataset= dataset, time_steps=time_steps, window_size=window_size,
             forward_predict=forward_predict, num_features=num_features, column_names=column_names)

trainX, trainY, testX, testY = read_datasets(
    PATH=PATH, num_features=num_features, time_steps=time_steps)


# Create the model and train it

model = create_model(trainX=trainX)

train_model(model, trainX, trainY, PATH="./", MODEL_NAME="LSTM2FEATURES.pb")

