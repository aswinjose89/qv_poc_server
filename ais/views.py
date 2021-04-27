from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic.base import View
from braces.views import JSONResponseMixin
import json
from django.conf import settings
import xml.etree.ElementTree as ET


from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers import LeakyReLU
from keras.utils import plot_model
from keras import optimizers
from keras import backend
import numpy as np
import matplotlib.pyplot as plt
import os.path
# from GetoptWrapper import *
import pandas as pd
from math import sin, cos, sqrt, atan2, radians

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from mlflow import log_metric, log_param, log_artifact, set_tracking_uri, set_experiment, start_run, end_run
import mlflow.keras
import tensorflow as tf
# Create your views here.


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

class Base(JSONResponseMixin, View):
    def __init__(self):
        self.remote_server_uri = "http://localhost:5000/" # set to your server URI
        set_tracking_uri(self.remote_server_uri)
        set_experiment("/ais-lstm")

    def ml_flow_training_tracker(self, *args):
        mlflow, ais_training, tags, params, metrics, user_inputs,default_values, data_file_path, model, parameters = args

        mlflow.set_tags(tags)
        
        mlflow.log_params(params)
        
        mlflow.log_metrics(metrics)

        mlflow.log_dict(user_inputs, "user_inputs.json")
        mlflow.log_dict(default_values, "default_values.json")

        log_artifact(data_file_path)
        
                
        log_artifact(parameters['model_json_path'])
        log_artifact(parameters['models_file_path'])
        log_artifact(parameters['model_plot_path'])
        log_artifact(parameters['rmse_plot_path'])
        log_artifact(parameters['loss_plot_path'])
        log_artifact(parameters['history_path'])

        log_artifact("model.json")

        mlflow.keras.log_model(
            keras_model = model,
            artifact_path="ais-model",
            registered_model_name="AIS SHIP Prediction Model"
        )
        print("ml-flow parent run_id: {}".format(ais_training.info.run_id))
        

class AisTrainingView(Base):    

    def post(self, request, *args, **kwargs):
        """
        :param request: API request
        :param args: API args
        :param kwargs: API kwargs
        :return: save/delete user preference settings based on the user id
        """
        data = json.loads(self.request.body.decode('utf-8'))
        status , result  = self.post_api_response(data)
        return self.render_json_response(dict(status=status,result = result))

    def get(self, request, *args, **kwargs):
        """
        :param request: API request
        :param args: API args
        :param kwargs: API kwargs
        :return: Get worklist user preference settings based on the user and its selected worklist
        """
        data = self.qdict_to_dict(request.GET)
        status , result = self.get_api_response(data.get('modeltraining'))
        return self.render_json_response(dict(status=status,result = result))

    def qdict_to_dict(self, qdict):
        """Convert a Django QueryDict to a Python dict.

        Single-value fields are put in directly, and for multi-value fields, a list
        of all values is stored at the field's key.

        """
        return {k: json.loads(v[0]) if len(v) == 1 else v for k, v in qdict.lists()}

    def post_api_response(self, data):
        """
        :param data: Input json data from GUI
        :return: save/delete input json in db by differentiating using "action_name" flag
        """
        kwargs = {
            'is_active': 1
        }
        return 'success', kwargs

    def get_api_response(self, data):

        kwargs = self.build_parameters(data)
        self.train(**kwargs)
        # full_path = '{}/{}'.format(settings.MEDIA_ROOT, "schema/sr2ml/FTmodel_skeletion.xml")
        # ft_model_schema = ET.parse(full_path)
        # ft_model_root = ft_model_schema.getroot()
        # trigger_elements = ft_model_root.find('Models')
        # ext_model_attrib = {
        #     "name": data.get('modelName').get('value'),
        #     "subType": data.get('subType').get('value')
        # }
        # ext_model = ET.SubElement(trigger_elements, 'ExternalModel', attrib= ext_model_attrib)
        # ET.SubElement(ext_model, 'topEvents').text = data.get('topEvents')
        # for map in data.get('map'):
        #     map_attrib = {}
        #     map_attrib['var'] = map.get('value')
        #     ET.SubElement(ext_model, 'map', attrib = map_attrib).text = map.get('label')
        # variables = [map.get('label') for map in data.get('map')]
        # variables.append(data.get('topEvents'))
        # ET.SubElement(ext_model, 'variables').text = ','.join(variables)
        # ET.dump(ft_model_root)
        # ft_model_schema.write('{}/{}'.format(settings.MEDIA_ROOT, "schema/sr2ml/input_files/FTmodel.xml"))
        results = {
            'is_active': 1
        }
        return 'success', results
    
    def build_parameters(self, data):
        kwargs = {}
        kwargs['DATASET_DIR']=""
        kwargs['DATASET']="aisdk_20181101_first35kLines_train_test"
        INFILE = 'ais/data/{}.npz'.format(kwargs['DATASET'])
        kwargs['data_file_path'] = '{}/{}'.format(settings.MEDIA_ROOT, INFILE)  

        kwargs['optimizer']= data.get('optimizer').get('value')
        kwargs['loss']= data.get('loss').get('value')
        kwargs['batch_size'] = 72
        kwargs['NUM_EPOCHS'] = data.get('epochs')
        kwargs['L1_SIZE'] = data.get('layer1_size')

        kwargs['num_features'] = 5       # The data we are submitting per time step (lat, long, speed, time, course)
        kwargs['num_timesteps'] = 5       # The number of time steps per sequence (track)
        kwargs['DROPOUT'] = data.get('dropout')
        return kwargs
        



    def get_training_data(self, data_file_path):

        #%% Pull Data From Numpy File
        training_data = np.load( data_file_path )
        x_train = training_data['x_train']
        y_train = training_data['y_train']
        x_test = training_data['x_test']
        y_test = training_data['y_test']
        return x_train, y_train, x_test, y_test

    def train(self, **kwargs):
        #
        SAVE = True

        #%% command line arguments
        # config_arguments_short_list = ["d", "a", "n", "b", "t", "1", "2", "3", "l", "r", "i"]
        # config_arguments_long_list = ["DATASET_DIR", "DATASET", "NUM_EPOCHS", "DBG", "BATCH_LENGTH", "L1_SIZE", "L2_SIZE", "L3_SIZE", "LEAKY", "DROPOUT", "TRAIN_BATCH_LENGTH"]
        # config_type_list = ["str", "str", "int", "int",
        #                     "int", "int", "int", "int",
        #                     "bool", "float", "int"]
        default_values = {
                        "NUM_EPOCHS": 5,
                        "DBG": 4,
                        "BATCH_LENGTH": 5,
                        "L1_SIZE": 32,
                        "L2_SIZE": 16,
                        "L3_SIZE": 8,
                        "LEAKY": True,
                        "DROPOUT": 0,
                        "TRAIN_BATCH_LENGTH": 2}
        # config_help = {"-d": "dataset dir",
        #             "-a": "dataset",
        #             "-n": "num epochs",
        #             "-b": "debug",
        #             "-t": "batch length",
        #             "-1": "l1 size",
        #             "-2": "l2 size",
        #             "-3": "l3 size",
        #             "-l": "use leaky or not",
        #             "-r": "dropout rate (0 for don't use)",
        #             "-i": "train batch length"}
        # config_parser = GetoptWrapper( config_arguments_short_list,
        #                             config_arguments_long_list=config_arguments_long_list,
        #                             config_type_list=config_type_list,
        #                             default_values=default_values,
        #                             config_help=config_help )
        # if ( config_parser.parse( sys.argv[1:] ) != True ):
        #     print( "parse error" ); quit()
        # config_parser.print()
        # config = config_parser.get_config()

        #%% unpack command line arguments
        # DATASET_DIR = config[ "DATASET_DIR" ]
        # DATASET = config[ "DATASET" ]
        NUM_EPOCHS = kwargs[ "NUM_EPOCHS" ]
        BATCH_LENGTH = kwargs['num_timesteps']
        # DBG = config[ "DBG" ]
        L1_SIZE = kwargs[ "L1_SIZE" ]
        # L2_SIZE = config[ "L2_SIZE" ]
        # L3_SIZE = config[ "L3_SIZE" ]
        # LEAKY = config[ "LEAKY" ]
        DROPOUT = kwargs[ "DROPOUT" ]
        DATASET_DIR = kwargs[ "DATASET_DIR" ]

        data_file_path = kwargs['data_file_path']
        # TRAIN_BATCH_LENGTH = config[ "TRAIN_BATCH_LENGTH" ]

        # INFILE = '../data/{}/{}_train_test.npz'.format( DATASET_DIR, DATASET )
        MODELS_DIR = "ais/models/"
        # REPORTS_DIR = "../reports/{}/".format( DATASET_DIR )
        MODELS_DIR_PATH = '{}/{}'.format(settings.MEDIA_ROOT, MODELS_DIR)
        
        tf.random.set_seed(221)

        # Constants     
        LEAKY = False                   # Include a Leaky RelU Activation Layer?

        num_features = kwargs['num_features']        # The data we are submitting per time step (lat, long, speed, time, course)
        num_timesteps = BATCH_LENGTH       # The number of time steps per sequence (track)

        optimizer= kwargs[ "optimizer" ]
        loss= kwargs[ "loss" ]
        batch_size = 72

        PLOTS_DIR= '{}/{}'.format(settings.MEDIA_ROOT, "ais/plots/")

        #%% make directories if they don't exist
        if ( not os.path.isdir( MODELS_DIR_PATH ) ):
            print( "creating {}".format( MODELS_DIR_PATH ) )
            os.makedirs( MODELS_DIR_PATH, exist_ok = True)
        # if ( True or not os.path.isdir( REPORTS_DIR ) ):
        #     print( "creating {}".format( REPORTS_DIR ) )
        #     os.makedirs( REPORTS_DIR, exist_ok = True )

        #     print( "creating {}".format( REPORTS_DIR + "/figures/" ) )
        #     os.makedirs( REPORTS_DIR + "figures/", exist_ok = True )
            
        #%% Create date string to use in the naming of model files
        datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')


        #%% Function Definitions
        def rmse(y_true, y_pred):
            return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

        def distance(lat1, lon1, lat2, lon2):
            r = 6373.0

            lat1 = radians(lat1)
            lon1 = radians(lon1)
            lat2 = radians(lat2)
            lon2 = radians(lon2)

            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))

            dist = r*c*1000

            return dist

        

        #%% Pull Data From Numpy File
        # training_data = np.load( INFILE )
        # x_train = training_data['x_train']
        # y_train = training_data['y_train']
        # x_test = training_data['x_test']
        # y_test = training_data['y_test']
        x_train, y_train, x_test, y_test = self.get_training_data(data_file_path)

        

        #%% Defining Layers, optimizer, and model; save to a file; orig size 32,6,8
        model = Sequential()
        # best for all
        model.add(LSTM(L1_SIZE, input_shape=(num_timesteps, num_features))) # 3x5 (batch len x inputs)
        model.add(Dropout(DROPOUT))
        model.add(Dense(2))
        model.compile(loss=loss, optimizer=optimizer, metrics=[rmse])
        print(model.summary())
        model_plot_path = "{}/model.png".format(PLOTS_DIR)
        if ( SAVE ):
            plot_model(model, to_file=model_plot_path, show_shapes=True )

        tags = {"engineering": "Maritime AIS Platform",
                "release.candidate": "RC1",
                "release.version": "1.0"}
        with mlflow.start_run(run_name='AIS_Model_Training') as ais_training:
            mlflow.keras.autolog()
            # fit network
            history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=72, validation_data=(x_test, y_test), verbose=2, shuffle=False)

            # save model
            if ( SAVE ):
                model_json = model.to_json()
                model_json_path = "{}/model.json".format(MODELS_DIR_PATH)
                with open( model_json_path, 'w') as json_file:
                    json_file.write(model_json_path)
            # save weights
            if ( SAVE ):
                models_file_path= "{}/weights.h5".format(MODELS_DIR_PATH)
                model.save_weights( models_file_path )
            
            
            

            rmse_plot_path = '{}/rmse_plot.png'.format(PLOTS_DIR)
            plt.figure(0)
            plt.plot(history.history['rmse'])
            plt.title('LSTM: Regression Analytics')
            plt.xlabel('Epoch')
            plt.legend(['RMSE'], loc='upper right')
            plt.savefig( rmse_plot_path, bbox_inches='tight' )
            # plt.show(block=False)

            loss_plot_path = '{}/loss_plot.png'.format(PLOTS_DIR)
            plt.figure(1)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('LSTM: Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Training', 'Validation'], loc='upper right')
            plt.savefig( loss_plot_path, bbox_inches='tight' )


            history_path = '{}/history.csv'.format(PLOTS_DIR)
            hist_df = pd.DataFrame( history.history )
            hist_df.to_csv( history_path )

            parameters = {
                'rmse_plot_path': rmse_plot_path,
                'loss_plot_path': loss_plot_path,
                'history_path': history_path,
                'model_json_path': model_json_path,
                'model_plot_path': model_plot_path,
                'models_file_path': models_file_path
            }

            # print( "" )

            # # make a prediction
            # yhat = model.predict(x_test)
            # print( "yhat shape {}".format( yhat.shape ) )
            # print( "test_y first 5\n{}".format( y_test[0:5,:] ) )
            # print( "yhat first 5\n{}".format( yhat[0:5,:] ) )

            # df = pd.DataFrame( {'pred_lat': yhat[:,0],
            #                     'pred_lon': yhat[:,1],
            #                     'gt_lat': y_test[:,0],
            #                     'gt_long': y_test[:,1] } )
            # if ( SAVE ):
            #     print( "writing ./pred.csv" )
            #     df.to_csv( "./pred.csv", index=False )

            # print( "rmse" )
            # rmse = 0.
            # avg_distance = 0.
            # for i in range( yhat.shape[0] ):
            #     tmp = sqrt(mean_squared_error(yhat[i,:], y_test[i,:]))
            #     rmse += tmp

            #     avg_distance += distance( yhat[i,0], yhat[i,1], y_test[i,0], y_test[i,1] )
            # rmse = rmse/((float)(yhat.shape[0]))
            # avg_distance = avg_distance/((float)(yhat.shape[0]))
            # avg_distance_km = avg_distance/(1000.0)

            # print( "rmse %.3f" % rmse )
            # print( "avg distance in meters %.3f" % avg_distance ) 
            # print( "avg distance in kilometers %.3f" % avg_distance_km )

            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # yhat_lat = []; yhat_long = []
            # ytest_lat = []; ytest_long = []
            # for i in range( yhat.shape[0] ):
            #     yhat_lat.append( yhat[i,0] )# y
            #     yhat_long.append( yhat[i,1] )# x
            
            #     ytest_lat.append( y_test[i,0] )# y
            #     ytest_long.append( y_test[i,1] )# x
                
            # ax.scatter( yhat_long, yhat_lat, color='blue' )
            # ax.scatter( ytest_long, ytest_lat, color='red' )
            # fig.savefig( 'pred.png' )
            # plt.show()
            params = {
                "Number of Epochs": NUM_EPOCHS, 
                "Batch Size": batch_size,
                "Optimizer": optimizer,
                "Loss Functions": loss,
                "x_train shape": x_train.shape,
                "y_train shape": y_train.shape,
                "First Layer Size": L1_SIZE
            }

            metrics = {
            }
            user_inputs = kwargs
            args = (mlflow, ais_training, tags, params, metrics, user_inputs, default_values, data_file_path, model, parameters)
            self.ml_flow_training_tracker(*args)
            

        # model_uri = "runs:/{}/ais-model".format(parent_run.info.run_id)
        # mv = mlflow.register_model(model_uri, "AisLstmModel")

        mlflow.end_run()