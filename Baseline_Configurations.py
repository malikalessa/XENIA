import Global_Config
import report
from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import Baseline_HyperModel
import os
import Global_Config as gc
import Create_Adv_Samples
import pandas as pd


class Baseline_Configurations():

    def __init__(self, dsConfig, config):
        self.config = config
        self.ds = dsConfig

    ##### Creating Baseline Model

    def Baseline_model(self, x_train,x_test, y_train, y_test):
        ##Hyperopt on T1 to learn  DNN####
        ########################### Config 2 #########################3
        path = self.ds.get('pathModels')
        n_class = int(self.ds.get('n_class'))
        gc.n_class = n_class
        report_name = path + 'Hyperopt_Config2.txt'
        try:
            os.remove(report_name)
        except:
            print('')

        config_No = 2

        model_hyperopt, time1,score = Baseline_HyperModel.hypersearch(x_train,y_train,x_test,y_test,path,config_No)
        model_hyperopt.save(path+ self.ds.get('baseline_model'))
        # model_hyperopt = load_model(path + self.ds.get('baseline_model'))

        Y_predicted = np.argmax(model_hyperopt.predict(x_test), axis=1)
        Confusion_matrix = confusion_matrix(y_test, Y_predicted)
        Classification_report = classification_report(y_test, Y_predicted)
        Accuracy = accuracy_score(y_test, Y_predicted)
        print('Accuracy : ', Accuracy)
        name = 'Hyperopt_Configuration2'

        report.report(report_name, Accuracy, Confusion_matrix, Classification_report, name)
        return model_hyperopt

    def Model_trained_on_Adv_Samples(self,x_train,x_test, y_train,y_test):


        path = self.ds.get('pathModels')
        n_class = int(self.ds.get('n_class'))
        gc.n_class = n_class


        #### Adversarial Samples ##############################################################

        adversarial_original_samples, y_label_adversarial, adversarial_samples, _ = Create_Adv_Samples.Create_Adv_Samples.\
            read_T_A_Datasets(self,x_train,y_train)
        print('Adv Samples.shape', adversarial_samples.shape)


        ################ Config 6 #########################################3
        if (int(self.config.get('Dalex_model')) == 6):
            print('Training Conf-6')

            report_name = path + 'Hyperopt_Config_6.txt'

            ### Calling function to predict Adversarial Samples only


            print('Adv.shape used in conf-6', adversarial_original_samples.shape)
            print('Adv label shape used in conf-6 ', y_label_adversarial.shape)


            config_No = 6
            gc.Attack_Type = self.config.get('Attack_Type')
            model, time1, best_score = Baseline_HyperModel.hypersearch(adversarial_original_samples, y_label_adversarial,
                                                              x_test,y_test, path, config_No)

            Y_predicted = np.argmax(model.predict(x_test), axis=1)
            Confusion_matrix = confusion_matrix(y_test, Y_predicted)
            Classification_report = classification_report(y_test, Y_predicted)
            Accuracy = accuracy_score(y_test, Y_predicted)
            print('Accuracy : ', Accuracy)

            try:
               os.remove(report_name)
            except:
               print('')
            name = 'AdvSample_OriginalData'
            report.report(report_name, Accuracy, Confusion_matrix, Classification_report, name)
