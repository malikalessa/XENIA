import pandas as pd
from keras.models import load_model
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import Hyperopt_Tunning
import Global_Config as gc
import os
import report
import Create_Adv_Samples



class fine_tuning():

    def __init__(self, dsConfig, config):
        self.config = config
        self.dsConfig = dsConfig

    def tuning_XAI(self, x_train,x_test, y_train, y_test):

        path =self.dsConfig.get('pathModels')
        n_class = int(self.dsConfig.get('n_class'))
        gc.n_class = n_class

        report_name = path + 'fine_tuning_XAI_Conf6.txt'
        try:
            os.remove(report_name)
        except:
            print('')

        XAI_path  = self.dsConfig.get('XAI_DatasetPath')

        XAI_train = pd.read_csv(XAI_path + 'XAI_Train_Conf6.csv')
        XAI_test = pd.read_csv(XAI_path + 'XAI_Test_Conf6.csv')

        config_No =22
        y_train = y_train.append(y_train)
        model, time, score = Hyperopt_Tunning.hypersearch(XAI_train,y_train,XAI_test,y_test, path, config_No)
        model.save(path + 'XAI_fine_tuning_Conf6.h5')
       # model = load_model(path + 'XAI_fine_tuning.h5')
        Y_predicted = np.argmax(model.predict(XAI_test), axis=1)
        Confusion_matrix = confusion_matrix(y_test, Y_predicted)
        Classification_report = classification_report(y_test, Y_predicted)
        Accuracy = accuracy_score(y_test, Y_predicted)

        print('Acc : ', Accuracy)
        report_name = path + 'fine_tuningXAI_Config6.txt'

        try:
            os.remove(report_name)
        except:
            print('')
        name = 'fine_tuningXAI'
        report.report(report_name, Accuracy, Confusion_matrix, Classification_report, name)

    def tuning_after_XAI(self, x_train,x_test, y_train, y_test):

        path = self.dsConfig.get('pathModels')
        n_class = int(self.dsConfig.get('n_class'))
        gc.n_class = n_class

        T_A_dataset, y_advertrain, _, _ = Create_Adv_Samples.Create_Adv_Samples.read_T_A_Datasets(self,x_train, y_train)

        config_No =55
        model, time, score = Hyperopt_Tunning.hypersearch(T_A_dataset,y_advertrain,x_test,y_test, path, config_No)
        model.save(path + 'XAI_double_fine_tuning_config6.h5')
        Y_predicted = np.argmax(model.predict(x_test), axis=1)
        Confusion_matrix = confusion_matrix(y_test, Y_predicted)
        Classification_report = classification_report(y_test, Y_predicted)
        Accuracy = accuracy_score(y_test, Y_predicted)

        print('Acc : ', Accuracy)

        report_name = path + 'double_fine_tuning_Conf6.txt'

        try:
            os.remove(report_name)
        except:
            print('')
        name = 'double_fine_tuning'
        report.report(report_name, Accuracy, Confusion_matrix, Classification_report, name)




    def baseline_fine_tuning_XAI(self, x_train,x_test, y_train, y_test):

        path = self.dsConfig.get('pathModels')
        n_class = int(self.dsConfig.get('n_class'))

        gc.baseline_tobe_tunned =self.dsConfig.get('pathModels') + self.dsConfig.get('baseline_model')

        gc.n_class = n_class

        report_name = path + 'fine_tuning_XAI_baseline.txt'
        try:
            os.remove(report_name)
        except:
            print('')


        XAI_path  = self.dsConfig.get('XAI_DatasetPath')
        XAI_train = pd.read_csv(XAI_path + 'XAI_Train_Baseline.csv')
        XAI_test = pd.read_csv(XAI_path + 'XAI_Test_Baseline.csv')


        ######### To fine tune baseline using XAI
        config_No = 99
        model, time, score = Hyperopt_Tunning.hypersearch(XAI_train, y_train, XAI_test, y_test, path, config_No)
        model.save(path + 'XAI_fine_tuning_baseline.h5')
        Y_predicted = np.argmax(model.predict(XAI_test), axis=1)
        Confusion_matrix = confusion_matrix(y_test, Y_predicted)
        Classification_report = classification_report(y_test, Y_predicted)
        Accuracy = accuracy_score(y_test, Y_predicted)

        print('Acc : ', Accuracy)

        name = 'fine_tuning_XAI_baseline'
        report.report(report_name, Accuracy, Confusion_matrix, Classification_report, name)

        ######### To fine tune baseline after XAI (the double fine tuning)
        config_No = 111
        model, time, score = Hyperopt_Tunning.hypersearch(x_train, y_train, x_test, y_test, path, config_No)
        model.save(path + 'XAI_dounel_fine_tuning_baseline.h5')
        Y_predicted = np.argmax(model.predict(x_test), axis=1)
        Confusion_matrix = confusion_matrix(y_test, Y_predicted)
        Classification_report = classification_report(y_test, Y_predicted)
        Accuracy = accuracy_score(y_test, Y_predicted)

        print('Acc : ', Accuracy)

        report_name = path + 'fine_tuning_After_XAI_baseline.txt'

        try:
            os.remove(report_name)
        except:
            print('')
        name = 'fine_tuning_After_XAI'
        report.report(report_name, Accuracy, Confusion_matrix, Classification_report, name)