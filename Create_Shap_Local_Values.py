import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import Create_Adv_Samples


class Create_Shap_Local_Values():

    def __init__(self, dsConfig, config):
        self.config = config
        self.dsConfig = dsConfig

    def shap_local_values(self, train,test,y_train):

        model_path = self.dsConfig.get('pathModels')
        ################# Computing Shap Values for Config 2 ###############
        if ((int(self.config.get('Config_model'))) == 2) :

            model = load_model(model_path+self.dsConfig.get('baseline_model'))
            XTraining, XValidation, YTraining, YValidation = train_test_split(train, y_train, stratify=y_train,
                                                                      train_size=100,random_state = 42)

            print('Shap XTraining shape : ', XTraining.shape)

            explainer = shap.KernelExplainer(model, XTraining, algorithm='permutation')

        ############ Training Explanation #########################
            shap_values_train = explainer.shap_values(train, nsamples=1000)

            train_values = np.asarray(shap_values_train)
            np.save(self.dsConfig.get('XAI_DatasetPath') +'XAI_Train_Baseline', train_values)
            print(train_values.shape)

            ######## Selecting samples from shap values for training dataset based on the original labels ########

            train_array = []
            for i in range(train.shape[0]):
                j = y_train[i]
                p = (train_values[np.ix_([j], [i])])
                p = p.reshape(1, train.shape[1])
                train_array.append(p)

            train_array = np.asarray(train_array)
            train_array = train_array.reshape(train.shape[0], train.shape[1])

            print( train_array.shape)

            train_XAI = pd.DataFrame(train_array, columns=train.columns)
            print('Train Shap CSV file shape : ',train_XAI.shape)

           ################ Testing Explanation #################

            shap_values_test = explainer.shap_values(test, nsamples=1000)

            test_values = np.asarray(shap_values_test)
            np.save(self.dsConfig.get('XAI_DatasetPath') +'XAI_Test_Baseline', test_values)
            print(test_values.shape)

            prediction_test = model.predict(test)
            prediction_test = np.argmax(prediction_test, axis=1)

         ######## Selecting samples from shap values for testing dataset based on the predctions of Config 2 ########

            y_test = prediction_test

            test_array = []
            for i in range(test.shape[0]):
              j = y_test[i]
              p = (test_values[np.ix_([j], [i])])
              p = p.reshape(1, test.shape[1])

              test_array.append(p)

            test_array = np.asarray(test_array)
            test_array = test_array.reshape(test.shape[0], test.shape[1])

            test_XAI = pd.DataFrame(test_array, columns=test.columns)
            print('Test Shape CSV shape : ',test_XAI.shape)

            #### Scaling the XAI training and Testing datasets###########

            scaler = MinMaxScaler()
            train_scaler = pd.DataFrame(scaler.fit_transform(train_XAI.values),
                                     columns=train_XAI.columns, index=train_XAI.index)

            test_scaler = pd.DataFrame(scaler.transform(test_XAI.values), columns=test_XAI.columns,
                                    index=test_XAI.index)

            path_XAI_Dataset = self.dsConfig.get('XAI_DatasetPath')

            train_scaler.to_csv(path_or_buf = path_XAI_Dataset + 'XAI_Train_Baseline.csv', index=False)

            test_scaler.to_csv(path_or_buf=path_XAI_Dataset + 'XAI_Test_Baseline.csv', index=False)



            ############ Shap Values for Conf-6 ##################################

            #### Config 6 has been trained using T+A samples #############

        if (int(self.config.get('Config_model')) == 6):

                T_A_dataset,y_advertrain,_,_ = Create_Adv_Samples.Create_Adv_Samples.read_T_A_Datasets(self,train,y_train)
                model = load_model(model_path + 'ConfigNo6Attack_Type_1_NN.h5')
                print('y_advtrain : ', y_advertrain.shape)
                XTraining, XValidation, YTraining, YValidation = train_test_split(T_A_dataset, y_advertrain, stratify=y_advertrain,
                                                                                  train_size=100, random_state=42)

                print('Shap XTraining shape for Config 6 : ', XTraining.shape)

                explainer = shap.KernelExplainer(model, XTraining, algorithm='permutation')

                ############ Training Explanation #########################
                shap_values_train = explainer.shap_values(T_A_dataset, nsamples=1000)

                train_values = np.asarray(shap_values_train)
                np.save(self.dsConfig.get('XAI_DatasetPath') + 'XAI_Train_Conf6', train_values)
                print(train_values.shape)
                ######## Selecting samples from shap values for training dataset based on the original labels ########
                y_advertrain = np.asarray(y_advertrain)
                train_array = []
                for i in range(T_A_dataset.shape[0]):
                    j = y_advertrain[i]
                    print('mm',j)
                    p = (train_values[np.ix_([j], [i])])
                   # print(p,'m')
                    p = p.reshape(1, T_A_dataset.shape[1])
                    train_array.append(p)

                train_array = np.asarray(train_array)
                train_array = train_array.reshape(T_A_dataset.shape[0], T_A_dataset.shape[1])
                print(train_array.shape)

                train_XAI = pd.DataFrame(train_array, columns=T_A_dataset.columns)
                print(train_XAI.shape)

                ################ Testing Explanation #################

                shap_values_test = explainer.shap_values(test, nsamples=1000)

                test_values = np.asarray(shap_values_test)
                np.save(self.dsConfig.get('XAI_DatasetPath') + 'XAI_Test_Conf6', test_values)
                print(test_values.shape)


                ######## Selecting samples from shap values for testing dataset based on the predctions of Config 6 ########

                prediction_test = model.predict(test)
                prediction_test = np.argmax(prediction_test, axis=1)

                y_test = prediction_test

                test_array = []
                for i in range(test.shape[0]):
                    j = y_test[i]
                    p = (test_values[np.ix_([j], [i])])
                    p = p.reshape(1, test.shape[1])

                    test_array.append(p)

                test_array = np.asarray(test_array)
                test_array = test_array.reshape(test.shape[0], test.shape[1])

                test_XAI = pd.DataFrame(test_array, columns=test.columns)
                print(test_XAI.shape)

             ####### Scaling XAI datasets ####################33
                scaler = MinMaxScaler()
                train_scaler = pd.DataFrame(scaler.fit_transform(train_XAI.values),
                                            columns=train_XAI.columns, index=train_XAI.index)

                test_scaler = pd.DataFrame(scaler.transform(test_XAI.values), columns=test_XAI.columns,
                                           index=test_XAI.index)

                print('train_scaler min',train_scaler.min())
                print('train_scaler max',train_scaler.max())


                path_XAI_Dataset = self.dsConfig.get('XAI_DatasetPath')

                train_scaler.to_csv(path_or_buf=path_XAI_Dataset + 'XAI_Train_Conf6.csv', index=False)

                test_scaler.to_csv(path_or_buf=path_XAI_Dataset + 'XAI_Test_Conf6.csv', index=False)

        return train_scaler, test_scaler


    def load_shap_values(self):

        XAI_DatasetPath = self.dsConfig.get('XAI_DatasetPath')

        if (int(self.config.get('Config_model')) == 2):
            train_scaler = pd.read_csv(XAI_DatasetPath+'XAI_Train_Conf2.csv')
            test_scaler = pd.read_csv(XAI_DatasetPath+'XAI_Test_Conf2.csv')
            print('Train XAI Conf2 Shape : ', train_scaler.shape)
            print('Test XAI Con2 Shape : ', test_scaler.shape)

        if (int(self.config.get('Config_model')) == 6):
            train_scaler = pd.read_csv(XAI_DatasetPath+'XAI_Train_Conf6.csv')
            test_scaler = pd.read_csv(XAI_DatasetPath+'XAI_Test_Conf6.csv')
            print('Train XAI Conf6 Shape : ', train_scaler.shape)
            print('Test XAI Con6 Shape : ', test_scaler.shape)

        return train_scaler, test_scaler
