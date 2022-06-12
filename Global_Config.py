
import numpy as np

best_score = np.inf
best_accuracy=0
best_scoreTest=0
best_model = None
best_model_test=None
best_numparameters = 0
best_score2=0
best_time =0
savedScore=[]
savedTrain=[]
n_class=None
test_path=None
model_path = None
train_X= None
train_Y= None
test_X=None
test_Y =None
train_R =None

XAI_train  = None
Xai_test = None

y_xaitrain = None
y_xaitest = None


DNN_Model = None
DNN_Model_Batch_Size = 0

Attack_Type = None

config_No = 0

validation_Y = None
XValidation = None

baseline_tobe_tunned = None