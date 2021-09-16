import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DecisionStump:
    
    def __init__(self):
        self.classLabel = None
        self.threshold = None
        self.feature = None
        self.alpha = None
    
    def predict(self , x):
        Feature = x[:,self.feature]
        predictions = np.where(Feature < self.threshold , self.classLabel ,-1 * self.classLabel)
        return predictions

class AdaBoost:
    
    def __init__(self , nbr_classifiers = 10 , epsilon = 1e-10):
        self.nbr_classifiers = nbr_classifiers
        self.epsilon = epsilon
        self.classifiers = []
        
    def fit(self , x , y):
        self.x_train = x
        self.y_train = y
        #Initialize The weights for all the samples with 1 / nbr_samples
        self.weights = np.full(self.x_train.shape[0] , (1 / self.x_train.shape[0]) , dtype=np.float64)
    
    def train(self ):
        for i in range(self.nbr_classifiers):
            Weak_Classifier_i = DecisionStump()
            minimum_Error = float("inf")
            
            #Iterate Over all the features to find the perfect one that will split our data
            for feature in range(self.x_train.shape[1]):
                current_Feature = self.x_train[:,feature]
                #find thresholds Values which is the unique values of the feature that we're working with
                thresholds = np.unique(current_Feature)
                #iterate over all the thresholds to find the perfect one that will split the current feature
                for threshold in thresholds:
                    """
                    we don't know what the class of samples where feature < threshold , this is way we will test with class 1 ,
                    if the error more than 0.5 which is mean the majority of the samples that we calssified as 1 are -1 , so what we will do in this case ?
                    we will flip the error and assign -1 to our class label .
                    
                    if error (label used is 1) = 0.8 
                    then error(label is -1) = 0.2 .
                    
                    """
                    class_Label = 1
                    predictions = np.where(current_Feature < threshold , class_Label , -1 * class_Label)
                    error = np.sum(self.weights[self.y_train != predictions])
                    #flip The Error and The classLabel
                    if error > 0.5 :
                        error = 1-error
                        class_Label = -1
                    #if we find a better error less than the previous (we initialize The Error with float("if") which is a very small number)     
                    if error < minimum_Error:
                        Weak_Classifier_i.classLabel = class_Label
                        Weak_Classifier_i.threshold = threshold
                        Weak_Classifier_i.feature = feature
                        minimum_Error = error                    
            #Calculate The Performance of the Current Weak Classifier            
            Weak_Classifier_i.alpha = 0.5 * np.log((1 - minimum_Error + self.epsilon) / (minimum_Error + self.epsilon))        
            
            #Update The Weights
            predictions = Weak_Classifier_i.predict(self.x_train)
            self.weights *= (np.exp( - Weak_Classifier_i.alpha * predictions * self.y_train)) / (np.sum(self.weights))
            #save our Weak Classifier
            self.classifiers.append(Weak_Classifier_i)
            
    def predict(self , x):
         classifiers_predictions = [classifier.alpha * classifier.predict(x) for classifier in self.classifiers]
         y_pred = np.sum(classifiers_predictions , axis = 0)
         return np.sign(y_pred)
     
    def plotTheModel(self):
        fig , ax = plt.subplots()
        Weak_Classifiers = []
        Errors = []
        for i in range(self.nbr_classifiers):
            Weak_Classifiers.append("c"+str(i))
            Errors.append(self.classifiers[i].alpha)
        ax.bar(Weak_Classifiers , Errors)
        ax.set_ylabel("Error")
        ax.set_xlabel("classifiers")
        ax.set_title("Error of classifiers")
        plt.show()
#Test AdaBoost 

def Accuracy(y , y_hat):
    return np.sum(y != y_hat) / len(y)


x , y = make_blobs(n_samples=500 , n_features=10 , centers=2 , random_state=0)
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.25)

adaBoost  = AdaBoost()
adaBoost.fit(x_train, y_train)
adaBoost.train()
y_hat = adaBoost.predict(x_test)
print("AdaBoost Accuracy : ",Accuracy(y_test, y_hat))
adaBoost.plotTheModel()
 



























"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

class DecisionStump:
    
    def __init__(self):
        self.classLabel = None
        self.threshold = None
        self.feature = None
        self.Alpha = None
        
    def predict(self , x):
         features = x[:,self.feature]
         predictions = np.where(features < self.threshold , self.classLabel  , -1 * self.classLabel)
         return predictions
     
class AdaBoost :

    def __init__(self , nbr_classifiers = 10 , epsilon = 1e-10):
        self.nbr_classifiers = nbr_classifiers
        self.epsilon = epsilon
        self.classifiers = []
        
        
    def fit(self , x , y):
        self.x_train = x
        self.y_train = y
        
    def train(self):
        #Initialize The Weights
        self.weights = np.full(self.x_train.shape[0] , (1 / self.x_train.shape[1]))
        print("weights.shape : ",self.weights.shape)
        #Create nbr_classifiers Weak Classifier and train them
        for i in range(self.nbr_classifiers):
            Weak_Classifier_i = DecisionStump()
            error = self.greedySearch(Weak_Classifier_i)
            Weak_Classifier_i.Alpha = self.StumpPerformance(error)
            predictions = Weak_Classifier_i.predict(self.x_train)
            #Update The Weights 
            self.weights *= (np.exp(- Weak_Classifier_i.Alpha * self.y_train * predictions)) / np.sum(self.weights)
            self.classifiers.append(Weak_Classifier_i)
            
    def StumpPerformance(self , error):
        return 0.5 * np.log((1 - error + self.epsilon) /( error + self.epsilon))         
        
    def greedySearch(self , Weak_Classifier):
        #Iterate Over all The Features to find The best one that can split our data in the most accurate way (minimum Error)
        for feature_Index in range(self.x_train.shape[1]):
            feature_value = self.x_train[:,feature_Index]
            thresholds = np.unique(feature_value)
            minimum_Error = float("inf")
            #Iterate Over All The Thresholds 
            for threshold in thresholds:
                class_label = 1
                predictions = np.where(feature_value < threshold , 1 , -1)
                
                error = sum(self.weights[self.y_train != predictions]) 
                if error > 0.5 :
                    error = 1 -error 
                    class_label = -1
                
                if error < minimum_Error:
                    Weak_Classifier.classLabel = class_label
                    Weak_Classifier.threshold = threshold
                    Weak_Classifier.feature = feature_Index
                    minimum_Error = error
        return minimum_Error      
    
    def predict(self , x):
        classifiers_predictions = [classifier.Alpha * classifier.predict(x) for classifier in self.classifiers]
        y_hat = np.sum(classifiers_predictions , axis = 0)
        return np.sign(y_hat)


#Test AdaBoost
def accuracy(y_true , y_pred):
    return np.sum(y_true != y_pred) / len(y_true)

x , y = make_blobs(n_samples=200 , n_features=2 , centers=2 , random_state=0)
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.25)

adaBoost = AdaBoost()
adaBoost.fit(x_train, y_train)
adaBoost.train()
y_pred = adaBoost.predict(x_test)
print("AdaBoost Accuracy : ",accuracy(y_test, y_pred))   
                

"""