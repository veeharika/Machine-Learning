from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import numpy as np
from tkinter import ttk
import PIL
from PIL import ImageTk, Image
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


main = tkinter.Tk()
main.title("Software Defect Estimation-17MIS0344")
main.geometry("1200x1200")

global filename
global balance_data
global X, Y, X_train, X_test, y_train, y_test
global mlp_acc, rbf_acc, svm_acc, bagging_acc, forest_acc, naive_acc, multinomial_acc

def splitdataset(balance_data):
    cols = balance_data.shape[1]-1
    X = balance_data.values[:, 0:cols] 
    Y = balance_data.values[:, cols]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.2, random_state = 0)
    print(y_test,"y_test valuesssssssss")  #thisss
    return X, Y, X_train, X_test, y_train, y_test 

def upload(event):
    global filename
    global balance_data
    global X, Y, X_train, X_test, y_train, y_test
    filename = askopenfilename(initialdir = "test")
    pathlabel.config(text=filename)
    balance_data = pd.read_csv(filename)
    X, Y, X_train, X_test, y_train, y_test = splitdataset(balance_data)
    text.insert(END,"Dataset Length : "+str(len(X))+"\n");
    text.insert(END,"Splitted Training Length : "+str(len(X_train))+"\n");
    text.insert(END,"Splitted Test Length : "+str(len(X_test))+"\n\n");

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
#Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details):
    print(y_test,"y_testtttttttttttttttttt")
    print(y_pred,"y_preddddddddddddddddddd")
    ind=[]
    def1=[]
    for i in range(len(y_test)):
        if y_test[i]!=y_pred[i]:
            ind.append(i)
        if y_test[i]==y_pred[i] and y_pred[i]==1 and y_test[i]==1:
            def1.append(i)
    print("Defects",def1)
            
    print("error found at", ind)
            
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    text.insert(END,"Report : "+str(classification_report(y_test, y_pred))+"\n")
    return accuracy

#Function to calculate accuracy 
def RBFcal_accuracy(y_test, y_pred, details): 
    accuracy = accuracy_score(y_test,y_pred.round(), normalize=False)*100
    text.insert(END,details+"\n\n")
    text.insert(END,"Report : "+str(classification_report(y_test, y_pred.round()))+"\n")
    return accuracy/100  
        
def runMLP(event):
    global mlp_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    mlp_acc = cal_accuracy(y_test, prediction_data,'Multilayer Perceptron Algorithm Accuracy, Classification Report')
    text.insert(END,"Multilayer Perceptron Accuracy : "+str(mlp_acc)+"\n\n")
    
def runRBF(event):
    global rbf_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    kernel = RBF(length_scale=1)
    cls = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=9)
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    rbf_acc = RBFcal_accuracy(y_test, prediction_data,'Radial Basis Function Algorithm Accuracy, Classification Report')
    text.insert(END,"Radial Basis Function Accuracy : "+str(rbf_acc)+"\n\n")
    
    
def runSVM(event):
    global svm_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2) 
    cls.fit(X_train, y_train) 
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    svm_acc = cal_accuracy(y_test, prediction_data,'SVM Accuracy, Classification Report')
    text.insert(END,"SVM Accuracy : "+str(svm_acc)+"\n\n")
    
def runBagging(event):
    global bagging_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0)
    cls.fit(X_train, y_train) 
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    bagging_acc = cal_accuracy(y_test, prediction_data,'Bagging Classifier Accuracy, Classification Report')
    text.insert(END,"Bagging Classifier Accuracy : "+str(bagging_acc)+"\n\n")
    
def runRandomForest(event):
    global forest_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = RandomForestClassifier(n_estimators=1,max_depth=0.9,random_state=None)
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    forest_acc = cal_accuracy(y_test, prediction_data,'Random Forest Algorithm Accuracy, Classification Report')
    text.insert(END,"Random Forest Accuracy : "+str(forest_acc)+"\n\n")

    
def runNaiveBayes(event):
    global naive_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = GaussianNB()
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    naive_acc = cal_accuracy(y_test, prediction_data,'Naive Bayes Algorithm Accuracy, Classification Report')
    text.insert(END,"Naive Bayes Accuracy : "+str(naive_acc)+"\n\n")
    
def runMultinomial(event):
    global multinomial_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = MultinomialNB()
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    multinomial_acc = cal_accuracy(y_test, prediction_data,'Multinomial Naive Bayes Algorithm Accuracy, Classification Report')
    text.insert(END,"Multinomial Naive Bayes Algorithm Accuracy : "+str(multinomial_acc)+"\n\n")

def graph(event):
    height = [mlp_acc, rbf_acc, svm_acc, bagging_acc, forest_acc, naive_acc, multinomial_acc]
    bars = ('MLP', 'RBF', 'SVM Accuracy', 'Bagging Accuracy', 'Random Forest Accuracy', 'Naive Bayes Accuracy', 'Multinomial Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    
font = ('times', 20, 'bold')
title = Label(main, text='Software Defect Estimation Using Machine Learning Algorithms')
# title.config(bg='green', fg='black')  
title.config(font=font)           
# title.config(height=1, width=80)       
title.place(x=5,y=0)

font1 = ('times', 15, 'bold')
img = ImageTk.PhotoImage(PIL.Image.open("icons/b1.png"))
uploadButton = Button(main, text="", image=img)
uploadButton.place(x=30,y=60)
uploadButton.config(font=font1)
uploadButton.bind('<Button-1>',upload)

pathlabel = Label(main)
pathlabel.config(bg='blue', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=600,y=70)

img1 = ImageTk.PhotoImage(PIL.Image.open("icons/b2.png"))
mlpButton = Button(main, text="", image=img1)
mlpButton.place(x=30,y=110)
mlpButton.config(font=font1)
mlpButton.bind('<Button-1>',runMLP)

img2 = ImageTk.PhotoImage(PIL.Image.open("icons/b3.png"))
rbfButton = Button(main, text="", image=img2)
rbfButton.place(x=30,y=160)
rbfButton.config(font=font1)
rbfButton.bind('<Button-1>',runRBF)

img3 = ImageTk.PhotoImage(PIL.Image.open("icons/b4.png"))
svmButton = Button(main, text="", image=img3)
svmButton.place(x=30,y=210)
svmButton.config(font=font1)
svmButton.bind('<Button-1>',runSVM)

img4 = ImageTk.PhotoImage(PIL.Image.open("icons/b5.png"))
baggingButton = Button(main, text="", image=img4)
baggingButton.place(x=30,y=260)
baggingButton.config(font=font1)
baggingButton.bind('<Button-1>',runBagging)

img5 = ImageTk.PhotoImage(PIL.Image.open("icons/b6.png"))
forestButton = Button(main, text="", image=img5)
forestButton.place(x=30,y=310)
forestButton.config(font=font1)
forestButton.bind('<Button-1>',runRandomForest)

img6 = ImageTk.PhotoImage(PIL.Image.open("icons/b7.png"))
naiveButton = Button(main, text="", image=img6)
naiveButton.place(x=30,y=360)
naiveButton.config(font=font1)
naiveButton.bind('<Button-1>',runNaiveBayes)

img7 = ImageTk.PhotoImage(PIL.Image.open("icons/b8.png"))
multinomialButton = Button(main, text="", image=img7)
multinomialButton.place(x=30,y=410)
multinomialButton.config(font=font1)
multinomialButton.bind('<Button-1>',runMultinomial)

img8 = ImageTk.PhotoImage(PIL.Image.open("icons/b9.png"))
graphButton = Button(main, text="", image=img8)
graphButton.place(x=30,y=460)
graphButton.config(font=font1)
graphButton.bind('<Button-1>',graph)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=60)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=600,y=55)
text.config(font=font1)


main.config(bg='seagreen')
main.mainloop()
