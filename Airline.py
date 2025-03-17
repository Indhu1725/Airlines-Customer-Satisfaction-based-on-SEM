from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

main = tkinter.Tk()
main.title("Airline Customer Review")
main.geometry("1300x1200")

global filename
global x, y, X_train, X_test, y_train, y_test
global model
global df
global random_acc,lr_acc, svm_acc, dt_acc

def upload():
    global filename
    global df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Dataset Size : "+str(len(df))+"\n")
    # Data preprocessing
    df = df.sample(n=15000)
    text.insert(END, "Sampled dataset size: {}\n".format(df.shape))
    text.insert(END, "Data types:\n{}\n".format(df.dtypes))
    text.insert(END, "Dataset info:\n{}\n".format(df.info()))
    text.insert(END, "Missing values per column:\n{}\n".format(df.isnull().sum()))
    df.dropna(inplace=True)
    text.insert(END, "Dataset shape after dropping NaNs: {}\n".format(df.shape))
    # Display descriptive statistics in a table format
    stats_text = df.describe().to_string()
    text.insert(END, "Descriptive statistics:\n{}\n".format(stats_text))


def Labeling():
    global df
    df = df.drop(['id', 'Unnamed: 0', 'Customer Type', 'Gate location', 'Type of Travel'], axis=1)
    
    # Replace categorical values with numerical equivalents
    refactored_values = {
        'Gender': {'Male': 0, 'Female': 1},
        'satisfaction': {'neutral or dissatisfied': 0, 'satisfied': 1},
        'Class': {'Business': 0, 'Eco': 1, 'Eco Plus': 2}
    }
    df.replace(refactored_values, inplace=True)
    text.insert(END, "Labeling : {}\n".format(df.head()))
    

def split_data():
    global df, X_train, X_test, y_train, y_test
    x = df.drop(columns='satisfaction', axis=1)
    y = df['satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2)
    text.insert(END, "X_train shape: {}\n".format(X_train.shape))
    text.insert(END, "X_test shape: {}\n".format(X_test.shape))
    text.insert(END, "y_train shape: {}\n".format(y_train.shape))
    text.insert(END, "y_test shape: {}\n".format(y_test.shape))

def random():
    global X_train, X_test, y_train, y_test, random_acc,random
    random = RandomForestClassifier()
    random.fit(X_train, y_train)
    y_pred = random.predict(X_test)
    random_acc = accuracy_score(y_test, y_pred)
    text.insert(END, 'Random Forest Classifier\n')
    text.insert(END, f'Model accuracy: {random_acc}\n')
    text.insert(END, f'Accuracy in Percentage: {"{:.1%}".format(random_acc)}\n')
    text.insert(END, classification_report(y_test, y_pred) + '\n')
    con_matrix = confusion_matrix(y_test, y_pred)
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(con_matrix, annot=True, annot_kws={"size":16}, cmap="Blues", fmt='g')
    plt.xlabel('Predicted value')
    plt.ylabel('Actual value')
    plt.title('Confusion matrix for Random Forest Classifier')
    plt.show()

def LLR():
    global X_train, X_test, y_train, y_test,lr_acc
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    LRmodel = LogisticRegression(max_iter=1000)
    LRmodel.fit(X_train_scaled, y_train)
    y_pred = LRmodel.predict(X_test_scaled)
    lr_acc = LRmodel.score(X_test_scaled, y_test)
    text.insert(END, 'Logistic Regression Classifier\n')
    text.insert(END, f'Model accuracy: {lr_acc}\n')
    text.insert(END, f'Accuracy in Percentage: {"{:.1%}".format(lr_acc)}\n')
    text.insert(END, classification_report(y_test, y_pred) + '\n')
    con_matrix = confusion_matrix(y_test, y_pred)
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(con_matrix, annot=True, annot_kws={"size":16}, cmap="Blues", fmt='g')
    plt.xlabel('Predicted value')
    plt.ylabel('Actual value')
    plt.title('Confusion matrix for Logistic Regression Classifier')
    plt.show()

def support():
    global X_train, X_test, y_train, y_test,sv,svm_acc
    sv = SVC()
    sv.fit(X_train, y_train)
    y_pred = sv.predict(X_test)
    svm_acc = sv.score(X_test, y_test)
    text.insert(END, 'Support Vector Classifier\n')
    text.insert(END, f'Model accuracy: {svm_acc}\n')
    text.insert(END, f'Accuracy in Percentage: {"{:.1%}".format(svm_acc)}\n')
    text.insert(END, classification_report(y_test, y_pred) + '\n')
    con_matrix = confusion_matrix(y_test, y_pred)
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(con_matrix, annot=True, annot_kws={"size":16}, cmap="Blues", fmt='g')
    plt.xlabel('Predicted value')
    plt.ylabel('Actual value')
    plt.title('Confusion matrix for Support Vector Classifier')
    plt.show()
    
def Decision():
    global X_train, X_test, y_train, y_test,dt_acc
    Decision = DecisionTreeClassifier()
    Decision.fit(X_train, y_train)
    y_pred = Decision.predict(X_test)
    dt_acc = Decision.score(X_test, y_test)
    text.insert(END, 'Decision Tree\n')
    text.insert(END, f'Model accuracy: {dt_acc}\n')
    text.insert(END, f'Accuracy in Percentage: {"{:.1%}".format(dt_acc)}\n')
    text.insert(END, classification_report(y_test, y_pred) + '\n')
    con_matrix = confusion_matrix(y_test, y_pred)
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(con_matrix, annot=True, annot_kws={"size":16}, cmap="Blues", fmt='g')
    plt.xlabel('Predicted value')
    plt.ylabel('Actual value')
    plt.title('Confusion matrix for Decision Tree')
    plt.show()
    

def comparison():
    # List to store model names and their corresponding accuracy scores
    models = ['Random Forest', 'Logistic Regression', 'Support Vector', 'Decision Tree']
    accuracies = [random_acc, lr_acc, svm_acc, dt_acc]  # Assuming these variables are globally accessible
    # Plot comparison graph
    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Model Accuracies')
    plt.ylim(0.5, 1.0)  # Set y-axis limits
    plt.show()

def predict_satisfaction():
    global random, text
    # Assuming `random` is the trained RandomForestClassifier model
    filename = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(filename)
    # Display information about the dataset
    text.insert(END, "Original dataset shape: {}\n".format(test.shape))
    test = test.sample(n=15)
    text.insert(END, "Sampled dataset shape: {}\n".format(test.shape))
    text.insert(END, "Data types:\n{}\n".format(test.dtypes))
    text.insert(END, "Missing values per column:\n{}\n".format(test.isnull().sum()))
    # Drop unnecessary columns
    test = test.drop(['id', 'Unnamed: 0', 'Customer Type', 'Gate location', 'Type of Travel', 'satisfaction'], axis=1)
    # Replace categorical values with numerical equivalents
    refactored_values = {
        'Gender': {'Male': 0, 'Female': 1},
        'Class': {'Business': 0, 'Eco': 1, 'Eco Plus': 2}
    }
    test.replace(refactored_values, inplace=True)
    # Convert float columns to integer if necessary
    for column in test.columns:
        if test[column].dtype == 'float64':
            test[column] = test[column].astype(int)
    # Display the shape of the preprocessed DataFrame
    text.insert(END, "Processed dataset shape: {}\n".format(test.shape))
    # Make predictions using the trained model
    prediction = random.predict(test)
    # Display prediction results
    for pred in prediction:
        if pred == 1:
            text.insert(END, 'Passenger Satisfied!!!\n')
        else:
            text.insert(END, 'Passenger Neutral or Dissatisfied\n')


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

font = ('times', 16, 'bold')
title = Label(main, text='Airline Customer Review')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=2, width=30)       
title.place(x=50, y=20)

font1 = ('times', 14, 'bold')
upload_btn = Button(main, text="Upload Dataset", command=upload)
upload_btn.place(x=700, y=100)
upload_btn.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700, y=150)

LabelingButton = Button(main, text="Labeling", command=Labeling)
LabelingButton.place(x=700, y=200)
LabelingButton.config(font=font1)


split_dataButton = Button(main, text="Split_Data", command=split_data)
split_dataButton.place(x=700,y=250)
split_dataButton.config(font=font1)

randomButton = Button(main, text="Random forest", command=random)
randomButton.place(x=700, y=300)
randomButton.config(font=font1)

LLRButton = Button(main, text="Logistic Regression", command=LLR)
LLRButton.place(x=700, y=350)
LLRButton.config(font=font1)

supportButton = Button(main, text="Support vector Mechine", command=support)
supportButton.place(x=700, y=400)
supportButton.config(font=font1)


DecisionButton = Button(main, text="Decision Tree", command=Decision)
DecisionButton.place(x=700, y=450)
DecisionButton.config(font=font1)


comparisonButton = Button(main, text="comparison", command=comparison)
comparisonButton.place(x=700, y=500)
comparisonButton.config(font=font1)

predict_satisfactionButton = Button(main, text="Predict", command=predict_satisfaction)
predict_satisfactionButton.place(x=700, y=550)
predict_satisfactionButton.config(font=font1)



main.config(bg='turquoise')
main.mainloop()
