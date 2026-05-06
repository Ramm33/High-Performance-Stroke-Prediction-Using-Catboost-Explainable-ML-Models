#importing python classes and packages
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import catboost as cb
#import require python classes and packages

from flask import *
from auth_utils import *
from werkzeug.utils import secure_filename
import os,random
import numpy as np
import pandas as pd


random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt','csv'}
MAX_UPLOAD_SIZE_MB = 512  # Maximum upload size in megabytes

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # Set max content length

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#define global variables to calculate and store accuracy and other metrics
precision = []
recall = []
fscore = []
accuracy = []
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/login')
def login():
    return render_template('login.html')


@app.route("/signup")
def signup_route():
    return signup()


@app.route("/signin")
def signin_route():
    return signin()

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/load_dataset')
def load_dataset():
    global dataset
    # Load the dataset and fill missing values with 0
    try:
        dataset = pd.read_csv("Dataset/healthcare-dataset-stroke-data.csv")
        dataset.fillna(0, inplace=True)
        
        # Convert the dataset to HTML for easy display
        dataset_html = dataset.to_html(classes='table table-striped')
        message = "Dataset loaded successfully."
    except FileNotFoundError:
        dataset_html = "<p>Dataset file not found. Please ensure the file path is correct.</p>"
        message = "Error: Could not load dataset."
    except Exception as e:
        dataset_html = f"<p>An error occurred: {str(e)}</p>"
        message = "An error occurred while loading the dataset."

    # Render the template and display the dataset and message
    return render_template("dataset.html", dataset_html=dataset_html, message=message)

@app.route("/preprocess")
def preprocess_and_split_data():
    global labels,X_train,X_test,y_train,y_test,encoder1,encoder2,encoder3,encoder4,encoder5,sc1,features_selector
    #labels in dataset
    unique, count = np.unique(dataset['stroke'], return_counts = True)
    labels = ['Normal', 'Stroke']
    #convert categorical (non-numeric data) into numeric format using label encoder class
    encoder1 = LabelEncoder()
    encoder2 = LabelEncoder()
    encoder3 = LabelEncoder()
    encoder4 = LabelEncoder()
    encoder5 = LabelEncoder()
    dataset['gender'] = pd.Series(encoder1.fit_transform(dataset['gender'].astype(str)))#encode all str columns to numeric 
    dataset['ever_married'] = pd.Series(encoder2.fit_transform(dataset['ever_married'].astype(str)))#encode all str columns to numeric
    dataset['work_type'] = pd.Series(encoder3.fit_transform(dataset['work_type'].astype(str)))
    dataset['Residence_type'] = pd.Series(encoder4.fit_transform(dataset['Residence_type'].astype(str)))
    dataset['smoking_status'] = pd.Series(encoder5.fit_transform(dataset['smoking_status'].astype(str)))
    
    
    #drop ID column and then extract X training features and Y target label
    Y = dataset['stroke'].ravel()
    dataset.drop(['id', 'stroke'], axis = 1,inplace=True)
    X = dataset.values
    total_size = X.shape[0]
    #normalized dataset features
    sc1 = MinMaxScaler(feature_range = (0, 1))
    X = sc1.fit_transform(X)#features normalization
    #now apply Smote to balance dataset
    sm = SMOTE()
    X, Y = sm.fit_resample(X, Y)
    #now select features uisng CHI Square features selection algorithm
    print("Features available in dataset before selection : "+str(X.shape[1]))
    features_selector = SelectKBest(score_func=chi2, k = 9)
    selected_features = features_selector.fit_transform(X, Y)
    print("Features available in dataset after selection : "+str(selected_features.shape[1]))
    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(selected_features, Y, test_size=0.2) #split dataset into train and test
    print()
    print("Dataset train & test split as 80% dataset for training and 20% for testing")
    print("Training Size (80%): "+str(X_train.shape[0])) #print training and test size
    print("Testing Size (20%): "+str(X_test.shape[0]))
    training_size = X_train.shape[0]
    testing_size = X_test.shape[0]
    
    return render_template("preprocess.html", 
                           total_size=total_size,
                           training_size=training_size, 
                           testing_size=testing_size,
                           train_percentage=80, 
                           test_percentage=20)

def calculateMetrics(algorithm, predict, testY):
    p = round(precision_score(testY, predict,average='macro') * 100,2)
    r = round(recall_score(testY, predict,average='macro') * 100,2)
    f = round(f1_score(testY, predict,average='macro') * 100,2)
    a = round(accuracy_score(testY,predict)*100,2)
    print()
    print(algorithm+' Accuracy  : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' FMeasure    : '+str(f))    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    return a,p,r,f



@app.route('/existing_alg')
def existing_algorithm():
    
    #now train SVM algorithm
    svm_cls = svm.SVC()#define SVM object
    svm_cls.fit(X_train, y_train)#train SVM on training data
    predict = svm_cls.predict(X_test)#perform prediction on test data
    
    # Pass metrics to the template
    # Get accuracy metrics from calculateMetrics
    accuracy,precision,recall,fscore = calculateMetrics("SVM", predict, y_test)
    
    # Pass metrics to the template
    return render_template("existing_alg.html", 
                           accuracy=accuracy, 
                           precision=precision, 
                           recall=recall, 
                           fscore=fscore  
                           )



@app.route('/proposed_alg')
def proposed_algorithm():
    
    
    #train random forest algorithm on training dataset and test its prediction capability on test data
    #now train Random Forest algorithm
    rf_cls = RandomForestClassifier()
    rf_cls.fit(X_train, y_train)
    predict = rf_cls.predict(X_test)
    
    # Get accuracy metrics from calculateMetrics
    accuracy,precision,recall,fscore = calculateMetrics("Random Forest", predict, y_test)
    
    # Pass metrics to the template
    return render_template("proposed_alg.html", 
                           accuracy=accuracy, 
                           precision=precision, 
                           recall=recall, 
                           fscore=fscore  
                           )

@app.route('/extension_alg')
def extension_algorithm():
    global cb_cls
    #now train extension CATBOOST algorithm as extension which is more advanced then other ML algorithm
    cb_cls = cb.CatBoostClassifier(iterations=300, learning_rate=0.1)
    cb_cls.fit(X_train, y_train)#train CatBoost on training data
    predict = cb_cls.predict(X_test)#perform prediction on test data
    # Get accuracy metrics from calculateMetrics
    accuracy,precision,recall,fscore =calculateMetrics("Extension CatBoost", predict, y_test)#calculate accuracy and other metrics

    # Pass metrics to the template
    return render_template("extension_alg.html", 
                           accuracy=accuracy, 
                           precision=precision, 
                           recall=recall, 
                           fscore=fscore  
                           )


@app.route('/graph')
def display_graph():
    df = pd.DataFrame([
                   ['SVM','Precision',precision[0]],
                   ['SVM','Recall',recall[0]],
                   ['SVM','F1 Score',fscore[0]],
                   ['SVM','Accuracy',accuracy[0]],
                   ['Propose RF','Precision',precision[1]],
                   ['Propose RF','Recall',recall[1]],
                   ['Propose RF','F1 Score',fscore[1]],
                   ['Propose RF','Accuracy',accuracy[1]],
                   ['Extension CatBoost','Precision',precision[2]],
                   ['Extension CatBoost','Recall',recall[2]],
                   ['Extension CatBoost','F1 Score',fscore[2]],
                   ['Extension CatBoost','Accuracy',accuracy[2]],
                  ],columns=['Parameters','Algorithms','Value'])
    
    # Create the bar graph
    fig, ax = plt.subplots(figsize=(5, 3))  # Increase the figure size (10x6 inches)
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', ax=ax)

    # Set the title and labels
    ax.set_title("Algorithms Performance Comparison", fontsize=6)
    ax.set_xlabel("Metrics", fontsize=4)
    ax.set_ylabel("Values", fontsize=4)
    
    # Adjust tick labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(fontsize=4)

    # Move the legend outside the plot
    plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=4)

    # Save the graph as an image with high DPI for clarity
    graph_path = os.path.join(app.static_folder, 'graph.png')
    plt.tight_layout()  # Ensure everything fits well
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')  # Save the plot with 300 DPI for higher resolution
    plt.close()  # Close the plot to free memory

    # Render the HTML template and pass the image path
    return render_template("graph.html", graph_url='/static/graph.png')



@app.route('/predict')
def upload():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def upload_file():
    global df,extension_model
    if 'testdata' not in request.files:
        message = 'No file selected'
        return render_template('predict.html', message=message)

    dataset = request.files['testdata']

    if dataset.filename == '':
        message = 'No selected file'
        return render_template('upload.html', message=message)

    if dataset and allowed_file(dataset.filename):
        filename = secure_filename(dataset.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        dataset.save(filepath)
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            testData = pd.read_csv(filepath)#reading test data
            #testData = testData.values
            testData.fillna(0, inplace = True)
            temp = testData.values
            testData['gender'] = pd.Series(encoder1.transform(testData['gender'].astype(str)))#converting non-numeric data to numeric
            testData['ever_married'] = pd.Series(encoder2.transform(testData['ever_married'].astype(str)))
            testData['work_type'] = pd.Series(encoder3.transform(testData['work_type'].astype(str)))
            testData['Residence_type'] = pd.Series(encoder4.transform(testData['Residence_type'].astype(str)))
            testData['smoking_status'] = pd.Series(encoder5.transform(testData['smoking_status'].astype(str)))
            testData.drop(['id'], axis=1, inplace=True)#drop id column
            testData = testData.values
            test = sc1.transform(testData)#normalizing values
            test = features_selector.transform(test)#select relevant features using CHI2 selector
            predict = cb_cls.predict(test)#performing prediction on test data using XGBOOST 
            for i in range(len(predict)):
                print("Test Data = "+str(temp[i])+" Predicted As ====> "+labels[predict[i]]+"\n")

            results = []
            # Collect the prediction results
            for i in range(len(predict)):
                # Formatting the test data to make it readable
                formatted_test_data = ", ".join([str(value) for value in testData[i]])

                # Appending to the results list
                results.append({
                    'test_data': str(temp[i]), 
                    'predicted_performance': labels[predict[i]]
                })

            return render_template('predict.html', results=results)
        except Exception as e:
            message = f"Error processing file: {e}"
        return render_template('predict.html', message=message)
    else:
        message = 'Allowed file types: .csv'
        return render_template('predict.html', message=message)



@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)