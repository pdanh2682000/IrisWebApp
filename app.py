from models import model_logistic # Import the python file containing the ML model
from models import model_decisionTree
from models import model_KNN
from models import model_Bayes
from models import model_SVM
from flask import Flask, request, render_template # Import flask libraries

# Initialize the flask class and specify the templates directory
app = Flask(__name__,template_folder="templates")

# Default route set as 'home'
@app.route('/')
def home():
    return render_template('home.html') # Render home.html


# Route 'classify' accepts GET request
@app.route('/classify',methods=['GET'])
def classify_type():
    try:
        sepal_len = request.args.get('slen') # Get parameters for sepal length
        sepal_wid = request.args.get('swid') # Get parameters for sepal width
        petal_len = request.args.get('plen') # Get parameters for petal length
        petal_wid = request.args.get('pwid') # Get parameters for petal width

        # Get the output from the classification model
        variety_logistic = model_logistic.classify(sepal_len, sepal_wid, petal_len, petal_wid)
        variety_decisionTree = model_decisionTree.classify(sepal_len, sepal_wid, petal_len, petal_wid)
        variety_KNN = model_KNN.classify(sepal_len, sepal_wid, petal_len, petal_wid)
        variety_Bayes = model_Bayes.classify(sepal_len, sepal_wid, petal_len, petal_wid)
        variety_SVM = model_SVM.classify(sepal_len, sepal_wid, petal_len, petal_wid)
        # Render the output in new HTML page
        return render_template('output.html', variety_logistic=variety_logistic,
         variety_decisionTree=variety_decisionTree, variety_KNN=variety_KNN, 
         variety_Bayes=variety_Bayes, variety_SVM=variety_SVM)
    except:
        return 'Error'

# Run the Flask server
if(__name__=='__main__'):
    app.run(debug=True)