# Pass-Class

For our classifier, we chose to build a logistic regression model trained on 700,000 (password, strength) pairs obtained via Kaggle. More information on how the training data was collected and classified can be found <a href="https://www.kaggle.com/bhavikbb/password-strength-classifier-dataset"> here </a>.

We tried using several other classification models (random forest, svm, and neural nets) that often give higher accuracy at the cost of longer training and prediction time. Ultimately, the improvements in accuracy weren't worth the expense, and we settled on a simpler logistic regression model that gives 82% test accuracy. 

We drew on prior knowledge of ML and data science from MATH154 - Computational Statistics and MATH158 - Statistical Linear Models to complete this assignment. These classes use R and the RStudio IDE, so this assignment marks our first experience using Python libraries for data cleaning and model building. The transition was fairly smooth, and we made quick progress with the help of library documentation and online beginner tutorials. 

## Dependencies
Given our newness to python ML libraries and dependencies, we may not perfectly describe the setup process but will try our best. 

Our program imports the following libraries:
<ul>
    <li>NumPy</li>
    <li>Pandas</li>
    <li>joblib</li>
    <li>sklearn</li>
</ul>

Before running the program, macOS users should follow these steps (double-checked on a clean Mac):
<ul>
    <li>Install python3: <code>brew install python</code></li>
    <li>Install NumPy: <code>pip install numpy</code></li>
    <li>Install Pandas: <code>pip install pandas</code> </li>
    <li>Install sklearn: <code>pip install -U scikit-learn</code></li>
</ul>

## Running Classifier
Steps for running our program are simple: 
<ol>
    <li>cd to directory containing classify.py</li>
    <li>Type <code>python classify.py</code> and hit enter</li>
    <li>Give the program a few moments to read data and build the model </li>
    <li>When prompted for input, type the password you want to classify and hit enter</li>
</ol>
