# 🎓 Student Performance Prediction Using Machine Learning

## 1. Project Overview

The **Student Performance Prediction** project is a Machine Learning-based web application that predicts a student's academic performance using various student-related factors.

This project uses **Linear Regression**, a supervised Machine Learning algorithm, to predict the Performance Index based on study hours, previous scores, extracurricular activities, sleep hours, and sample question papers practiced.

The application is developed using Python, Flask, Pandas, NumPy, and Scikit-learn. Users can enter their academic details through a web interface and receive an instant predicted Performance Index.

### Objectives

- Predict student academic performance using Machine Learning.
- Understand the relationship between study habits and student performance.
- Apply Linear Regression for prediction.
- Build an interactive web application using Flask.
- Evaluate model performance using R² Score and Mean Squared Error.

## 2. Dataset Information

The project uses the **Student_Performance.csv** dataset.

- Total Records: 10,000
- Total Columns: 6
- Problem Type: Supervised Learning
- Machine Learning Task: Regression
- Target Variable: Performance Index

### Dataset Features

| Column Name | Description | Variable Type |
|---|---|---|
| Hours Studied | Number of hours spent studying | Independent |
| Previous Scores | Student's previous examination scores | Independent |
| Extracurricular Activities | Participation in extracurricular activities (Yes/No) | Independent |
| Sleep Hours | Number of hours slept | Independent |
| Sample Question Papers Practiced | Number of sample papers practiced | Independent |
| Performance Index | Student's performance score | Dependent |

### Independent and Dependent Variables

**Independent Variables (X):**
- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Question Papers Practiced

**Dependent Variable (Y):**
- Performance Index

### Dataset Preprocessing

- Loaded the dataset using Pandas.
- Separated independent and dependent variables.
- Converted categorical values into numerical values using `pd.get_dummies()`.
- Used `drop_first=True` to avoid redundant dummy columns.
- Divided the dataset into training and testing sets.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning |
| Linear Regression | Prediction algorithm |
| Flask | Web application development |
| HTML | Frontend structure |
| CSS | Frontend styling |
| Pickle | Saving trained ML model |
| PyCharm | Development environment |
| GitHub | Version control and project hosting |

## 4. Installation Steps

Follow these steps to run the project on your local machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Student-Performance-Prediction.git
```

### Step 2: Navigate to Project Folder

```bash
cd Student-Performance-Prediction
```

### Step 3: Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn flask
```

Or install using requirements.txt:

```bash
pip install -r requirements.txt
```

### Step 5: Train the Machine Learning Model

Run the following command:

```bash
python main.py
```

This will:
- Load the dataset.
- Preprocess the data.
- Split training and testing data.
- Train the Linear Regression model.
- Evaluate the model.
- Save the trained model as `Std.Pred.pkl`.

### Step 6: Run Flask Application

```bash
python app.py
```

### Step 7: Open the Application

Open your browser and visit:

```text
http://127.0.0.1:5000/
```

Enter the student details and click the Predict Performance button to get the predicted Performance Index.

## 5. Project Structure

```text
Student-Performance-Prediction/
│
├── Student_Performance.csv
│
├── main.py
│
├── app.py
│
├── Std.Pred.pkl
│
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── screenshots/
│   ├── home_page.png
│   ├── input_page.png
│   └── prediction_result.png
│
└── README.md
```

### File Descriptions

- `Student_Performance.csv` – Dataset used for model training.
- `main.py` – Data preprocessing, model training, testing, evaluation, and model saving.
- `app.py` – Flask backend for handling user inputs and predictions.
- `Std.Pred.pkl` – Saved Linear Regression model.
- `templates/index.html` – Frontend webpage.
- `requirements.txt` – Required Python libraries.
- `screenshots/` – Project screenshots.
- `README.md` – Project documentation.

## 6. Model Training Process

The project uses **Linear Regression** to predict the Performance Index.

### Step 1: Dataset Loading

The dataset is loaded using Pandas.

```python
self.df = pd.read_csv(path)
```

### Step 2: Feature Selection

Independent and dependent variables are separated.

```python
self.X = self.df.iloc[:, :-1]
self.Y = self.df.iloc[:, -1]
```

### Step 3: Encoding Categorical Data

Convert categorical columns into numerical dummy variables.

```python
self.X = pd.get_dummies(
    self.X,
    drop_first=True,
    dtype=int
)
```

### Step 4: Train-Test Split

The dataset is divided into 80% training data and 20% testing data.

```python
train_test_split(
    self.X,
    self.Y,
    test_size=0.2,
    random_state=42
)
```

### Step 5: Model Training

Linear Regression is used to train the model.

```python
self.reg = LinearRegression()

self.reg.fit(
    self.X_train,
    self.y_train
)
```

### Step 6: Model Evaluation

The model is evaluated using:

- R² Score
- Mean Squared Error (MSE)

### Step 7: Model Saving

The trained model is saved using Pickle.

```python
with open("Std.Pred.pkl", "wb") as f:
    pickle.dump(self.reg, f)
```

## 7. Model Evaluation

The model performance is evaluated using training and testing datasets.

### Evaluation Metrics

**R² Score:**

Measures how well the model explains the variation in the target variable.

**Mean Squared Error (MSE):**

Measures the average squared difference between actual and predicted values. Lower MSE indicates smaller prediction errors.

### Results

| Evaluation Metric | Training Data | Testing Data |
|---|---|---|
| R² Score | Add actual value | Add actual value |
| Mean Squared Error | Add actual value | Add actual value |

*Note: Update the table with the actual values displayed when you execute main.py. Results are not hardcoded.*

## 8. Web Application Screenshots

The Flask application provides an interactive interface for predicting student performance.

### 1. Home Page

The home page displays the Student Performance Prediction application and its features.

![Home Page](screenshots/home_page.png)

### 2. Student Input Page

Users can enter their study hours, previous scores, extracurricular activities, sleep hours, and sample question papers practiced.

![Student Input Page](screenshots/input_page.png)

### 3. Prediction Result

After submitting the student details, the application displays the predicted Performance Index.

![Prediction Result](screenshots/prediction_result.png)

*Add your actual project screenshots to the screenshots folder using the filenames shown above.*

## 9. Deployment Instructions

The project can be deployed using platforms such as Render.

### Step 1: Prepare requirements.txt

Create a file named `requirements.txt`:

```text
numpy
pandas
matplotlib
scikit-learn
flask
gunicorn
```

### Step 2: Train the Model

Run:

```bash
python main.py
```

Ensure that `Std.Pred.pkl` is generated.

### Step 3: Upload Project to GitHub

```bash
git init

git add .

git commit -m "Student Performance Prediction Project"

git branch -M main

git remote add origin https://github.com/your-username/Student-Performance-Prediction.git

git push -u origin main
```

### Step 4: Deploy on Render

1. Create an account on Render.
2. Select New Web Service.
3. Connect your GitHub repository.
4. Select the Python environment.
5. Set the Build Command:

```bash
pip install -r requirements.txt
```

6. Set the Start Command:

```bash
gunicorn app:app
```

7. Click Deploy.

Make sure `Student_Performance.csv` and the trained `Std.Pred.pkl` are included in the deployed project.

For production deployment, use Gunicorn instead of Flask's development server and disable debug mode.

## 10. Project Results

The Student Performance Prediction application successfully:

- Loads and preprocesses student performance data.
- Trains a Linear Regression model.
- Predicts student Performance Index values.
- Evaluates the model using R² Score and MSE.
- Saves the trained model using Pickle.
- Provides an interactive web interface using Flask.

## 11. Future Enhancements

- Add more Machine Learning algorithms.
- Improve prediction performance through hyperparameter tuning.
- Add student performance visualizations.
- Create an interactive analytics dashboard.
- Store prediction history in a database.
- Improve the application with additional student-related features.

## 12. Conclusion

The Student Performance Prediction project demonstrates how Machine Learning can be used to estimate academic performance based on student-related factors.

By combining Python, Scikit-learn, Linear Regression, and Flask, this project provides an easy-to-use web application for predicting student performance.

It also provides practical experience in data preprocessing, model training, evaluation, and web application development.

## 13. Author

**Syed Ashraf Pasha**

Machine Learning | Data Science | Data Analytics

**Project:** Student Performance Prediction Using Machine Learning

