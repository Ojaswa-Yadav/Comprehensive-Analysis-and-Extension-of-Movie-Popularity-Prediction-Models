# **Comprehensive Analysis and Extension of Movie Popularity Prediction Models**

## **Overview**

This project investigates the prediction of movie popularity using both conventional movie attributes and social media metrics. The study extends the analysis of the **CSM (Conventional and Social Media Movies) Dataset** for 2014 and 2015, providing insights into movie ratings and gross income prediction.

By utilizing machine learning models and feature engineering techniques, the project explores how conventional and social media features contribute to the prediction of movie popularity.

---

## **Objectives**

1. Predict **movie ratings**.
2. Predict **movie gross income**.

---

## **Dataset**

The **CSM Dataset**, curated by Mehreen Ahmed, integrates conventional features (e.g., genre, budget) with social media metrics (e.g., likes, comments) from IMDb, Twitter, and YouTube.

### **Features**
- **Conventional:**
  - Genre, Gross, Rating, Budget, Screens, Sequel
- **Social Media:**
  - Sentiment, Views, Likes, Dislikes, Comments, Aggregate Followers

### **Dataset Statistics**
- **Number of Movies:** 232

---

## **Data Preprocessing**

1. **General Preprocessing:**
   - Removed redundant data.
   - Filled missing values (numeric with median, categorical with mode).
   - Normalized features.

2. **Categorical Encoding:**
   - One-hot encoded `Genre` and `Sequel`.

3. **Feature Integration:**
   - Combined social and conventional data for better predictive performance.

4. **Target Variable Preparation:**
   - Encoded `GrossLabel` and `RatingLabel` into categorical labels.

5. **Train-Test Split:**
   - Dataset split into 80% training and 20% test.

---

## **Target Variable Categories**

### **Rating Labels**
- **Poor:** ≤ 5.0
- **Average:** 5.0–6.4
- **Good:** 6.4–7.4
- **Excellent:** 7.4–10

### **Gross Income Labels**
- **Flop:** < $900,000
- **Average:** $900,000–$19,999,999
- **Success:** $20,000,000–$99,999,999
- **Blockbuster:** ≥ $100,000,000

---

## **Machine Learning Methods**

The following models were implemented to predict movie popularity:

1. **Naive Bayes**
2. **Neural Networks**
3. **Support Vector Machine (SVM)**:
   - Linear Kernel
   - RBF Kernel
4. **Quadratic Discriminant Analysis (QDA)**
5. **Linear Discriminant Analysis (LDA)**
6. **Decision Trees**
   - Standard
   - Pruned
7. **Random Forests**
8. **Linear Regression**

---

## **Results**

### **Gross Income Prediction**
- **Best Classifiers:**
  - Tree-based approaches (e.g., Decision Trees, Random Forests).
  - Linear Regression also performed well.
- **Feature Observations:**
  - Conventional features outperformed Social Media features.
  - Combined features did not outperform the conventional set alone.

### **Rating Prediction**
- **Best Classifiers:**
  - Tree-based approaches, Neural Networks, and Naive Bayes.
- **Feature Observations:**
  - Social Media features were slightly better at detecting ratings.
  - Combining both feature sets improved accuracy slightly.

---

## **Project Structure**

```plaintext
movie-popularity-prediction/
├── data/                       # Dataset files
│   └── csm_dataset.csv         # CSM dataset for 2014 and 2015
├── Processing and EAda/                 
│   ├── split_data.R            # Data preprocessing and feature engineering
│   ├── visualise.R             # Exploratory data analysis and visualisation
├── Neural_Network/                  
│   ├── ann.py                  # Neural Network Implemetation
├── trad_ml_scripts/                  
│   ├── naive_bayes.r           # Naive Bayes implementation
│   ├── svm.r                   # SVM implementation
    ├── lda.r                   # lda implementation
    ├── qda.r                   # qda implementation
    ├── Linear Regression.r      # linear regression implementation
├── trees/
│   ├── decision_tree.py         # Decision Tree implementation
│   └── random_forest.py         # Random Forest implementation
│   └── comparison_metrics.csv   # Model performance comparison
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
