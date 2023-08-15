# Cross Sell Ranking

This repository contains code and documentation for a data science project. The main goal of this project is to generate an orderly list of customers with a propensity to buy a secondary product of the company. In this case, it is sought to predict health insurance owners interested in vehicle insurance. All the data used was provided by Kaggle.

# 1. Business Problem

Cross-selling involves offering supplementary products or services to customers based on their current purchase or interests, aiming to deepen business-customer relations and boost overall sales. Examples include a bank proposing a credit card to a new account holder or an online store suggesting a phone case to a smartphone buyer. Striking the right balance is crucial; over-promotion or irrelevant suggestions can erode trust. 

# 2. Business Assumptions


The main assumption considered in this project was that prospecting customers were only possible for 20% of the test set, that is 15244 customers.



# 3. Solution Strategy

### Step 1: Data Description

The main objective of this section was to organize and understand raw data using descriptive statistics.

### Step 2: Feature Engineering

In this step, the hypotheses were defined to help see if creating new features was possible.


### Step 3: Variable Filtering

This step was responsible for dropping IDs information since we didn't use them.

### Step 4: Exploratory Data Analysis

Three different analyses were done in this step: univariate, bivariate, and multivariate analyses. Moreover, this section was carried out to evaluate the hypotheses and get insights into the variables' importance.


### Step 5: Data Preparation

In this section, all the data was prepared with encoding, transformation, and scaling to perform properly in a machine learning algorithm.


### Step 6: Feature Selection

In this section, a manual selection was used to choose the best features. They were chosen based on insights gotten previously.


### Step 7: Machine Learning Modelling

The approach chosen to solve this business problem was a raking-to-learn algorithm. Moreover, four models were trained and evaluated: Logistic Regression, KNN, Random Forest, and XGBoost.


### Step 8: Hyperparameter Fine Tunning

Since XGBoost seemed to be the most promising, a Random Search fine-tuning algorithm was applied.


### Step 9: Results Interpretation

First, the generalization level was evaluated based on the test set, after that the final cumulative gain curve and lift curve were generated.


### Step 10: Deployment


In this project, a local API was coded to be used to predict remotely.



# 4. Top 3 Data Insights

**H5.** On average, people who have damaged their car tend to be more interested in car insurance.

**True:** There is a slight positive correlation between variables.


**H6.** Customers tend to lose interest when the cost of insurance increases.

**False:** There is no relation between those variables.

**H3.** There is a positive correlation between the car's age and the insurance cost.

**True:** The older the vehicle, the more likely is the owner to be interested in the insurance.



# 5. Machine Learning Model Applied

In this project, four machine learning models were tested: Logistic Regression, KNN, Random Forest, and XGBoost. The final model was built using XGBoost.


# 6. Machine Learning Model Performance

The final model was built with a precision at 15244 of 34.10%, and an average precision at 15244 of 36.56%.


# 7. Business Results

The algorithm was capable of ranking interested customers in which way that approaching 20% of the database results in around 55% of the interested customers being approached, which represents around 2.75 times more efficiency than a random approach.


# 8. Conclusions

In conclusion, Learning to Rank algorithms have emerged as a vital tool in modern information systems, significantly enhancing the organization and presentation of relevant information across various domains. While they require substantial labeled data and can be complex to implement, their adaptability and precision make them invaluable in solving business problems.

# 9. Next Steps to Improve

The next improvement to be sought will be to develop a neural network that will do the predictions. Moreover, it is sought to use an embedding approach to enhance the algorithm.
