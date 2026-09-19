<h1>Predicting Loan Defaults With Machine Learning Models</h1>

 ### [View Full Report](https://github.com/jcastro712/CustomerSatisfaction/blob/main/Predicting%20Customer%20Satisfaction%20Short%20Version.pdf)

 ### [View R Code](https://github.com/jcastro712/LoanDefaultPrediction/blob/main/Loan%20Default%20Code.R)

<h2>Description</h2>
This project uses machine learning in R to predict whether a borrower will default on a loan. Using a dataset of 255,347 borrowers with 18 variables covering demographics, employment, credit history, and loan terms, the analysis trains and compares Logistic Regression, Random Forest, and XGBoost, then tunes the best-performing model with a grid search before evaluating all four models on an untouched holdout sample.
<br />

<h2>Languages and Utilities Used</h2>

- <b>R</b>
- <b>tidyverse</b> (data cleaning and wrangling)
- <b>randomForest</b>
- <b>xgboost</b>
- <b>broom</b> (extracting logistic regression coefficients)
- <b>foreach / doSNOW</b> (parallelized grid search for hyperparameter tuning)
- <b>pROC</b> (AUC calculation)
- <b>scales / ggplot2</b> (plot formatting and visualization)

<h2>Business Problem</h2>

Banks and other lenders need to know how likely a borrower is to default before extending credit. Misjudging that risk is costly in both directions: approving a loan that defaults results in a direct loss, while rejecting a creditworthy borrower means lost business. This project frames loan default as a binary classification problem: given a borrower's demographic, employment, credit, and loan information, predict whether they will default. Rising consumer debt and delinquency rates make this a timely problem for lenders looking to manage risk more precisely.

<h2>Data</h2>

<b>Source:</b> Loan Default Dataset from Kaggle, originally used in Coursera's Data Science Coding Challenge: Loan Default Prediction

<b>Size:</b> 255,347 observations, 18 variables, including borrower demographics (age, education, marital status), financial details (income, credit score, debt-to-income ratio), and loan characteristics (amount, term, interest rate, purpose)

<b>Class balance:</b> around 88% of borrowers did not default, versus 12% who did, a meaningful imbalance that makes accuracy alone a misleading metric


<b>Preparation:</b>
- Removed <code>LoanID</code>, which is an identifier with no predictive value
- Converted Yes/No fields (<code>HasMortgage</code>, <code>HasDependents</code>, <code>HasCoSigner</code>) into binary 0/1 values
- Converted categorical fields (<code>Education</code>, <code>EmploymentType</code>, <code>MaritalStatus</code>, <code>LoanPurpose</code>) into factors for Logistic Regression and Random Forest, and into one-hot encoded dummy columns for XGBoost, since it requires fully numeric input
- Set aside 20% of the data as a holdout sample before any training or tuning; split the remaining 80% into training (80%) and test (20%) sets, checking that the default rate stayed consistent across all splits
- Used a fixed random seed throughout so the splits are reproducible

<h2>Results</h2>

Performance on the holdout set:
| Model | Accuracy | Precision | Recall | AUC |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.795 | 0.810 | 0.959 | 0.676 |
| Random Forest | 0.831 | 0.842 | 0.962 | 0.755 |
| XGBoost | 0.814 | 0.826 | 0.962 | 0.720 |

Random Forest was the strongest model on every metric, with 83% accuracy and the highest AUC (0.755). All three models achieved high recall (~96%), meaning they rarely missed a genuinely positive review, but differed more in precision and AUC (how well they separated the two classes overall).

<p align="center"> <br/> 
<img src="images/Figure_1.png" height="50%" width="50%" alt="Distribution of Customer Satisfaction"/> <br/>
Most customers left positive reviews, indicating overall high satisfaction
<br /> 
<br /> 
<img src="images/Figure_2.png" height="50%" width="50%" alt="Customer Satisfaction vs Delivery Delay"/> <br/> 
Delayed orders were consistently associated with lower review scores
<br /> 
<br /> 
<img src="images/Figure_3.png" height="50%" width="50%" alt="Logit Model - Top 10 Most Important Variables"/> <br/> 
Logistic Regression ranks delivery delay first by coefficient magnitude, with customer state also playing a role 
<br /> 
<br /> 
<img src="images/Figure_4.png" height="50%" width="50%" alt="Random Forest - Top 10 Most Important Variables"/> <br/> 
Random Forest highlights a broader set of drivers, including total payment, price, and product weight
<br /> 
<br /> 
<img src="images/Figure_5.png" height="50%" width="50%" alt="XGBoost - Top 10 Most Important Variables"/> <br/>
XGBoost also places delivery delay at the top, while emphasizing payment amount, price, and approval time
</p>

Across all three models, <b>delivery delay</b> was consistently the single most important predictor of a positive or negative review, though the tree-based models (Random Forest and XGBoost) surfaced additional signal from payment totals, product price, and product weight that the linear logistic model captured less clearly.

<h2>Next Steps and Limitations</h2>

<b>Low recall across all models:</b> even the best model only catches about 7% of true defaulters, so before use in a real lending pipeline the classification threshold should be tuned (rather than left at the default 0.5) or class weighting applied to better balance precision against recall

<b>Random search vs. grid search:</b> tuning used a full grid search across three parameters; a random search would explore a wider parameter space at similar computational cost and might find a better combination

<b>No behavioral or history data:</b> the dataset captures a single snapshot of each borrower rather than repayment history over time, adding longitudinal or repeat-borrower data could improve the model's ability to flag risk early

<b>Model is not standalone:</b> as noted in the report, the model should support, not replace, human underwriting judgment, since it doesn't capture every factor a loan officer would consider
