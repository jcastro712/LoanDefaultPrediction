<h1>Predicting Loan Defaults With Machine Learning Models</h1>

 ### [View Full Report](https://github.com/jcastro712/LoanDefaultPrediction/blob/main/Loan%20Default%20Prediction.pdf)

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

Because predicted probabilities (not just 0/1 classifications) matter for a risk model, Mean Squared Error (MSE) was used as the primary comparison metric, alongside accuracy, precision, recall, and AUC. Results on the test set:


| Model | MSE | Accuracy | Precision | Recall | AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.09267 | 0.884 | 0.652 | 0.033 | 0.515 |
| Random Forest | 0.09384 | 0.884 | 0.578 | 0.053 | 0.524 |
| Untuned XGBoost | 0.09222 | 0.884 | 0.651 | 0.039 | 0.518 |
| Tuned XGBoost | 0.09157 | 0.885 | 0.622 | 0.066 | 0.530 |

XGBoost outperformed both baseline models even before tuning, and a grid search over <code>max_depth</code>, <code>lambda</code>, and <code>eta</code> (with early stopping on <code>nrounds</code>) improved it further, landing on <code>max_depth</code> = 2, <code>nrounds</code> = 385, <code>lambda</code> = 1, </code>eta = 0.15. The tuned model was then re-evaluated on the untouched holdout sample, where it again led on MSE, accuracy, recall, and AUC, confirming the improvement wasn't just an artifact of the test split:

| Model | MSE | Accuracy | Precision | Recall | AUC |
| --- | --- | --- | --- | --- | --- |
| Tuned XGBoost | 0.09087 | 0.887 | 0.635 | 0.069 | 0.532 |
| Untuned XGBoost | 0.09146 | 0.886 | 0.673 | 0.043 | 0.520 |
| Logistic Regression | 0.09212 | 0.886 | 0.618 | 0.037 | 0.517 | 
| Random Forest | 0.09339 | 0.886 | 0.598 | 0.054 | 0.525 |


Recall stayed low across every model, a reflection of how rare and hard-to-catch defaults are in this dataset, but the tuned XGBoost model consistently identified more true defaulters than the alternatives, which matters more than raw accuracy in a risk-screening context.

<p align="center"> 
About 12% of borrowers in the dataset defaulted, a meaningful class imbalance: <br/> 
<img src="loan_images/figure1_class_balance.png" height="50%" width="50%" alt="Default Rate (Class Balance)"/> 
<br /> 
<br /> 
Defaulters skewed toward lower credit scores, though the two groups overlap heavily: <br/> 
<img src="loan_images/figure2_credit_score.png" height="50%" width="50%" alt="Credit Score by Default Status"/> 
<br /> 
<br /> 
Defaulters also tended to have lower income on average: <br/> 
<img src="loan_images/figure3_income.png" height="50%" width="50%" alt="Income by Default Status"/> 
<br /> 
<br /> 
Higher interest rates were associated with a higher likelihood of default: <br/> 
<img src="loan_images/figure4_interest_rate.png" height="50%" width="50%" alt="Interest Rate by Default Status"/> 
<br /> 
<br /> 
Logistic Regression ranks age and interest rate as the strongest predictors: <br/> 
<img src="loan_images/figure5_logit_importance.png" height="50%" width="50%" alt="Logit Model - Most Important Variables"/> 
<br /> 
<br /> 
Random Forest puts income, interest rate, and loan amount at the top, and gives credit score more weight than the logit model does: <br/> 
<img src="loan_images/figure6_rf_importance.png" height="50%" width="50%" alt="Random Forest - Most Important Variables"/> 
<br /> 
<br /> 
XGBoost concentrates most of its predictive power in five variables — age, interest rate, income, loan amount, and months employed: <br/> 
<img src="loan_images/figure7_xgb_importance.png" height="50%" width="50%" alt="XGBoost - Most Important Variables"/> 
<br /> 
<br /> 
The tuned XGBoost model relies on largely the same top variables, with credit score and number of credit lines gaining a bit more relative weight: <br/> 
<img src="loan_images/figure8_tuned_xgb_importance.png" height="50%" width="50%" alt="Tuned XGBoost - Most Important Variables"/> </p>

<h2>Next Steps and Limitations</h2>

<b>Low recall across all models:</b> even the best model only catches about 7% of true defaulters, so before use in a real lending pipeline the classification threshold should be tuned (rather than left at the default 0.5) or class weighting applied to better balance precision against recall

<b>Random search vs. grid search:</b> tuning used a full grid search across three parameters; a random search would explore a wider parameter space at similar computational cost and might find a better combination

<b>No behavioral or history data:</b> the dataset captures a single snapshot of each borrower rather than repayment history over time, adding longitudinal or repeat-borrower data could improve the model's ability to flag risk early

<b>Model is not standalone:</b> as noted in the report, the model should support, not replace, human underwriting judgment, since it doesn't capture every factor a loan officer would consider
