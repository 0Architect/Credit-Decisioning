import shap
import pandas as pd

shap.initjs()
shap_values = [0.15768986,  0.02032858,  0.13347578,  0.02682582, -0.01408261, -0.00053297,
 -0.02868272, -0.04470266, -0.00537616,  0.00331742]
expected_value = 0.49991051893658633
X = pd.DataFrame([{'RevolvingUtilizationOfUnsecuredLines': 0.77, 'age': 45, 'NumberOfTime30-59DaysPastDueNotWorse': 2, 
                   'DebtRatio': 0.8, 'MonthlyIncome': 9120, 'NumberOfOpenCreditLinesAndLoans': 13, 
                   'NumberOfTimes90DaysLate': 0, 'NumberRealEstateLoansOrLines': 6, 
                   'NumberOfTime60-89DaysPastDueNotWorse': 0, 'NumberOfDependents': 2}])

shap.summary_plot(shap_values, X)
# shap.force_plot(expected_value, shap_values, X, matplotlib=True, show=False)

# import xgboost
# from IPython.display import display

# import shap

# # load JS visualization code to notebook
# shap.initjs()

# # train XGBoost model
# X, y = shap.datasets.california(n_points=1_000)
# bst = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# # explain the model's predictions using SHAP values
# explainer = shap.TreeExplainer(bst)
# explanation = explainer(X)

# shap.plots.force(explanation[0])