import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from category_encoders import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

from hyperopt import Trials, tpe, hp, fmin, space_eval
from hyperopt.pyll import scope

import shap
# import data
# import data
project_path=os.getcwd()
df2= os.path.join(project_path,"data/dataset2.csv")
df2 = pd.read_csv(df2)
df_type='df2'
df2.columns=['Cmix', 'wc_u','wc_std', 'Vp_u','Pv_std','Yd_u','Yd_std', # Table 1
             'ξg_u','ξg_std','Xog_u','Xog_std','ξp_u','ξp_std','Xop_u','Xop_std','dp_u','dp_std','dpmax_u','dpmax_std','po_u','po_std', # Tabl2
             'φ','h','m','ρ','Fmax','CS']
df2.describe().T.to_csv(f'results/{df_type}_statSumm.csv')

# Make a copy of the dataframe
df2_corr = df2.copy()
df2_corr['Cmix'] = pd.factorize(df2_corr['Cmix'])[0]

# Create correlation matrix
df2_corr = df2_corr.corr().round(4)
df2_corr.to_csv(f'data/{df_type}_corrSummary.csv')

# Make a copy of the dataframe
df_encoded = df2.copy()

# Initialize OneHotEncoder
encoder = OneHotEncoder(cols=['Cmix'], use_cat_names=True)

# Fit and transform the DataFrame
df_encoded = encoder.fit_transform(df_encoded)

# Concatenate the original DataFrame with the encoded DataFrame
df_encoded['Cmix']=df2['Cmix']

Input_features = df_encoded.columns.drop('CS')
X = df_encoded[Input_features]
y = df_encoded['CS']


# Splitting the data into training, temporary test, and validation sets (80%, 10%, 10%) with stratified split based on 'Cmix_merge'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2, stratify=X['Cmix'])
X_train_wo_valid, X_val, y_train_wo_valid, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=2, stratify=X_train['Cmix'])


X_train = X_train.drop(columns=['Cmix'])
X_train_wo_valid = X_train_wo_valid.drop(columns=['Cmix'])
X_test = X_test.drop(columns=['Cmix'])
X_val = X_val.drop(columns=['Cmix'])
print(X_train.columns)


# Define the objective function 
def objFun(params):
        model = XGBRegressor(**params, random_state=np.random.RandomState(7), n_jobs=-1)
        model.fit(X_train_wo_valid, y_train_wo_valid)    
        predict_val = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, predict_val)
        return rmse
# Define the search space for model hyperparameters
space={'n_estimators': scope.int(hp.quniform('n_estimators', 50, 10000,5)),
       'min_split_loss': scope.float(hp.uniform('min_split_loss', 0.00001, 1)),
        'max_depth': scope.int(hp.quniform('max_depth', 2, 5,1)),
        'subsample': scope.float(hp.uniform('subsample', 0.9, 1)),
        'reg_alpha': scope.float(hp.uniform('reg_alpha', 0.01, 1.0)),
        'grow_policy': hp.choice('grow_policy', ['depthwise', 'lossguide']),
        'learning_rate': scope.float(hp.uniform('learning_rate', 0.01, .2)),
        'colsample_bylevel': scope.float(hp.uniform('colsample_bylevel', 0.9, 1)),
        'colsample_bytree': scope.float(hp.uniform('colsample_bytree', 0.9, 1)),
        }

# Initialize an empty trials database
trials = Trials()
# Defining random state for reproducible results
rng = np.random.default_rng(7)
# Perform 200 evaluations on the search space
best = fmin(fn=objFun,
            space=space,
            algo=tpe.suggest,
            max_evals=500,
            trials=trials,
            rstate=rng)


# # Save the trained model
df_type='df2'
with open( f'results/{df_type}_xgb_traisl.pkl', 'wb') as file:
    pickle.dump(trials, file)

# Load the saved model
with open( f'results/{df_type}_xgb_traisl.pkl', 'rb') as file:
    trials = pickle.load(file)
    
with open(f'results/{df_type}_xgb_best.pkl', 'wb') as file:
    pickle.dump(best, file)

# Load the saved model
with open(f'results/{df_type}_xgb_best.pkl', 'rb') as file:
    xgb_best = pickle.load(file)
    
best_hyperparams =  space_eval(space, xgb_best)   
best_hyperparams.update({'RMSE': trials.best_trial['result']['loss']})

print(best_hyperparams)
pd.DataFrame.from_dict([best_hyperparams]).to_csv(f'results/{df_type}_best_hyperparams.csv', index=False)

# Remove outliers in losses
losses = [trial['result']['loss'] for trial in trials.trials]
q3 = np.percentile(losses, 75)
upper_bound = q3 + 1.5 * (q3 - np.percentile(losses, 25))
filtered_trials = [trial for trial in trials.trials if trial['result']['loss'] <= upper_bound]

df_trials = pd.DataFrame([{**{k: v[0] for k, v in trial['misc']['vals'].items()}, 'loss': trial['result']['loss']} for trial in filtered_trials])

# Define the hyperparameters for plotting
hyperparameters = ['n_estimators', 'min_split_loss', 'max_depth', 'subsample', 'reg_alpha', 'grow_policy',
                   'learning_rate', 'colsample_bylevel', 'colsample_bytree']
# Plotting nine hyperparameters against losses



fig, axes = plt.subplots(3, 3, figsize=(9, 9))
axes = axes.flatten()
min_loss_index = df_trials['loss'].idxmin()
for i, hyperparam in enumerate(hyperparameters):
    ax = axes[i]
    sns.scatterplot(x=hyperparam, y='loss', data=df_trials, ax=ax)
    sns.kdeplot(y='loss',data=df_trials, x=hyperparam, ax=ax, fill=True, alpha=.7)
    ax.set_xlabel(hyperparam)
    ax.set_ylabel('RMSE')
    best_hyperparam_value = df_trials.loc[min_loss_index, hyperparam]
    best_loss = df_trials.loc[min_loss_index, 'loss']

    # ax.annotate(f"Best RMSE: {best_loss:.2f}", xy=(best_hyperparam_value, best_loss),
    # fontsize=8,ha='center', va='top', xycoords='data')
    # ax.scatter(best_hyperparam_value, best_loss, color='red', marker='x', s=40)

plt.tight_layout()
plt.savefig(f'results/{df_type}_trailsvsloss.svg', format='svg')
plt.savefig(f'results/{df_type}_trailsvsloss.png', format='png')
plt.show()
## Getting the results on training, validation and testing data sets
df_best_hyperparams=pd.read_csv(f'results/{df_type}_best_hyperparams.csv')
XGBmodel_opt = XGBRegressor(n_estimators=df_best_hyperparams['n_estimators'].values[0],
                            max_depth=df_best_hyperparams['max_depth'].values[0],
                            min_split_loss=df_best_hyperparams['min_split_loss'].values[0],
                            reg_alpha=df_best_hyperparams['reg_alpha'].values[0],
                            grow_policy=df_best_hyperparams['grow_policy'].values[0],

                            
                            colsample_bylevel=df_best_hyperparams['colsample_bylevel'].values[0],
                            colsample_bytree=df_best_hyperparams['colsample_bytree'].values[0],

                            learning_rate=df_best_hyperparams['learning_rate'].values[0],
                            subsample=df_best_hyperparams['subsample'].values[0],
                            random_state=26,
                            n_jobs=-1)

XGBmodel_opt.fit(X_train, y_train)
# Save the trained model
df_type='df2'
xgb_opt_path = f'results/{df_type}_xgb_opt.pkl'
with open(xgb_opt_path, 'wb') as file:
    pickle.dump(XGBmodel_opt, file)

# Load the saved model
with open(xgb_opt_path, 'rb') as file:
    model = pickle.load(file)
    
 
predictions_train = model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, predictions_train))
r2_train = r2_score(y_train, predictions_train)
mae_train = mean_absolute_error(y_train, predictions_train)

predictions_val = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, predictions_val))
r2_val = r2_score(y_val, predictions_val)
mae_val = mean_absolute_error(y_val, predictions_val)

predictions_test = model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, predictions_test))
r2_test = r2_score(y_test, predictions_test)
mae_test = mean_absolute_error(y_test, predictions_test)

results_df = pd.DataFrame({
    'RMSE_train': [rmse_train], 'R2_train': [r2_train], 'MAE_train': [mae_train],
    'RMSE_val': [rmse_val], 'R2_val': [r2_val], 'MAE_val': [mae_val],
    'RMSE_test': [rmse_test], 'R2_test': [r2_test], 'MAE_test': [mae_test]})
results_df.to_csv(f'results/{df_type}results.csv')
results_df


plt.figure(figsize=(4.2, 4.2))
plt.scatter(y_test, predictions_test)
plt.xlabel('Actual values (MPa)', fontsize=12)
plt.ylabel('Predicted values (MPa)', fontsize=12)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.annotate(f"R_squared = {r2_test:.3f}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12)
plt.title('dataset2', fontsize=14)
plt.ylim([0, 80])  # Set the y-axis limits to -10 and 10
plt.xlim([0, 80]) 
plt.tight_layout()

plt.savefig(f'results/{df_type}_predvsact.svg', format='svg',dpi=600)
plt.savefig(f'results/{df_type}_predvsact.png', format='png', dpi=600)
plt.show()


plt.figure(figsize=(4.2, 4.2))
residuals = y_test - predictions_test
plt.scatter(predictions_test, residuals)
plt.xlabel('Predicted values (MPa)', fontsize=12)
plt.ylabel('Residuals (Actual - Predicted) (MPa)', fontsize=12)
plt.axhline(y=0, color='red', linestyle='--')
plt.annotate(f"RMSE = {rmse_test:.3f}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12)
plt.title('dataset2', fontsize=14)
plt.ylim([-10, 10])  # Set the y-axis limits to -10 and 10
plt.xlim([0, 80]) 
plt.tight_layout()
plt.savefig(f'results/{df_type}_residualvspred.svg', format='svg', dpi=600)
plt.savefig(f'results/{df_type}_residualvspred.png', format='png', dpi=600)
plt.show()

df_exp = X_test[X_test.columns]
explainer = shap.TreeExplainer(model)
shap_values = explainer(df_exp)

# Show a summary of feature importance

plt.figure()
shap.summary_plot(shap_values, df_exp, feature_names=df_exp.columns,max_display=10)
plt.savefig(f'results/{df_type}_shapfeatimp.svg',format='svg',dpi=600, bbox_inches='tight')
plt.savefig(f'results/{df_type}_shapfeatimp.png',format='png',dpi=600, bbox_inches='tight')
plt.show()

# dependencies plots
shap_values = explainer.shap_values(X_test[X_test.columns])
top_3_features= ["Fmax","Xog_u","h"]
fig, axes = plt.subplots(len(top_3_features), 2, figsize=(9,9))

for feat_level, feat in enumerate(top_3_features):
    inds = shap.approximate_interactions(feat, shap_values, df_exp)
    for i in range(2):
        ax = axes[feat_level, i]
        shap.dependence_plot(feat, shap_values, X_test, interaction_index=inds[i], ax=ax)

plt.tight_layout()
plt.savefig(f'results/shap/{df_type}_all_interactions.svg', format='svg', dpi=600, bbox_inches='tight')
plt.close()








