#!/usr/bin/env python
# coding: utf-8

# # Initialize

# ## ASW session

# In[81]:


import boto3

session = boto3.Session()
s3 = session.resource('s3')


# ## SageMaker Session

# In[82]:


import sagemaker

sage_session = sagemaker.Session()


# # Define Globals

# In[83]:


# bucket name
bucket = 'aida-project'

# path for s3_input
s3_path = f"s3://{bucket}/"

## define source files
# TRAIN source file
train_source_path = 'team_remote'
train_source_name = 'title-train.csv'

# VALID source file
valid_source_path = 'team_remote'
valid_source_name = 'title-valid.csv'

# TEST source file
test_source_path = 'team_remote'
test_source_name = 'title-test.csv'

# output for result
output_path =''
output_location = f'{s3_path}{output_path}output'


# In[84]:


# Create pointers to the S3 train and test datasets

from sagemaker import s3_input

# print(f"{s3_path}{train_source_path}/{train_source_name}")
# print(f"{s3_path}{valid_source_path}/{valid_source_name}")
# print(f"{s3_path}{test_source_path}/{test_source_name}")
s3_input_train = sagemaker.session.s3_input(s3_data=f"{s3_path}{train_source_path}/{train_source_name}", content_type="text/csv")
s3_input_valid = sagemaker.session.s3_input(s3_data=f"{s3_path}{valid_source_path}/{valid_source_name}", content_type="text/csv")
s3_input_test = sagemaker.session.s3_input(s3_data=f"{s3_path}{test_source_path}/{test_source_name}", content_type="text/csv")


# In[85]:


role = sagemaker.get_execution_role()


# # Define Model

# ### Factorization Machine

# In[86]:


# Factorization Machine

# estimator call
# <ecr_path>/factorization-machines:<tag>

ecr_path = '664544806723.dkr.ecr.eu-central-1.amazonaws.com'
tag = 'latest'
estimator = f'{ecr_path}/factorization-machines:{tag}'


# In[87]:


JPC_factor = sagemaker.estimator.Estimator(estimator,
                              role,
                              train_instance_count=1,
                              train_instance_type="ml.m5.4xlarge",
                              output_path=output_location,
                              sagemaker_session=sage_session,
                              base_job_name="JPC-factor")


# In[88]:


JPC_factor.set_hyperparameters(
    feature_dim = 44,
    num_factors = 64,
    predictor_type = 'regressor'
    )


# In[89]:


JPC_factor.fit({"train": s3_input_train, "validation": s3_input_valid}, wait=True)


# In[ ]:


JPC_factor.deploy(
#     initial_instance_count = 1, 
#     instance_type = "ml.m5.4xlarge")


# ### neu Matrix Factorization

# In[ ]:


get_ipython().system(' python')


# In[ ]:


get_ipython().system(' pip install pymf3')


# In[ ]:


import cvxopt, numpy, and scipy


# In[ ]:





# In[ ]:





# In[ ]:





# ### ALS Model: 
# https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1

# In[52]:


def _jvm():
    """
    Returns the JVM view associated with SparkContext. Must be called
    after SparkContext is initialized.
    """
    jvm = SparkContext._jvm
    if jvm:
        return jvm
    else:
        raise AttributeError("Cannot load _jvm from SparkContext. Is SparkContext initialized?")


# In[50]:


from pyspark.ml.recommendation import ALS
from pyspark import SparkContext


def tune_ALS(train_data, validation_data, maxIter, regParams, ranks):
    """
    grid search function to select the best model based on RMSE of
    validation data
    Parameters
    ----------
    train_data: spark DF with columns ['userId', 'movieId', 'rating']
    
    validation_data: spark DF with columns ['userId', 'movieId', 'rating']
    
    maxIter: int, max number of learning iterations
    
    regParams: list of float, one dimension of hyper-param tuning grid
    
    ranks: list of float, one dimension of hyper-param tuning grid
    
    Return
    ------
    The best fitted ALS model with lowest RMSE score on validation data
    """
    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in regParams:
            # get ALS model
            als = ALS().setMaxIter(maxIter).setRank(rank).setRegParam(reg)
            # train ALS model
            model = als.fit(train_data)
            # evaluate the model by computing the RMSE on the validation data
            predictions = model.transform(validation_data)
            evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print('{} latent factors and regularization = {}: '
                  'validation RMSE is {}'.format(rank, reg, rmse))
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_regularization = reg
                best_model = model
    print('\nThe best model has {} latent factors and '
          'regularization = {}'.format(best_rank, best_regularization))
    return best_model


# In[54]:


tune_ALS(train_source_name, valid_source_name, 2, [0.1,0.2], [10.0,20.0])


# ### XGBoost

# In[53]:


# find pre-defined models here:
# https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html

# Create an XGBoost Estimator

# estimator call
# <ecr_path>/sagemaker-xgboost:1.2-1
    
ecr_path = "492215442770.dkr.ecr.eu-central-1.amazonaws.com"
estimator = f"{ecr_path}/sagemaker-xgboost:1.0-1-cpu-py3"


# In[15]:


xgboost = sagemaker.estimator.Estimator(estimator,
                              role,
                              train_instance_count=1,
                              train_instance_type="ml.m5.4xlarge",
                              output_path=output_location,
                              sagemaker_session=sage_session,
                              base_job_name="JPC-xgboost")


# In[16]:


# Select the your specific hyperparameters (Optional)
# https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html
# https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst#learning-task-parameters 

xgboost.set_hyperparameters(
    eta=.35,
    num_round=30,                     # required parameter!
    objective = 'reg:logistic',
    )


# ## Fit Model

# In[ ]:


#xgboost.fit({"train": s3_input_train, "validation": s3_input_valid}, wait=True)


# # Deploy Model

# ## Deploy

# In[ ]:


# Deploy  model to an endpoint

# xgb_predictor = xgboost.deploy(
#     initial_instance_count = 1, 
#     instance_type = 'ml.t2.medium')


# ## Configure Predictor

# In[ ]:


# Configure the predictor's serializer and deserializer

# INSERT CODE HERE

from sagemaker.predictor import csv_serializer, json_deserializer

# xgb_predictor.content_type = "text/csv"
# xgb_predictor.serializer = csv_serializer
# xgb_predictor.deserializer = json_deserializer


# # Display Endpoint

# In[18]:


# xgb_predictor.endpoint


# # Prediction

# In[20]:


# carefully just one prediction
# prediction = xgb_predictor.predict(test.iloc[0, 1:])
# prediction

