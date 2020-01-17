# LambdaMart
Python implementation of LambdaMart

LambdaMART(**kwargs)

Parameters:

	kwargs: LGBMRegressor parameters
    default :
    (boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000,
    objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0,
    subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True,
    importance_type='split', **kwargs)


Methods:

	fit: Fits the model on the training data.
		Parameters: None
		Returns: None
	predict: Predicts the scores for the test dataset.
		Parameters: Numpy array of documents with each document’s format is [query index, feature vector] 
		Returns: Numpy array of scores
	save: Saves the model into file with the name given as a parameter
		Parameters: Filename
		Returns: None
	load: Loads the model from the file given as a parameter
		Parameters: Filename
		Returns: None
