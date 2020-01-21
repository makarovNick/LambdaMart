# LambdaMart
Python implementation of LambdaMart

LambdaMART(**kwargs)

Parameters:

	kwargs: XGBRegressor parameters
    		default :
	    (max_depth=3, learning_rate=0.1, n_estimators=100, verbosity=1, objective='reg:squarederror', booster='gbtree',
	    tree_method='auto', n_jobs=1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, 
	    colsample_bylevel=1, colsample_bynode=1, reg_alpha=0,reg_lambda=1,scale_pos_weight=1,base_score=0.5,random_state=0, 
	    missing=None, num_parallel_tree=1, importance_type='gain', **kwargs)


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
