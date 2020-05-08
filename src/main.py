from __future__ import division

import utils

from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
import scipy.ndimage

import numpy as np
import pandas as pd
import sys
import copy

import os
import shutil
import warnings
import multiprocessing
from multiprocessing import Process, Manager


def main(base_dir, input_path, out_path):

	def run_preprocessing_pipeline(image_id):
	    image_path = os.path.join(base_dir, image_id)
	    image = skimage.io.imread(image_path)
	    image = utils.segment_lungs(color.rgb2gray(image), display=True)
	    image = utils.clean_noise(image)
	    feature_dictionary = utils.extract_features(image)
	    feature_dictionary['image_id'] = image_id
	    return feature_dictionary


	input_df = feather.read_dataframe(input_path)
	image_ids = list(input_df['image_id'])

	with multiprocessing.Pool() as pool:
	    features = pool.map(run_preprocessing_pipeline, image_ids)

	for feature_dict in features:
	    for key, value in feature_dict.items():
	        if (isinstance(value, str)):
	            pass
	        elif (isinstance(value, float)):
	            feature_dict[key] = [value]
	        elif (isinstance(value, int)):
	            feature_dict[key] = [value]
	        elif (isinstance(value[0], np.ndarray)):
	            feature_dict[key] = value[0]
	        elif (isinstance(value[0], float)):
	            feature_dict[key] = value

	df = pd.DataFrame()

	for feature_dict in features:
	    feature_df = pd.DataFrame.from_dict(feature_dict)
	    df = pd.concat([df, feature_df], 
	    				ignore_index=False, 
	    				sort=False)
	    
	df = df.reset_index(drop=True)
	df = df.merge(input_df, 
				  on='image_id', 
				  how='inner')

	image_ids = df.pop('image_id').to_frame()
	df.insert(0, 'image_id', image_ids)

	df = shuffle(df)
	df.dropna(inplace=True)

	# Train and Evaluate Random Forest Classifier

	features = copy.deepcopy(df.columns.tolist())
	features.remove('image_id')
	features.remove('is_covid')

	kfold = KFold(n_splits=10, shuffle=False, random_state=42)

	X = df[features]
	y = df['is_covid']

	accuracies = []

	random_forest_classifier = RandomForestClassifier(n_estimators=200, 
													  max_depth=60, 
													  max_features=15, 
													  random_state=42)

	for train_index, test_index in kfold.split(X):

	    X_train, X_test = X.iloc[train_index[0]: train_index[-1]], X.iloc[test_index[0]: test_index[-1]]
	    y_train, y_test = y.iloc[train_index[0]: train_index[-1]], y.iloc[test_index[0]: test_index[-1]]
	    
	    random_forest_classifier.fit(X_train, y_train)
	    
	    acc = random_forest_classifier.score(X_test, y_test)
	    accuracies.append(acc)

	print("K-Fold Cross Validation Accuracy is {}".format(sum(accuracies)/len(accuracies)))


if __name__ == "__main__":
	base_dir = '...'
	input_path = '...'
	out_path = '...'
	main(base_dir, input_path, out_path)