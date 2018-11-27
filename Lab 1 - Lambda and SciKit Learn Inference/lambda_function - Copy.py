from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import boto3
import random
import json

#Warning is the base class so it should catch all types of warnings
warnings.filterwarnings(action='ignore', category=Warning) 
s3 = boto3.resource('s3')

# make Prediction on the trained model
def lambda_handler(event, context):

	# Bucket_name is passed on from event
	data = json.loads(json.dumps(event))
	BUCKET_NAME = data['bucket_name']
	#TEST_TWEET_NEG = data['test_tweet_neg']
	#TEST_TWEET_POS = data['test_tweet_pos']
	
	# Initialize onstants variables
	PICKLE_FILE_NAME = '/tmp/sentiment_analysis_tweet.pkl'
	PICKLE_FILE_KEY = 'ServerlessAIWorkshop/SentimentAnalysis/sentiment_analysis_tweet.pkl'
	X_TEST = '/tmp/X_test.data'
	y_TEST = '/tmp/y_test.data'
	X_TEST_FILE_KEY = 'ServerlessAIWorkshop/SentimentAnalysis/X_test.data'
	y_TEST_FILE_KEY = 'ServerlessAIWorkshop/SentimentAnalysis/y_test.data'

	
	# get trained model/pkl file from S3 bucket
	s3.Object(BUCKET_NAME, PICKLE_FILE_KEY).download_file(PICKLE_FILE_NAME)
	# get dataset files from s3 bucket
	s3.Object(BUCKET_NAME, X_TEST_FILE_KEY).download_file(X_TEST)
	s3.Object(BUCKET_NAME, y_TEST_FILE_KEY).download_file(y_TEST)
	
	# loading the trained model as well as test data and label
	model = joblib.load(PICKLE_FILE_NAME) 
	X_test = joblib.load(X_TEST)
	y_test = joblib.load(y_TEST)
	
	# predict using the test dataset
	y_pred = model.predict(X_test)
	
	# evaluate predictions
	accuracy = accuracy_score(y_test, y_pred)
	
	print("Prediction for the first test tweet is " + model.predict([X_test[0]])[0])
	
	return {
        "statusCode": 200,
        "body": json.dumps("Accuracy from the validation dataset: %.2f%%" % (accuracy * 100.0))
    }
