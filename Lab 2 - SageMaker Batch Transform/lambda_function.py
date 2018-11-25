import boto3
import sagemaker
import json
from sagemaker.session import Session

s3 = boto3.resource('s3')

# make Prediction on the trained model
def lambda_handler(event, context):

	# Bucket_name and model name are passed on from event
	data = json.loads(json.dumps(event))
	BUCKET = data['bucket']
	MODEL = data['model']
	
	# Initialize onstants variables
	sagemaker_session = sagemaker.Session()
	region = sagemaker_session.boto_session.region_name
	INPUT_BUCKET = 'sagemaker-sample-data-{}'.format(region)
	DATADIR = 'batch-transform/mnist-1000-samples'
	OUTPUT = 'ServerlessAIWorkshop/BatchTransform/output'

	input_location = 's3://{}/{}'.format(INPUT_BUCKET, DATADIR)
	output_location = 's3://{}/{}'.format(BUCKET, OUTPUT)

	# Initialize the transformer object
	transformer =sagemaker.transformer.Transformer(
		base_transform_job_name='Serverless-Workshop',
		model_name=MODEL,
		instance_count=4,
		instance_type='ml.c5.4xlarge',
		output_path=output_location
		)
	
	# To start a transform job:
	transformer.transform(input_location, content_type='text/csv', split_type='Line')

	# Then wait until transform job is completed
	transformer.wait()

	s3 = boto3.resource('s3')
	predictions = []
	results = []
	for i in range(10):
		file_key = '{}/data-{}.csv.out'.format(OUTPUT, i)

		output_obj = s3.Object(BUCKET, file_key)
		output = output_obj.get()["Body"].read().decode('utf-8')

		predictions.extend(json.loads(output)['outputs']['classes']['int64Val'])

	originals='7, 2, 1, 0, 4, 1, 4, 9, 5, 9'
	predictions=', '.join(predictions)

	return {
        "statusCode": 200,
        "body": json.dumps({'originals': originals, 'predictions': predictions})
    }