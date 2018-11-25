# SageMaker Batch Transform Hands-on

In this workshop you will use the MNIST dataset to create a predictive model using SageMaker's K-Means built-in algorithm. After the model is trained, Sagemaker's batch transform job will be created and called from a Lambda function for inference.

## Prerequisites

### AWS Account

To complete this workshop you'll need an AWS Account with access to create AWS IAM, S3, Lambda, and Sagemaker. The code and instructions in this workshop assume only one student is using a given AWS account at a time. If you try sharing an account with another student, you'll run into naming conflicts for certain resources. You can work around these by appending a unique suffix to the resources that fail to create due to conflicts, but the instructions do not provide details on the changes required to make this work.

All of the resources you will launch as part of this workshop are eligible for the AWS free tier if your account is less than 12 months old. See the [AWS Free Tier page](https://aws.amazon.com/free/) for more details.

### Browser

We recommend you use the latest version of Firefox or Chrome to complete this workshop.

## Steps

1. If you have already completed Steps 1 through 5 from the "Lab 1 - Lambda and SciKit Learn Inference" section of this workshop, please proceed to step 2. Otherwise, go to [Lab 1 - ServerlessAI-Workshop/Lambda and SciKit Learn Inference](https://github.com/aws-samples/serverless-ai-workshop/tree/master/Lab%201%20-%20Lambda%20and%20SciKit%20Learn%20Inference) and complete the 5 steps required to setup SageMaker and add the required IAM roles and policies. Also, this workshop assumes all SageMaker instances and S3 buckets are in the Oregon region: us-west-2.

1. Open the notebook, **MNIST Clusters.ipynb** in this directory. Examine, understand, and run each cell and return to these instructions. If any cell is unclear please ask for help either in the workshop or at http://bit.ly/serverlessAI

	Don't be alarmed by red log messages while training your model. SageMaker prints ordinary log messages in red. Wall time for training takes about 4 minutes. **Do not proceed to the nest step until the notebook has fully completed and each cell has run.**

	When training is complete you can get the full path to your newly trained model from the notebook in the next cell. The model name is embedded in the s3 URL. You'll need the name of your model for subsequent steps. Make a copy now. Your model name looks like ```kmeans-2018-10-10-22-45-30-733```.

1. Return to the Jupyter file browser tab. Launch the SageMaker terminal. 

	Create Lambda deployment package on the notebook instance by running the following commands in SageMaker's terminal.

	```
	cd ~/SageMaker/ServerlessAI-Workshop/Lab\ 2\ -\ SageMaker\ Batch\ Transform/LambdaPackage/
	pip install --ignore-installed --target=. sagemaker
	find . -name "*.so" | xargs strip
	7z a -mm=Deflate -mfb=258 -mpass=15 -r ../ServerlessAI_BatchTransform.zip *	
	```

1. Upload the deployment package to your S3 bucket by running the following command:

	```
	aws s3api put-object --bucket <your-bucket-name> --key ServerlessAIWorkshop/BatchTransform/ServerlessAI_BatchTransform.zip --body ../ServerlessAI_BatchTransform.zip --region us-west-2
	```
	*Tip: Use control-a to quickly move your cursor to the beginning of the line.*

1. Create a Lambda function using the deployment package stored in your S3 bucket. You can reuse the role you created in Lab 1: Lambda_ServerlessWorkshop_Role.
 
	```
	aws lambda create-function --role <arn from IAM Console> --code S3Bucket=<your-bucket-name>,S3Key="ServerlessAIWorkshop/BatchTransform/ServerlessAI_BatchTransform.zip"  --function-name ServerlessAI_BatchTransform --runtime python3.6 --handler lambda_function.lambda_handler --memory-size 128 --timeout 900 
	```

	If you need to update the code, use the following command:
	
	```
	aws lambda update-function-code --role <arn from IAM Console> --code S3Bucket=<your-bucket-name>,S3Key="ServerlessAIWorkshop/BatchTransform/ServerlessAI_BatchTransform.zip" --function-name ServerlessAI_BatchTransform --runtime python3.6 --handler lambda_function.lambda_handler --memory-size 128 --timeout 900
	```

	Close the terminal window.


1. Naviage to the Lambda console. Click on your new batch transform lambda function:  ServerlessAI_BatchTransform. Click on Test. 

	You'll need the model name you saved in an earlier step. Your model name looks like ```kmeans-2018-10-10-22-45-30-733```. If you didn't save your model name you can obtain it from the SageMaker console. Click on Models. It's the top, the most recent model. Create a test event with the following JSON. 

	```
	{
	  "bucket": "<your-bucket-name>",
	  "model": "<your-model-name>" 
	}
	```

	Click on Test and view the results. It takes 3-4 minutes to run. The tradeoff for the flexibility of batch transform is latency. Whereas our lambda functions calling Scikit-learn run in milliseconds batch transform requires provisioning a compute instance, loading the Docker container we created in the notebook, loading up the inference data, and calling the model. 

## Congratulations!

You've successfully created a batch transform job from a Lambda function. This particular job predicted the handwritten digit of 10,000 images. The scripts are generic enough for you to use as a template for your own predictions. 

## Cleanup
After you have completed the workshop you can delete all of the resources that were created in the following order.
1. Stop the Sagemaker notebook instance.
1. Delete the Sagemaker notebook instance.
1. Delete the S3 Buckets you created. 
