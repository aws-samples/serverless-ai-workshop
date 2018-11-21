# ServerlessAI

## Run machine learning inference serverless with AWS Lambda and SageMaker 
This workshop demonstrates two methods of machine learning inference for globally-scalable production using AWS Lambda and Amazon SageMaker. **Scikit-learn** is a popular machine learning library that covers most aspects of shallow learning. With these techniques the library module and your custom inference code can be combined into a flexible package for instantaneous execution in the cloud. 

Though scikit-learn is vast and deep, it can't do everything. Importantly, it can't call inference on deep learning models, importantly those created with MXNet, TensorFlow, or pytorch. **Amazon SageMaker** is ideally suited to build and deploy models with those frameworks. The SageMaker feature **batch-transform** is demonstrated calling inference in the cloud on-demand. As such it can handle nearly every inference workload without the need for a costly persistent endpoint. 

### What is AI? 
AI enables machine functionality that can be interpreted as human intelligence. In practice this is the ability to infer predictions based on data collected from previous experiences. Machine learning, a subset of AI, is currently enjoying a renascence from its dormant state since the mid-90s thanks to new GPU-driven software techniques. 

### What is Serverless
Serverless computing enables you to build & run applications without thinking about servers and other network infrastructure. Relieving a developer from the burden of bootstraping, configuring, networking, clustering, updating, and all aspects of physical server management enables focus on the tasks at hand without sacrificing global access and scalability. In most cases serverless computing dramatically reduces the cost of application workloads. Benefits include:

### A Fully Managed Service
- Provides fully managed services that do not require provisioning, 
- system administration, 
- updates including security patches,
- or any developer responsibility to achieve fault tolerance.
- AWS takes care of the underlying compute infrastructure. 

### Developer Productivity
- Serverless aims to lower the cost of experimentation and help developers innovate more quickly. 
- The time you save, you can invest it into development of your application. 

### Cost-Effective Continuous Scaling
This underlying compute infrastructure scales underneath your application
It means:
- You are guaranteed a uniform experience regardless of scale.
- You never have to pay for idle capacity.

### Limitations of Batch Transformation
Though the capacity for SageMaker is limitless there is a cost when using Batch Transformation: latency. When calling inference on Batch Transformation you must endure the cost of loading a Docker image for the workload. This usually takes 2-3 minutes. There's no way to speed this up. If millisecond latency is required for your application you should not choose this option. Instead take advantage of SageMaker's ability to deploy persistent endpoints. 


## Workshop
We recommend you start with Lambda and SciKit Learn Inference and complete the workshop until you have successful results. Then, provide new test data to explore the robustness of your model. When you're satisfied with your ability to deploy scikit-learn move on to SageMaker Batch Transform. Here the possibilities are endless. 
