# Get Started

In this tutorial we'll see how to monitor the status of the AWS API.
There are two main "services" that interest us:

* [Lambda](https://console.aws.amazon.com/lambda/home?region=us-east-1#/functions)
    * This contains one lambda "getTranslation" that is the endpoint called by "nllb200"
    * The lambda takes care of converting the language codes and call the model
    endpoint.
    * You can also find here monitoring of the API in the "monitoring" tab.
    This will give you latency numbers and number of errors
    * Use the testing tab to test the model
* [Sagemaker](https://console.aws.amazon.com/sagemaker/home?region=us-east-1#)
    * The "Models" submenu will list all models we have uploaded at some point:
        https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/models
    * The "Endpoints" submenu will show our "wikipedia" [endpoint](https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/endpoints/wikipedia)
        * Here the monitoring is hardware based (CPU utilization, ...)
        * The endpoint configuration shows which model the endpoint is running on  which hardware
        * The monitoring section contains a link to the **endpoint logs**
            This will contain several log files, indeed deploying a new model to the same
            endpoint will create a new log file. Look at the most recent one.
            The logs are very verbose filter by date and search for specific messages eg: "Traceback"


# How to ?

## How to deploy a model from the FAIR cluster

### Pre-requirements

You'll need a conda environment with pytorch installed.

conda activate ???
cd misc/aws_publish
pip install -r requirements.txt

You also need to be able to create docker images:

`sudo usermod -aG docker $USER`

You'll need to exit and relogin for this change to take effect.

### Run local tests on the model

cd wikipedia-en-xx-distill-aan
python handler.py

Note that the tests and the handler may need some adjustment 
if you change how the model is trained.

### Upload the model through command line

`python deploy.py -f $FOLDER`

This will:
* build a docker image
* push the docker image to AWS
* create a model pointing to the docker image
* create an endpoint configuration using this model and hardware specified in the code
* deploy the model to `nllb200-staging` endpoint

**Note**: you can pass the name of a docker image to reuse, 
skipping the long docker build with `--docker`. 

## How to test a staging model

Go to the [lambda testing tab](https://console.aws.amazon.com/lambda/home?region=us-east-1#/functions/getTranslation?tab=testing)
In the "test event" dropdown, chose the sample "with specific endpoint".
This will show how to send a request to the staging environment.
You can modify the JSON and send other queries. 
Just make sure to not save your modifications.


## How to pass a model from staging to prod

Go to the "nllb200" (prod) endpoint.
Scroll down to the "Endpoint configuration settings" section,
and click on [Change](https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/endpoints/nllb200/edit).
Scroll down to "Change the Endpoint configuration", and chose "Use an existing endpoint configuration".
Find the configuration with the name of the staging model.
Note that configuration for staging ends with "-staging", 
chose the one without the suffix.


## How to increase the number of GPUs

From the [endpoint page](https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/endpoints/nllb200)
scroll down to the "Endpoint runtime settings".
Normally there should be only one model in the list.
Select it and then click on "Update instance count".
Once validated it takes some time to actually allocate the extra GPUs.

## How to change a model configuration through the UI

To change the model used by an endpoint:
* [create a new endpoint configuration](https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/endpointConfig/create)

* Scroll to "Production Variants", click "Add model"
* Chose a model from the list
* Change the default instance type to something with gpu. 
Typically ml.g4dn.4xlarge is big enough for our models.
ml.p3.2xlarge could also be an option.
* Chose how many instances to use
* Name the endpoint configuration with the model name and instance name
* Create the endpoint
* From the endpoint configuration view, select your new config
* "Apply to endpoint" > "nllb200" > "Chose endpoint"
* Wait for the changes to propagate
* Go to the Lambda testing tab to test the changes
