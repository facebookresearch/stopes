# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_SESSION_TOKEN = os.environ["AWS_SESSION_TOKEN"]
AWS_ACCOUNT = os.environ["AWS_ACCOUNT"]

config = {
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_session_token": AWS_SESSION_TOKEN,
    "aws_region": "us-east-1",
    "account": AWS_ACCOUNT,
    "arn_role": f"arn:aws:iam::{AWS_ACCOUNT}:role/AmazonSageMaker-ExecutionRole",
    "sagemaker_role": "AmazonSageMaker-ExecutionRole",
}
