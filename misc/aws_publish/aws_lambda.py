# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import boto3

runtime = boto3.client("sagemaker-runtime")
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
SECRET = os.environ["SECRET"]
STAGING_SECRET = os.environ["STAGING_SECRET"]

assert SECRET
assert STAGING_SECRET

WIKI2ISO = {
    "en": "eng",  # English
    "oc": "oci",  # Occitan
    "lg": "lug",  # Luganda
    "zu": "zul",  # Zulu
    "ig": "ibo",  # Igbo
    "zh": "zho_simp",  # Chinese simplified
    "is": "isl",  # Icelandic
    "ha": "hau",  # Hausa
}


def lambda_handler(event, context):
    """
    {
      "samples": [
        {
          "uid": "sample0",
          "sourceText": "It is a good day !",
          "sourceLanguage": "en",
          "targetLanguage": "zh",
        }
      ],
      "model": "endpoint-name",
      "secret": "some_top_secret"
    }

            returns - a json object with probability and signed. Passthrough from model
    """
    if event.get("secret") not in (SECRET, STAGING_SECRET):
        raise Exception("Invalid authentication")
    endpoint = event.get("model") or ENDPOINT_NAME
    for sample in event["samples"]:
        src = sample["sourceLanguage"]
        tgt = sample["targetLanguage"]
        try:
            sample["sourceLanguage"] = WIKI2ISO.get(src, src)
            sample["targetLanguage"] = WIKI2ISO.get(tgt, tgt)
        except KeyError as e:
            raise Exception(
                f"Unknown language in {src}->{tgt}. Chose from: {', '.join(WIKI2ISO.keys())}"
            )

    payload = "\n".join(json.dumps(s) for s in event["samples"])

    # Invoke sagemaker endpoint to get model result
    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Body=payload,
        )
        result = response["Body"].read()
    except Exception as ex:
        print("Exception = ", ex)
        raise

    return result
