import boto3
import json
import os

runtime = boto3.client("sagemaker-runtime")
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
SECRET = os.environ["SECRET"]
STAGING_SECRET = os.environ["STAGING_SECRET"]

assert SECRET
assert STAGING_SECRET
assert ENDPOINT_NAME

WIKI2ISO = {
    "as": "asm", # Assamese
    "ast": "ast", # Asturian
    "ay": "ayr", # Central Aymara
    "ba": "bak", # Bashkir
    "bem": "bem", # Bemba
    "ca": "cat", # Catalan
    "ckb": "ckb", # Central Kurdish
    "en": "eng", # English
    "fr": "fra", # French
    "ha": "hau", # Hausa
    "ig": "ibo", # Igbo
    "ilo": "ilo", # Iloko
    "is": "isl", # Icelandic
    "kg": "kon", # Kongo
    "ln": "lin", # Lingala
    "lg": "lug", # Ganda
    "nso": "nso", # Norther Sotho
    "oc": "oci", # Occitan
    "om": "orm", # Oromo
    "pt": "por", # Portuguese
    "ss": "ssw", # Swati
    "qu": "que", # Quechua
    "ru": "rus", # Russian
    "es": "spa", # Spanish
    "ss": "ssw", # Swati
    "ti": "tir", # Tigrinya
    "tn": "tsn", # Tswana
    "ts": "tso", # Tswana
    "wo": "wol", # Wolof
    "zh-yue": "yue", # Yue Chinese
    "yue": "yue",
    "zh": "zho_simp", # Chinese
    "zu": "zul", # Zulu
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
      "model_name": "endpoint-name",
      "secret": "some_top_secret"
    }

            returns - a json object with probability and signed. Passthrough from model
    """
    if event.get("secret") not in (SECRET, STAGING_SECRET):
        raise Exception("Invalid authentication")
    endpoint = event.get("model_name", ENDPOINT_NAME)
    if event.get("secret") == STAGING_SECRET:
        endpoint = "wikipedia-staging"

    for sample in event["samples"]:
        src = sample["sourceLanguage"]
        tgt = sample["targetLanguage"]
        try:
            sample["sourceLanguage"] = WIKI2ISO[src]
            sample["targetLanguage"] = WIKI2ISO[tgt]
        except KeyError as e:
            error_message = f"CLIENTERROR Unknown language in {src}->{tgt}. Chose from: {', '.join(WIKI2ISO.keys())}"

            return error_message


    payload = "\n".join(json.dumps(s) for s in event["samples"])

    # Invoke sagemaker endpoint to get model result
    assert endpoint
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
