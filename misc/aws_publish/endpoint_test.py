# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import json
import requests

API_KEY = "<enter api key here>"
SECRET = "<enter secret here>"
ENDPOINT = "<enter endoint here>"

def mk_sample(sentence: str, uid: int, src: str, tgt: str):
    return {
        "uid": uid,
        "sourceText": sentence,
        "sourceLanguage": src,
        "targetLanguage": tgt,
    }


def api_call(data, model: str = None):
    if model:
        data["model"] = model
    data["secret"] = SECRET
    text_data = json.dumps(data, ensure_ascii=False)
    start_time = time.time()
    x = requests.post(
        ENDPOINT,
        data=text_data,
        headers={"x-api-key": API_KEY},
    )
    end_time = time.time()
    print(f"observed api latency: {end_time - start_time:.3f}")
    x.raise_for_status()
    if x.text.startswith('{"errorMessage"'):
        raise Exception(x.json()["errorMessage"])
    return x


def translate_one(sentence: str, *, tgt: str, src: str = "en", model: str = None):
    data = {"samples": [mk_sample(sentence, 0, src, tgt)]}
    x = api_call(data, model)
    answer = json.loads(x.text)
    return answer["translatedText"]


def translate(paragraph: str, *, tgt: str, src: str = "en", model: str = None):
    samples = [
        mk_sample(line, i, src, tgt)
        for i, line in enumerate(paragraph.strip().splitlines())
    ]
    data = {"samples": samples}
    x = api_call(data, model)
    return x

def parse_response(response, throw=True):
    answer = [json.loads(l) for l in response.text.split("\n")]
    if "errorMessage" in answer[0]:
        if throw:
            raise Exception(answer[0]["errorMessage"])
        else:
            return answer[0]["errorMessage"]

    answer.sort(key=lambda d: d["id"])
    return "\n".join(d["translatedText"] for d in answer)


APACHE = """
Apache Giraph is an Apache project to perform graph processing on big data.
Apache Giraph
Apache
graph processing
big data
Giraph utilizes Apache Hadoop's MapReduce implementation to process graphs.
Apache Hadoop
Facebook used Giraph with some performance improvements to analyze one trillion edges using 200 machines in 4 minutes.
Facebook
Giraph is based on a paper published by Google about its own graph processing system called Pregel.
It can be compared to other Big Graph processing libraries such as Cassovary.
Apache Software Foundation
Graph (computer science)
PC World
Apache Hadoop
Facebook
Gigaom
Big data
"""

def test_one(model=None):
    tr = translate(APACHE, tgt="zu", model=model).text
    print(tr)

def stress_testing_qps(qps, duration, paragraph=APACHE):
    freq = 1 / qps
    n = qps * duration
    stress_testing(n, freq, paragraph)

def stress_testing(n, freq, paragraph=APACHE):
    print(f"Stress testing with {n} parallel queries every {freq}s")
    src, tgt = "en", "zu"
    samples = [
        mk_sample(line, i, src, tgt)
        for i, line in enumerate(paragraph.strip().splitlines())
    ]
    data = json.dumps({"samples": samples, "secret": SECRET}, ensure_ascii=False)
    headers = {"x-api-key": API_KEY}

    from concurrent.futures import ThreadPoolExecutor

    ex = ThreadPoolExecutor(max_workers=n)
    queries = []
    for _ in range(n):
        time.sleep(freq)
        queries.append(
            ex.submit(requests.post, ENDPOINT, data=data, headers=headers)
        )

    responses = [q.result() for q in queries]
    latencies = [r.elapsed.total_seconds() for r in responses]
    latency = sum(latencies) / len(responses)
    print(f"Received {n} responses. Avg latency {latency:.3f}s: {latencies}")
    failures = [r for r in responses if not r.ok]
    if failures:
        print([r.status_code for r in failures])

    translations = [parse_response(r, throw=False) for r in responses if r.ok]
    success = sum(translations[0] == t for t in translations)
    assert success == len(responses), f"Received {success} valid translations in {len(responses)}"
    assert not failures, len(failures)

if __name__ == '__main__':
    test_one()
    # stress_testing_qps(duration=1, qps=100)
    # stress_testing_qps(duration=60, qps=10)
