// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { config } from "../constants/config";
import { AnnotationQuery, LineResult } from "../types/api";

const url = config.host + ":" + config.port + config.annotations_route;

async function fetchFiles(data: AnnotationQuery): Promise<LineResult[]> {
  let response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    throw response;
  }
  return response.json();
}

export default fetchFiles;
