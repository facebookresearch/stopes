// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { config } from "../constants/config";
import { AnnotationQuery, LineResult } from "../types/api";

const processFilesUrl = config.host + ":" + config.port + config.annotations_route;
const processFoldersUrl = config.host + ":" + config.port + config.general_route;

async function fetchFiles(data: AnnotationQuery): Promise<LineResult[]> {
  let response = await fetch(processFilesUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    throw response;
  }
  return response.json();
}

async function processFolder(folderPath: string): Promise<LineResult[]> {
  const folderQuery = {
    gz_path: folderPath,
    start_idx: 0, // Provide appropriate values
    end_idx: 0,   // Provide appropriate values
  };

  const response = await fetch(processFoldersUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(folderQuery),
  });

  if (!response.ok) {
    throw response;
  }

  return response.json();
}

export { fetchFiles, processFolder };

