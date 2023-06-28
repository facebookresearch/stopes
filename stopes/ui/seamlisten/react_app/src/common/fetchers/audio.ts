// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { config } from "../constants/config";
import { AudioQuery } from "../types/api";

const url: string = config.host + ":" + config.port + config.audio_route;

async function fetchAudio(data: AudioQuery): Promise<Blob> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    throw response;
  }
  return await response.blob();
}
export default fetchAudio;
