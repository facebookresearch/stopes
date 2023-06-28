// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { trackPromise } from "react-promise-tracker";
import { config } from "../constants/config";
import { UploadQuery, Audio } from "../types/api";

const url_base = config.host + ":" + config.port;
const url_upload: string = url_base + config.upload_route;
const url_embed: string = url_base + config.embed_route;
const url_faiss: string = url_base + config.faiss_route;
const url_postprocess: string = url_base + config.postprocess_route;

async function uploadAudio(
  data: UploadQuery,
  area: string,
  error_setter: Function
): Promise<string> {
  const formData = new FormData();
  // formData.append("filename", data.filename)
  formData.append("file", data.blob);
  let response = await trackPromise(
    fetch(url_upload, {
      method: "POST",
      headers: { "Enc-Type": "multipart/form-data" },
      body: formData,
    }),
    area
  );
  if (!response.ok) {
    error_setter(response.statusText);
    return "";
  }
  return await response.json();
}

function step_to_url(step: string): string {
  switch (step) {
    case "embed":
      return url_embed;
    case "faiss":
      return url_faiss;
    case "postprocess":
      return url_postprocess;
    default:
      return url_embed;
  }
}

async function mining_step(
  prev_step_id: string,
  error_setter: Function,
  step: string
): Promise<string | Array<Audio> | void> {
  let url = new URL(step_to_url(step));
  url.search = new URLSearchParams({ reference: prev_step_id }).toString();

  let response = await trackPromise(
    fetch(url, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    }),
    step
  );
  if (!response.ok) {
    error_setter(response.statusText);
    return;
  }
  return response.json();
}

export { uploadAudio, mining_step };
