// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

const BACKEND_PORT = process.env.NODE_ENV === "development" ? "8000" : "8800";

export const config = {
  host: "http://localhost",
  port: BACKEND_PORT,
  annotations_route: "/annotations/",
  general_route: "/general/",
  audio_route: "/servefile/",
  upload_route: "/upload_microphone/",
  embed_route: "/embed_audio/",
  faiss_route: "/faiss/",
  postprocess_route: "/postprocess/",
  default_color: "#7a26c1",
  secondary_color: "#2BAD60",
  main_area: "main",
  default_path:
    "/default/path/",
};
