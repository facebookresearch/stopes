// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { AudioQuery } from "../../types/api";

function get_area_name(query: AudioQuery): string {
  return query.path + "_" + String(query.start) + "_" + String(query.end);
}

export default get_area_name;
