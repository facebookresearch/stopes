// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { usePromiseTracker } from "react-promise-tracker";
import { ScaleLoader } from "react-spinners";
import { config } from "../../constants/config";
import "./spinner.css";

export function PromiseSpinner({ area }: { area: string }): JSX.Element {
  const { promiseInProgress } = usePromiseTracker({ area: area });
  if (promiseInProgress) {
    return <InnerScale loading={true} />;
  }
  return <></>;
}

function InnerScale({ loading }: { loading: boolean }): JSX.Element {
  if (loading) {
    return <ScaleLoader color={config.secondary_color} />;
  } else {
    return <div />;
  }
}

export default InnerScale;
