// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { config } from "../../../common/constants/config";
import Button from "react-bootstrap/Button";
import ButtonGroup from "react-bootstrap/ButtonGroup";

function PageSize({
  text,
  setPageSize,
  current_lines,
}: {
  text: number;
  current_lines: number;
  setPageSize: Function;
}): JSX.Element {
  if (text === current_lines) {
    return (
      <Button
        style={{
          backgroundColor: config.secondary_color,
          borderColor: "white",
        }}
        className="disabled"
      >
        {text}
      </Button>
    );
  } else {
    return (
      <Button
        style={{ backgroundColor: config.default_color, borderColor: "white" }}
        onClick={() => {
          setPageSize(text);
        }}
      >
        {text}
      </Button>
    );
  }
}

function LineSelector({
  lines_per_page,
  setPageSize,
}: {
  lines_per_page: number;
  setPageSize: Function;
}): JSX.Element {
  const possible_values = [10, 25, 50];
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        justifyContent: "flex-end",
      }}
    >
      lines per page:
      <ButtonGroup>
        {possible_values.map((text, index) => (
          <PageSize
            text={text}
            key={index}
            current_lines={lines_per_page}
            setPageSize={setPageSize}
          />
        ))}
      </ButtonGroup>
    </div>
  );
}

export default LineSelector;
