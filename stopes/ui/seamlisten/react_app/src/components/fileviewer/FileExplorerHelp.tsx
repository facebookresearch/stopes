// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import Collapse from "react-bootstrap/Collapse";

const Help = ({ displayHelper }: { displayHelper: boolean }) => {
  return displayHelper ? (
    <Collapse in={true}>
      <div id="help-text" style={{ textAlign: "start" }}>
        <h2>Usage</h2>
        <h5>Load File</h5>
        <ul>
          <li>
            Input{" "}
            <span
              style={{
                fontFamily: "monospace",
                backgroundColor: "lightgrey",
              }}
            >
              /path/to/file.tsv.gz
            </span>{" "}
            : load annotation file. Also works with .tsv file extension.
          </li>
          <li>
            Input:{" "}
            <span
              style={{
                fontFamily: "monospace",
                backgroundColor: "lightgrey",
              }}
            >
              /path/to/audio/file start end (optional field){" "}
            </span>
            : load the audio, where start and end are sample numbers
          </li>
          <li>
            Input:{" "}
            <span
              style={{
                fontFamily: "monospace",
                backgroundColor: "lightgrey",
              }}
            >
              /path/to/audio/file|start|end
            </span>
            : load the audio, assuming 16kHz sampling (i.e. start and end are
            given in ms)
          </li>
        </ul>
        <h5>Keyboard Shortcuts</h5>
        <ul>
          <li>
            <b>tab</b>: enter keyboard control mode | go to next component;
          </li>
          <li>
            <b>return</b>: load audiofile from path;
          </li>
          <li>
            <b>ArrowUp</b>: move to the row above;
          </li>
          <li>
            <b>ArrowDown</b>: move to the row below;
          </li>
          <li>
            <b>t</b>: jump to the top row;
          </li>
          <li>
            <b>b</b>: jump to the bottom row;
          </li>
          <li>
            <b>p</b>: toggle play / pause of the current track in focus;
          </li>
          <li>
            <b>s</b>: reset the current track in focus to start;
          </li>
        </ul>
      </div>
    </Collapse>
  ) : null;
};

export default Help;
