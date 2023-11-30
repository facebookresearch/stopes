// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// calls

import { KeyboardEvent } from "react";

interface AudioQuery {
  path: string;
  start?: number;
  end?: number;
  sampling?: string;
  context_size?: number;
}

interface AnnotationQuery {
  gz_path: string;
  start_idx: number;
  end_idx: number;
}

interface LineQuery {
  gz_path: string;
}

interface UploadQuery {
  blob: Blob;
}

// answers:
type TextAnswer = {
  kind: "text";
  content: string;
};

type Audio = {
  sampling: string;
  kind: "audio";
  path: string;
  start: number;
  end: number;
};

interface LineResult {
  columns: Array<number | TextAnswer | Audio>;
}

// controls:
type CurrentPlayingIDHandler = (params: string) => void;
type KeyDownHandler = (
  event: KeyboardEvent<HTMLTableRowElement>,
  key: string
) => void;
type WaveformKeyEventHandler = (
  event: KeyboardEvent<HTMLTableRowElement>
) => void;

interface NavControl {
  waveformKeyEvent?: KeyboardEvent<HTMLTableRowElement>;
  waveformKeyEventHandler?: WaveformKeyEventHandler;
  keyDownHandler?: KeyDownHandler;
  focusedRowID?: string;
  currentRowID?: string;
  currentPlayingID?: string;
  currentPlayingIDHandler?: CurrentPlayingIDHandler;
}

interface PlayAreaProps extends AudioQuery, NavControl {}

type RowProps = NavControl & {
  rowKey: string;
  item: LineResult;
};

interface CellRenderProps extends NavControl {
  object: number | TextAnswer | Audio;
  place: string;
}

type FaissResult = {
  audio: Audio;
  distance: number;
};

export {
  AudioQuery,
  AnnotationQuery,
  LineQuery,
  UploadQuery,
  LineResult,
  Audio,
  TextAnswer,
  FaissResult,
  PlayAreaProps,
  RowProps,
  CurrentPlayingIDHandler,
  CellRenderProps,
};
