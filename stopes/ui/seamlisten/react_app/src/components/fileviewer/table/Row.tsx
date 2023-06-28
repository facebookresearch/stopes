// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import PlayArea from "../../../common/components/audio/PlayArea";
import { CellRenderProps, RowProps } from "../../../common/types/api";

function CellRender({
  object,
  place,
  waveformKeyEvent,
  waveformKeyEventHandler,
  focusedRowID,
  currentRowID,
  currentPlayingID,
  currentPlayingIDHandler,
}: CellRenderProps): JSX.Element {
  if (typeof object === "number") {
    return <span>{object.toString()}</span>;
  }
  switch (object.kind) {
    case "audio":
      return (
        <PlayArea
          path={object.path}
          start={object.start}
          end={object.end}
          sampling={object.sampling}
          key={
            object.path +
            "_" +
            String(object.start) +
            "_" +
            String(object.end) +
            place
          }
          context_size={1}
          waveformKeyEvent={waveformKeyEvent}
          waveformKeyEventHandler={waveformKeyEventHandler}
          focusedRowID={focusedRowID}
          currentRowID={currentRowID}
          currentPlayingID={currentPlayingID}
          currentPlayingIDHandler={currentPlayingIDHandler}
        ></PlayArea>
      );
    case "text":
      return <span>{object.content}</span>;
  }
}

export const Row = ({
  item,
  waveformKeyEvent,
  waveformKeyEventHandler,
  keyDownHandler,
  focusedRowID,
  currentPlayingID,
  currentPlayingIDHandler,
  rowKey,
}: RowProps): JSX.Element => {
  const navControlProps = {
    waveformKeyEvent: waveformKeyEvent,
    waveformKeyEventHandler: waveformKeyEventHandler,
    focusedRowID: focusedRowID,
    currentRowID: rowKey,
    currentPlayingID: currentPlayingID,
    currentPlayingIDHandler: currentPlayingIDHandler,
  };

  return (
    <tr
      key={rowKey}
      id={rowKey}
      tabIndex={0}
      onKeyDown={(e) => keyDownHandler(e, rowKey)}
    >
      {item.columns.map((item_cell, item_cell_index) => (
        <td key={item_cell_index}>
          <CellRender
            object={item_cell}
            place={item_cell_index.toString()}
            {...navControlProps}
          />
        </td>
      ))}
    </tr>
  );
};
