// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { KeyboardEvent, useRef, useState } from "react";
import { hashString } from "react-hash-string";
import { Audio, LineResult, TextAnswer } from "../../common/types/api";

import "../../common/components/table/table.css";
import LineSelector from "./table/LinesSelector";
import Pagination from "./table/Pagination";
import { Row } from "./table/Row";

type TableProps = {
  items: LineResult[];
  page: number;
  lines_per_page: number;
  setNumberLines: Function;
  setPageNumber: Function;
};

function Table({
  items,
  page,
  lines_per_page,
  setNumberLines,
  setPageNumber,
}: TableProps): JSX.Element {
  const [currentPlayingID, setCurrentPlayingID] = useState("");
  const [waveformKeyEvent, setWaveformKeyEvent] = useState(null);
  const [focusedRowID, setFocusedRowID] = useState("");
  const tbodyRef = useRef(null);

  function handleKeyDown(
    e: KeyboardEvent<HTMLTableRowElement>,
    rowKey: string
  ) {
    const currentRow = tbodyRef.current?.children.namedItem(rowKey);
    switch (e.key) {
      // Table level navigations:
      case "t":
        tbodyRef.current?.firstElementChild?.focus();
        break;
      case "ArrowUp":
        currentRow?.previousElementSibling?.focus();
        break;
      case "ArrowDown":
        currentRow?.nextElementSibling?.focus();
        break;
      case "b":
        tbodyRef.current?.lastElementChild?.focus();
        break;
      // Waveform controls:
      case "p":
      case "s":
        setWaveformKeyEvent(e);
        setFocusedRowID(document.activeElement.id);
        break;
      default:
        break;
    }
  }

  function extractInfoFromObject(object: number | TextAnswer | Audio): string {
    if (typeof object === "number") {
      return object.toString();
    }
    switch (object.kind) {
      case "audio":
        return object.path + String(object.start) + String(object.end);
      case "text":
        return object.content;
      default:
        return "";
    }
  }

  function extractKeyFromItem(item: LineResult): string {
    return hashString(
      item.columns
        .map((item_cell) => extractInfoFromObject(item_cell))
        .join(" ")
    ).toString();
  }

  if (items.length > 0) {
    const first_row_cells = items[0].columns;
    return (
      <div style={{ marginTop: "10px" }}>
        <LineSelector
          lines_per_page={lines_per_page}
          setPageSize={setNumberLines}
        />
        <table className="table">
          <thead>
            <tr>
              {first_row_cells.map((cell_item, cell_item_index) => (
                <th key={cell_item_index} scope="col">
                  Column {cell_item_index + 1}
                </th>
              ))}
            </tr>
          </thead>
          <tbody ref={tbodyRef}>
            {items.map((row_item) => (
              <Row
                key={extractKeyFromItem(row_item)}
                rowKey={extractKeyFromItem(row_item)}
                item={row_item}
                waveformKeyEvent={waveformKeyEvent}
                waveformKeyEventHandler={setWaveformKeyEvent}
                keyDownHandler={handleKeyDown}
                focusedRowID={focusedRowID}
                currentPlayingID={currentPlayingID}
                currentPlayingIDHandler={setCurrentPlayingID}
              />
            ))}
          </tbody>
        </table>
        <Pagination page={page} setPage={setPageNumber} />
      </div>
    );
  }
}

export default Table;
