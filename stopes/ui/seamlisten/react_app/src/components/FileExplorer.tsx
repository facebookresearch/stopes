// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { useCallback, useEffect, useState } from "react";

import Button from "react-bootstrap/Button";
import { default as BCol } from "react-bootstrap/Col";
import Form from "react-bootstrap/Form";
import { default as BRow } from "react-bootstrap/Row";

import {
  Location,
  useLoaderData,
  useNavigate,
  useNavigation,
} from "react-router-dom";
import WaveSurferComponent from "../common/components/audio/WaveSurfer";
import InnerScale from "../common/components/spinners/spinner";
import { config } from "../common/constants/config";
import fetchFiles from "../common/fetchers/mining_result";
import { LineResult } from "../common/types/api";
import Help from "./fileviewer/FileExplorerHelp";
import Table from "./fileviewer/Table";

import { text_to_audio } from "../common/components/audio/audioquery_constructor";

const FILENAME_PARAM = "file";
const PAGENUMBER_PARAM = "page";
const NUMBERLINES_PARAM = "lines";

type LoaderReturn = {
  filename: string;
  pageNumber: number;
  numberLines: number;
  files: LineResult[];
  audioBlob: Blob;
  error: any;
};

function parseParams(searchParams) {
  return {
    filename: searchParams.get(FILENAME_PARAM)
      ? decodeURIComponent(searchParams.get(FILENAME_PARAM)).trim()
      : config.default_path,
    pageNumber: parseInt(searchParams.get(PAGENUMBER_PARAM), 10) || 1,
    numberLines: parseInt(searchParams.get(NUMBERLINES_PARAM), 10) || 10,
  };
}

function parseLocation(location: Location) {
  if (!location) {
    return null;
  }
  return parseParams(new URLSearchParams(location.search));
}

export async function loader({ request }): Promise<LoaderReturn> {
  const url = new URL(request.url);

  const { filename, numberLines, pageNumber } = parseParams(url.searchParams);
  const toRet = {
    filename,
    numberLines,
    pageNumber,
    files: [],
    audioBlob: undefined,
    error: null,
  };

  try {
    if (
      filename.endsWith("tsv.gz") ||
      filename.endsWith(".tsv") ||
      filename.endsWith(".zip")
    ) {
      const files: LineResult[] = await fetchFiles({
        gz_path: filename,
        start_idx: numberLines * (pageNumber - 1),
        end_idx: numberLines * pageNumber,
      });
      toRet.files = files;
      return toRet;
    }

    const audioResult = await text_to_audio(filename, 1);
    if (audioResult) {
      toRet.audioBlob = audioResult;
      return toRet;
    }
    toRet.error = { data: { detail: "Unknown File Format" } };
  } catch (err) {
    console.error(err);
    toRet.error = err;
  }
  return toRet;
}

function Error({ error }) {
  const msg = error.data
    ? error.data.detail
    : error.statusText || "Something went wrong.";
  // Uncaught ReferenceError: path is not defined
  return (
    <div>
      <p style={{ color: "red" }}>{msg}</p>
    </div>
  );
}

function useFileNavigate() {
  const navigate = useNavigate();
  return (file: string, page: number, numberLines: number) =>
    navigate(
      `?${FILENAME_PARAM}=${encodeURIComponent(file)}&${PAGENUMBER_PARAM}=${
        page || 0
      }&${NUMBERLINES_PARAM}=${numberLines || 0}`
    );
}

const Files = (): JSX.Element => {
  const [displayHelper, setDisplayHelper] = useState(false);
  const navigate = useFileNavigate();
  let { filename, pageNumber, numberLines, files, audioBlob, error } =
    useLoaderData() as LoaderReturn;
  const [newFilename, setNewFilename] = useState(
    filename || config.default_path
  );

  // if we have a location, we are in a transition between two urls
  const navigation = useNavigation();
  const locationParams = parseLocation(navigation.location);
  if (locationParams) {
    filename = locationParams.filename;
    pageNumber = locationParams.pageNumber;
    numberLines = locationParams.numberLines;
  }
  const loading = !!navigation.location;

  // in some navigation events (like back/forward navigation, the component is not remounted)
  // so we need to reset the "default" for the filename form.
  useEffect(() => setNewFilename(filename), [filename]);

  const setFilenameEventHandler = useCallback(
    (evt) => setNewFilename(evt.target.value),
    [setNewFilename]
  );
  const setPageNumber = useCallback(
    (pg: number) => navigate(newFilename, pg, numberLines),
    [newFilename, numberLines, navigate]
  );
  const setFilename = useCallback(
    () => navigate(newFilename, pageNumber, numberLines),
    [numberLines, pageNumber, newFilename, navigate]
  );
  const setNumberLines = useCallback(
    (size: number) => navigate(newFilename, pageNumber, size),
    [newFilename, pageNumber, navigate]
  );
  const fileInputHandleChange = useCallback(
    (evt) => {
      if (evt.key === "Enter") {
        setFilename();
      }
    },
    [setFilename]
  );

  return (
    <div style={{ marginTop: "10px" }}>
      <Form as={BRow}>
        <BCol sm="8">
          <Form.Group className="row mb-3">
            <BCol sm="2">
              <Form.Label>Filename</Form.Label>
            </BCol>
            <BCol sm="10">
              <Form.Control
                disabled={loading}
                type="text"
                onChange={setFilenameEventHandler}
                value={newFilename}
                onKeyDown={fileInputHandleChange}
                size="sm"
              />
              <Form.Text className="text-muted" />
            </BCol>
          </Form.Group>
        </BCol>
        <BCol sm="2" className="justify-content-start">
          <Button
            style={{
              backgroundColor: loading
                ? config.secondary_color
                : config.default_color,
              borderColor: "white",
            }}
            disabled={loading}
            onClick={setFilename}
          >
            Fetch!
          </Button>
          <Button
            onClick={() => {
              setDisplayHelper(!displayHelper);
            }}
            aria-controls="help-text"
            aria-expanded={displayHelper}
            style={{
              backgroundColor: config.default_color,
              borderColor: "white",
            }}
          >
            {" "}
            Help
          </Button>
        </BCol>
      </Form>
      <Help displayHelper={displayHelper} />
      {loading ? (
        <InnerScale loading={loading} />
      ) : error ? (
        <Error error={error} />
      ) : (
        <>
          <Table
            items={files}
            page={pageNumber}
            lines_per_page={numberLines}
            setNumberLines={setNumberLines}
            setPageNumber={setPageNumber}
          />
          {!!audioBlob && (
            <WaveSurferComponent
              key={filename}
              blob={audioBlob}
              area="unused_area123456789"
              waveformKeyEvent={null}
              waveformKeyEventHandler={() => {}}
              focusedRowID={""}
              currentRowID={""}
              currentPlayingID={""}
              currentPlayingIDHandler={() => {}}
            />
          )}
        </>
      )}
    </div>
  );
};

export default Files;
