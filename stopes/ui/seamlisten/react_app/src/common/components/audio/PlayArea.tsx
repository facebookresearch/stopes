// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { useCallback, useState } from "react";
import fetchAudio from "../../fetchers/audio";
import { AudioQuery, PlayAreaProps } from "../../types/api";
import InnerScale from "../spinners/spinner";
import get_area_name from "./area_constructor";
import "./PlayArea.css";
import WaveSurferComponent from "./WaveSurfer";

function Clipboard({ txt }) {
  const cb = useCallback(() => {
    navigator.clipboard.writeText(txt);
  }, [txt]);
  return <div onClick={cb}>{txt}</div>;
}

function PlayArea({
  path,
  start,
  end,
  sampling,
  context_size = 0,
  waveformKeyEvent = null,
  waveformKeyEventHandler,
  focusedRowID = "",
  currentRowID = "",
  currentPlayingID = "",
  currentPlayingIDHandler,
}: PlayAreaProps): JSX.Element {
  const [audioFile, setAudioFile] = useState<Blob>();
  const [errorMessage, setErrorMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const query: AudioQuery = {
    path: path,
    start: start,
    end: end,
    sampling: sampling,
    context_size: context_size,
  };
  const area = get_area_name(query);

  if (context_size === undefined) {
    context_size = 0;
  }
  async function getset_audio(query): Promise<void> {
    setIsLoading(true);
    try {
      const blob = await fetchAudio(query);
      if (!!blob) {
        setAudioFile(blob);
      }
    } catch (err) {
      setErrorMessage(err.statusText);
    } finally {
      setIsLoading(false);
    }
  }
  if (errorMessage !== "") {
    return <p color="red">{errorMessage}</p>;
  }
  const base = path + " " + start + " " + end;

  if (!!audioFile) {
    return (
      <>
        <Clipboard txt={base} />
        <WaveSurferComponent
          blob={audioFile}
          area={area}
          context_size={context_size}
          waveformKeyEvent={waveformKeyEvent}
          waveformKeyEventHandler={waveformKeyEventHandler}
          focusedRowID={focusedRowID}
          currentRowID={currentRowID}
          currentPlayingID={currentPlayingID}
          currentPlayingIDHandler={currentPlayingIDHandler}
        />
      </>
    );
  }
  if (isLoading) {
    return (
      <>
        <Clipboard txt={base} />
        <InnerScale loading={isLoading} />
      </>
    );
  }
  return (
    <div className="ButtonEncapsulation">
      <button
        onClick={() => {
          getset_audio(query);
        }}
        className="Button"
      >
        {base}
      </button>
    </div>
  );
}

export default PlayArea;
