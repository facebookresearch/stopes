// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { WaveSurfer, WaveForm, Region } from "wavesurfer-react";
import RegionsPlugin from "wavesurfer.js/dist/plugin/wavesurfer.regions.min";
import "./WaveSurfer.css";

// import TimelinePlugin from "wavesurfer.js/dist/plugin/wavesurfer.timeline.min";
import { useCallback, useEffect, useRef, useState } from "react";
import { config } from "../../constants/config";
import { FaPauseCircle, FaPlayCircle } from "react-icons/fa";

function WaveSurferComponent({
  blob,
  area,
  context_size = 0,
  waveformKeyEvent,
  waveformKeyEventHandler,
  focusedRowID,
  currentRowID,
  currentPlayingID,
  currentPlayingIDHandler,
}) {
  // blob (string): url to blob
  // area (string): name of the area to track with spinner
  //                also used to create the different css ids
  // waveformKeyEvent (KeyboardEvent): keyboard event relevant to waveform control
  // waveformKeyEventHandler (function): setWaveformKeyEvent passed down from Table component
  // focusedRowID (string): the row id that's currently in focus (by mouse or keyboard)
  // currentRowID (string): the current row id where this WaveSurferCompoment located in.
  // currentPlayingID (string): the waveform_id that's currently being played
  // handler (function): setPlayList function passed down from Table component
  const wavesurferRef = useRef();
  // Transform the name of the file into a valid HTML identifier
  const waveform_id =
    "wsurf" + encodeURIComponent(area).replace(/\.|%[0-9a-zA-Z]{2}/gi, "_");
  const timeline_id = waveform_id + "_timeline";
  const [regions, setRegions] = useState([]);
  // Is button in play or not play (pause) state.
  const [buttonIsPlay, setButtonIsPlay] = useState(false);
  const plugins = [
    {
      plugin: RegionsPlugin,
      options: { dragSelection: false },
    },
    /* {
      plugin: TimelinePlugin,
      options: {
        container: timeline_id,
      },
    },
      */
  ];
  const handleWSMount = (waveSurfer) => {
    wavesurferRef.current = waveSurfer;
    if (wavesurferRef.current) {
      if (!!blob) {
        wavesurferRef.current.on("ready", () => {
          let ws = wavesurferRef.current;
          if (context_size > 0) {
            let duration = ws.getDuration();
            ws.seekTo(context_size / duration);

            let regs = [
              {
                id: "pre",
                start: 0,
                end: 1,
                color: "rgba(0, 0, 0, .5)",
                drag: false,
                resize: false,
              },
              {
                id: "post",
                start: duration - 1,
                end: duration,
                color: "rgba(0, 0, 0, .5)",
                drag: false,
                resize: false,
              },
            ];
            setRegions(regs);
          }
        });
        wavesurferRef.current.loadBlob(blob);
        if (window) {
          window.surferidze = wavesurferRef.current;
        }
        wavesurferRef.current.on("finish", () => {
          setButtonIsPlay(false);
        });
      }
    }
  };

  const togglePlayPause = useCallback(() => {
    // When button not in playing mode, a click would trigger it to play,
    // so we set the currentPlayingID to this waveform_id
    if (!buttonIsPlay) {
      currentPlayingIDHandler(() => {
        return waveform_id;
      });
    }
    setButtonIsPlay((currentButtonIsPlay) => !currentButtonIsPlay);
    wavesurferRef.current.playPause();
  }, [buttonIsPlay, wavesurferRef, waveform_id, currentPlayingIDHandler]);

  const resetWaveformKeyEvent = useCallback(() => {
    waveformKeyEventHandler(null);
  }, [waveformKeyEventHandler]);

  useEffect(() => {
    // Pause the current waveform, if it's no longer the currentPlayingID
    if (currentPlayingID !== waveform_id) {
      setButtonIsPlay(false);
      wavesurferRef.current.pause();
    }
  }, [currentPlayingID, waveform_id]);

  function renderPlayButton() {
    if (!buttonIsPlay) {
      return (
        <FaPlayCircle className="playPauseIcon" onClick={togglePlayPause} />
      );
    }
    return (
      <FaPauseCircle className="playPauseIcon" onClick={togglePlayPause} />
    );
  }

  useEffect(() => {
    if (!!waveformKeyEvent && focusedRowID === currentRowID) {
      switch (waveformKeyEvent.key) {
        case "p":
          togglePlayPause();
          break;
        case "s":
          wavesurferRef.current.seekTo(0);
          break;
        default:
          break;
      }
      resetWaveformKeyEvent();
    }
  }, [
    waveformKeyEvent,
    focusedRowID,
    currentRowID,
    togglePlayPause,
    resetWaveformKeyEvent,
  ]);

  return (
    <div>
      {!!blob ? (
        <>
          <WaveSurfer plugins={plugins} onMount={handleWSMount}>
            <WaveForm
              id={waveform_id}
              progressColor={config.default_color}
              cursorColor="black"
            >
              {regions.map((regionProps) => (
                <Region key={regionProps.id} {...regionProps} />
              ))}
            </WaveForm>
            <div id={timeline_id} />
          </WaveSurfer>
          {renderPlayButton()}
        </>
      ) : (
        <></>
      )}
    </div>
  );
}

export default WaveSurferComponent;
