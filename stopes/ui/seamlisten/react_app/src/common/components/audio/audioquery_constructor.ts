// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import fetchAudio from "../../fetchers/audio";

async function text_to_audio_with_callbacks(
  inputQuery: string,
  setErrorMessage: Function,
  setAudioFile: Function,
  context_size: number
) {
  try {
    const blob = await text_to_audio(inputQuery, context_size);
    setAudioFile(blob);
    return true;
  } catch (er) {
    setErrorMessage(er.data ? er.data.detail : er.statusText);
  }
  return false;
}

async function text_to_audio(
  inputQuery: string,
  context_size: number
): Promise<Blob | void> {
  const space_split = inputQuery.split(" ");

  if (space_split.length === 1) {
    const ext = inputQuery.slice(-4);
    if (ext === ".mp3" || ext === ".wav" || ext === ".ogg") {
      return await fetchAudio({
        path: inputQuery,
      });
    }
  }

  if (space_split.length === 3 || space_split.length === 4) {
    // if trailing item, dismiss it
    return await fetchAudio({
      path: space_split[0],
      start: parseInt(space_split[1]),
      end: parseInt(space_split[2]),
      context_size: context_size,
      sampling: "wav",
    });
  }
  const pipe_split = inputQuery.split("|");
  if (pipe_split.length === 3) {
    return fetchAudio({
      path: pipe_split[0],
      start: parseInt(pipe_split[1]),
      end: parseInt(pipe_split[2]),
      context_size: context_size,
      sampling: "ms",
    });
  }
  return null;
}

export { text_to_audio, text_to_audio_with_callbacks };
