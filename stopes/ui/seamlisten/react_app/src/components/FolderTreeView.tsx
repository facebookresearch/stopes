import { useState } from "react";
import TreeView from "@mui/lab/TreeView";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import TreeItem from "@mui/lab/TreeItem";
import { processFolder } from "../common/fetchers/mining_result";
import PlayArea from "../common/components/audio/PlayArea";
import { text_to_audio } from "../common/components/audio/audioquery_constructor";
import WaveSurferComponent from "../common/components/audio/WaveSurfer";

const FolderTreeView = ({ folderContents }) => {
  const [expandedFolders, setExpandedFolders] = useState(new Set());
  const [selectedAudioFile, setSelectedAudioFile] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);

  const handleNodeToggle = (event, nodeIds) => {
    const clickedNodeId = nodeIds[0];

    if (expandedFolders.has(clickedNodeId)) {
      // Node is expanded, collapse it
      expandedFolders.delete(clickedNodeId);
    } else {
      // Node is collapsed, expand it and fetch its contents
      expandedFolders.add(clickedNodeId);
      fetchFolderContents(clickedNodeId);
    }

    setExpandedFolders(new Set(expandedFolders));
  };

  const handleAudioFileClick = async(audioFile) => {
    setSelectedAudioFile(audioFile);
    const audioResult = await text_to_audio(audioFile, 1);
    if (audioResult) {
      setAudioBlob(audioResult);
    }
  };

  const fetchFolderContents = async (folderPath) => {
    try {
      // Make a request to fetch the contents of the clicked folderPath
      const folderContents = await processFolder(folderPath);

      // Handle the fetched data and update your state accordingly
      // For example, you can update your state with the new folder contents
      // and re-render the component.
    } catch (error) {
      console.error("Error fetching folder contents:", error);
    }
  };

  const DownloadButton = ({ blob, filename }) => {
    const download = () => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
    };
  
    return (
      <button onClick={download}
        style={{
          marginTop: "10px",
          border: "none",
          padding: "10px",
          borderRadius: "5px",
          backgroundColor: "#4CAF50",
          color: "white",
          textAlign: "center",
          textDecoration: "none",
          display: "inline-block",
          fontSize: "16px",
        }}

        >
        Download
      </button>
    );
  }




  const renderFolderTree = (folderData) => {
    if (!folderData) {
      return null;
    }

    return (
      <TreeView
        aria-label="file system navigator"
        defaultCollapseIcon={<ExpandMoreIcon />}
        defaultExpandIcon={<ChevronRightIcon />}
        sx={{
          height: 240,
          flexGrow: 1,
          maxWidth: 400,
          overflowY: "auto",
          display: "flex",
        }}
      >
        <div style={{ flex: 1 }}>
          <TreeItem
            key={folderData.folder}
            nodeId={folderData.folder}
            label={folderData.folder}
            onClick={(event) => handleNodeToggle(event, [folderData.folder])}
          >
            {folderData.subfolders && folderData.subfolders.length > 0 && (
              folderData.subfolders.map((subfolder) => renderFolderTree(subfolder))
            )}
            {folderData.audio_files && folderData.audio_files.length > 0 && (
              folderData.audio_files.map((audioFile) => (
                <TreeItem
                  key={audioFile}
                  nodeId={audioFile}
                  label={audioFile}
                  onClick={() => handleAudioFileClick(
                    `${folderData.folder}/${audioFile}`
                  )}
                />
              ))
            )}
          </TreeItem>
        </div>
      </TreeView>
    );
  };

  return (
    <div>
      {renderFolderTree(folderContents)}
      {!!audioBlob && (
        <>
            <WaveSurferComponent
              key={selectedAudioFile}
              blob={audioBlob}
              area="unused_area123456789"
              waveformKeyEvent={null}
              waveformKeyEventHandler={() => {}}
              focusedRowID={""}
              currentRowID={""}
              currentPlayingID={""}
              currentPlayingIDHandler={() => {}}
            />
            <DownloadButton blob={audioBlob} filename={selectedAudioFile} />
            </>
          )}
    </div>
  );
};

export default FolderTreeView;


