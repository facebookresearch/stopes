import {  useState } from "react";
import { text_to_audio } from "../common/components/audio/audioquery_constructor";
import WaveSurferComponent from "../common/components/audio/WaveSurfer";
import { useSpring, animated } from '@react-spring/web';
import SvgIcon, { SvgIconProps } from '@mui/material/SvgIcon';
import { TransitionProps } from '@mui/material/transitions';
import Collapse from '@mui/material/Collapse';
import { alpha, styled } from '@mui/material/styles';
import { TreeView } from '@mui/x-tree-view/TreeView';
import { TreeItem, TreeItemProps, treeItemClasses } from '@mui/x-tree-view/TreeItem';
import React from 'react';


function MinusSquare(props: SvgIconProps) {
  return (
    <SvgIcon fontSize="inherit" style={{ width: 14, height: 14 }} {...props}>
      {/* tslint:disable-next-line: max-line-length */}
      <path d="M22.047 22.074v0 0-20.147 0h-20.12v0 20.147 0h20.12zM22.047 24h-20.12q-.803 0-1.365-.562t-.562-1.365v-20.147q0-.776.562-1.351t1.365-.575h20.147q.776 0 1.351.575t.575 1.351v20.147q0 .803-.575 1.365t-1.378.562v0zM17.873 11.023h-11.826q-.375 0-.669.281t-.294.682v0q0 .401.294 .682t.669.281h11.826q.375 0 .669-.281t.294-.682v0q0-.401-.294-.682t-.669-.281z" />
    </SvgIcon>
  );
}

function PlusSquare(props: SvgIconProps) {
  return (
    <SvgIcon fontSize="inherit" style={{ width: 14, height: 14 }} {...props}>
      {/* tslint:disable-next-line: max-line-length */}
      <path d="M22.047 22.074v0 0-20.147 0h-20.12v0 20.147 0h20.12zM22.047 24h-20.12q-.803 0-1.365-.562t-.562-1.365v-20.147q0-.776.562-1.351t1.365-.575h20.147q.776 0 1.351.575t.575 1.351v20.147q0 .803-.575 1.365t-1.378.562v0zM17.873 12.977h-4.923v4.896q0 .401-.281.682t-.682.281v0q-.375 0-.669-.281t-.294-.682v-4.896h-4.923q-.401 0-.682-.294t-.281-.669v0q0-.401.281-.682t.682-.281h4.923v-4.896q0-.401.294-.682t.669-.281v0q.401 0 .682.281t.281.682v4.896h4.923q.401 0 .682.281t.281.682v0q0 .375-.281.669t-.682.294z" />
    </SvgIcon>
  );
}

function CloseSquare(props: SvgIconProps) {
  return (
    <SvgIcon
      className="close"
      fontSize="inherit"
      style={{ width: 14, height: 14 }}
      {...props}
    >
      {/* tslint:disable-next-line: max-line-length */}
      <path d="M17.485 17.512q-.281.281-.682.281t-.696-.268l-4.12-4.147-4.12 4.147q-.294.268-.696.268t-.682-.281-.281-.682.294-.669l4.12-4.147-4.12-4.147q-.294-.268-.294-.669t.281-.682.682-.281.696 .268l4.12 4.147 4.12-4.147q.294-.268.696-.268t.682.281 .281.669-.294.682l-4.12 4.147 4.12 4.147q.294.268 .294.669t-.281.682zM22.047 22.074v0 0-20.147 0h-20.12v0 20.147 0h20.12zM22.047 24h-20.12q-.803 0-1.365-.562t-.562-1.365v-20.147q0-.776.562-1.351t1.365-.575h20.147q.776 0 1.351.575t.575 1.351v20.147q0 .803-.575 1.365t-1.378.562v0z" />
    </SvgIcon>
  );
}

function TransitionComponent(props: TransitionProps) {
  const style = useSpring({
    from: {
      opacity: 0,
      transform: 'translate3d(20px,0,0)',
    },
    to: {
      opacity: props.in ? 1 : 0,
      transform: `translate3d(${props.in ? 0 : 20}px,0,0)`,
    },
  });

  return (
    <animated.div style={style}>
      <Collapse {...props} />
    </animated.div>
  );
}

const CustomTreeItem = React.forwardRef(
  (props: TreeItemProps, ref: React.Ref<HTMLLIElement>) => (
    <TreeItem {...props} TransitionComponent={TransitionComponent} ref={ref} />
  ),
);

const StyledTreeItem = styled(CustomTreeItem)(({ theme }) => ({
  [`& .${treeItemClasses.iconContainer}`]: {
    '& .close': {
      opacity: 0.3,
    },
  },
  [`& .${treeItemClasses.group}`]: {
    marginLeft: 15,
    paddingLeft: 18,
    borderLeft: `1px dashed ${alpha(theme.palette.text.primary, 0.4)}`,
  },
}));

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
      aria-label="customized"
      defaultExpanded={['1']}
      defaultCollapseIcon={<MinusSquare />}
      defaultExpandIcon={<PlusSquare />}
      defaultEndIcon={<CloseSquare />}
      sx={{ height: 264, flexGrow: 1, maxWidth: 400, overflowY: 'auto' }}
      >
        <div style={{ flex: 1 }}>
          <StyledTreeItem
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
                <StyledTreeItem
                  key={audioFile}
                  nodeId={audioFile}
                  label={audioFile}
                  onClick={() => handleAudioFileClick(
                    `${folderData.folder}/${audioFile}`
                  )}
                />
              ))
            )}
          </StyledTreeItem>
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


