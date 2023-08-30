import TreeView from "@mui/lab/TreeView";
import TreeItem from "@mui/lab/TreeItem";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import { FolderStructure } from "../../common/types/api";

type FileTreeProps = {
  folderData: FolderStructure | null;
};

const FileTree: React.FC<FileTreeProps> = ({ folderData }) => {
  // Recursive function to render TreeItems
  const renderTree = (folderData: FolderStructure, nodeId: string) => {
    return (
      <TreeItem nodeId={nodeId} label={folderData.folder}>
        {folderData.subfolders &&
          folderData.subfolders.map((subfolder, index) =>
            renderTree(subfolder, `${nodeId}-${index}`)
          )}
        {folderData.audio_files &&
          folderData.audio_files.map((file, index) => (
            <TreeItem nodeId={`${nodeId}-file-${index}`} label={file} />
          ))}
      </TreeItem>
    );
  };

  return (
    <TreeView
      aria-label="file system navigator"
      defaultCollapseIcon={<ExpandMoreIcon />}
      defaultExpandIcon={<ChevronRightIcon />}
    >
      {renderTree(folderData, "root")}
    </TreeView>
  );
};

export default FileTree;
