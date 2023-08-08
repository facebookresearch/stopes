import  { useState } from 'react';
import './Notebook.css'; // Import your custom styling

export interface Folder {
  name: string;
  type: 'folder';
  content: File[];
}

interface File {
  name: string;
  type: 'file';
  content: string;
}

interface Props {
  folders: Folder[];
}

const Notebook: React.FC<Props> = ({ folders }) => {
  const [currentFolder, setCurrentFolder] = useState<Folder | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFolderClick = (folder: Folder) => {
    setCurrentFolder(folder);
    setSelectedFile(null);
  };

  const handleFileClick = (file: File) => {
    setSelectedFile(file);
  };

  return (
    <div className="notebook-container">
      <FolderNavigation
        folders={folders}
        currentFolder={currentFolder}
        onSelectFolder={handleFolderClick}
      />
      <div className="notebook-content">
        {currentFolder ? (
          <FolderView folder={currentFolder} onSelectFile={handleFileClick} />
        ) : (
          <div>Select a folder to view its contents</div>
        )}
        <FileViewer file={selectedFile} />
      </div>
    </div>
  );
};

interface FolderNavigationProps {
  folders: Folder[];
  currentFolder: Folder | null;
  onSelectFolder: (folder: Folder) => void;
}

const FolderNavigation: React.FC<FolderNavigationProps> = ({
  folders,
  currentFolder,
  onSelectFolder,
}) => (
  <div className="folder-navigation">
    {folders.map((folder) => (
      <div
        key={folder.name}
        onClick={() => onSelectFolder(folder)}
        className={`folder-item ${currentFolder === folder ? 'selected' : ''}`}
      >
        {folder.name}
      </div>
    ))}
  </div>
);

interface FileViewerProps {
  file: File | null;
}

const FileViewer: React.FC<FileViewerProps> = ({ file }) => (
  <div className="file-viewer">
    {file ? (
      <pre>{file.content}</pre>
    ) : (
      <div>Select a file to view its content</div>
    )}
  </div>
);

interface FolderViewProps {
  folder: Folder;
  onSelectFile: (file: File) => void;
}

const FolderView: React.FC<FolderViewProps> = ({ folder, onSelectFile }) => (
  <div>
    {folder.content.map((file) => (
      <div
        key={file.name}
        onClick={() => onSelectFile(file)}
        className="file-item"
      >
        {file.name}
      </div>
    ))}
  </div>
);

export default Notebook;
