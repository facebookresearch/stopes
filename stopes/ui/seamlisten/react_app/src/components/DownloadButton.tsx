import Tooltip from '@mui/material/Tooltip';
import IconButton from '@mui/material/IconButton';
import DownloadIcon from '@mui/icons-material/CloudDownload';

const DownloadButton = ({ id, blob }) => {
  const downloadUrl = window.URL.createObjectURL(blob);
  const fileName = `seamless-${id}.wav`;

  return (
    <Tooltip title='Download segment.'>
      <IconButton
        aria-label='download'
        size='small'
        href={downloadUrl}
        download={fileName}
      >
        <DownloadIcon fontSize='inherit'/>
      </IconButton>
    </Tooltip>
  );
};

export default DownloadButton;
