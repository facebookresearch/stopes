import { config } from "../constants/config";
import { FolderStructure } from "../types/api";

const url = config.host + ":" + config.port + config.folders_route;

async function fetchFolders(folderPath: string): Promise<FolderStructure> {
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_path: folderPath, max_depth: 5 }),
    });

    console.log(folderPath);

    if (!response.ok) {
      throw response;
    }

    return response.json();
  } catch (error) {
    console.error("Fetch folders failed:", error);
    throw error;
  }
}

export default fetchFolders;
