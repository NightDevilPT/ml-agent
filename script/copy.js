const fs = require('fs');
const path = require('path');
const readline = require('readline');

// Files and folders to ignore so the script doesn't crash or create massive files
const IGNORE_LIST = ['node_modules', '.git', 'package-lock.json', '.DS_Store', 'copy.txt', '__pycache__', '.venv', '.env'];
const IGNORE_EXTENSIONS = ['.pyc', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.woff', '.woff2', '.ttf', '.eot', '.zip', '.tar.gz', '.exe', '.dll', '.so', '.dylib', '.db', '.sqlite'];

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const OUTPUT_FILE = 'copy.txt';

// Clear the output file if it already exists from a previous run
fs.writeFileSync(OUTPUT_FILE, '');

rl.question('Enter the path of the folder or file you want to copy: ', (inputPath) => {
  const absolutePath = path.resolve(inputPath.trim());

  if (!fs.existsSync(absolutePath)) {
    console.error(`\n❌ Error: The path does not exist -> ${absolutePath}`);
    rl.close();
    return;
  }

  console.log(`\nScanning path: ${absolutePath}...\n`);
  
  try {
    processPath(absolutePath);
    console.log(`\n✅ Successfully copied all contents to ${OUTPUT_FILE}`);
  } catch (error) {
    console.error('\n❌ An error occurred:', error.message);
  } finally {
    rl.close();
  }
});

/**
 * Recursively goes through files and directories
 */
function processPath(targetPath) {
  const baseName = path.basename(targetPath);
  const extName = path.extname(targetPath).toLowerCase();

  // Skip files/folders in the ignore list or files with binary/ignored extensions
  if (IGNORE_LIST.includes(baseName) || IGNORE_EXTENSIONS.includes(extName)) {
    return;
  }

  const stats = fs.statSync(targetPath);

  if (stats.isFile()) {
    appendFileContent(targetPath);
  } else if (stats.isDirectory()) {
    const items = fs.readdirSync(targetPath);
    for (const item of items) {
      const fullPath = path.join(targetPath, item);
      processPath(fullPath); // Recursive call for sub-folders
    }
  }
}

/**
 * Reads the file content and appends it to copy.txt in the required format
 */
function appendFileContent(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    
    // Format the output exactly as requested
    const formattedOutput = `File Path : ${filePath}\nCode :\n${content}\n\n${'='.repeat(50)}\n\n`;
    
    fs.appendFileSync(OUTPUT_FILE, formattedOutput);
    console.log(`Copied: ${filePath}`);
  } catch (error) {
    // Fails safely if it tries to read a weird binary file or lacks permissions
    console.log(`⚠️ Skipped (could not read): ${filePath}`);
  }
}