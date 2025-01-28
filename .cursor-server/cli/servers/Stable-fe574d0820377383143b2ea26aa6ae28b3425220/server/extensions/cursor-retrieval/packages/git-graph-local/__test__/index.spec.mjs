import test from 'ava'

import { LocalGitGraph } from '../index.js'

test('basic construction', async (t) => {
  console.time("create");
  let graph = new LocalGitGraph("../../../../../");
  console.timeEnd("create");

  console.time("openFile");
  let open_file = await graph.openFile("vscode/src/vs/editor/browser/coreCommands.ts");
  console.timeEnd("openFile");

  console.time("findSimilarFiles");
  let similar = await open_file.findSimilarFiles(43);
  console.timeEnd("findSimilarFiles");
  t.assert(similar.length > 0);
  console.log(similar);

  console.time("findSimilarFiles");
  similar = await open_file.findSimilarFiles(43);
  console.timeEnd("findSimilarFiles");
})
