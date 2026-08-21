import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { buildPortableArtifact } from "file:///C:/Users/33696/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/build_portable_artifact.mjs";
import { extractPortableChartSvgs } from "file:///C:/Users/33696/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/extract_portable_chart_svgs.mjs";

const artifact = JSON.parse(readFileSync(new URL("./artifact.json", import.meta.url), "utf8"));
const output = new URL("./W007_精简结论包_最终审核草案.html", import.meta.url);
writeFileSync(output, buildPortableArtifact(artifact), "utf8");
const staticCharts = await extractPortableChartSvgs({ htmlPath: fileURLToPath(output) });
writeFileSync(output, buildPortableArtifact(artifact, { staticCharts }), "utf8");
console.log(fileURLToPath(output));
