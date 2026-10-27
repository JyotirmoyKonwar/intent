const fs = require("fs");

const file = process.argv[2];
const text = fs.readFileSync(file, "utf8");
const toc = text
  .split("\n")
  .filter((l) => /^#{2,3} /.test(l))
  .map((l) => {
    const depth = l.indexOf(" ") - 2;
    const title = l.replace(/^#+ /, "");
    const anchor = title.toLowerCase().replace(/[^a-z0-9]+/g, "-");
    return `${"  ".repeat(depth)}- [${title}](#${anchor})`;
  })
  .join("\n");
const out = text.replace(
  /<!-- toc -->[\s\S]*<!-- tocstop -->/,
  `<!-- toc -->\n${toc}\n<!-- tocstop -->`
);
fs.writeFileSync(file, out);
