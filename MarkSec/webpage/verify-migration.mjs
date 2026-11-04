const origin = process.argv[2];
const paths = ['/marksec', '/marksec/demo', '/marksec/toolkit'];
const resources = new Set();
for (const path of paths) {
  const response = await fetch(origin + path);
  if (response.status !== 200) throw new Error(`${path}: ${response.status}`);
  const html = (await response.text()).replace(/<link\b[^>]*rel="preconnect"[^>]*>/g, '');
  for (const match of html.matchAll(/(?:src|href)="([^"#]+)"/g)) {
    const url = match[1].replaceAll('&amp;', '&');
    if (url.startsWith('/')) {
      if (!url.startsWith('/marksec')) throw new Error(`Unprefixed URL: ${url}`);
      resources.add(url);
    }
  }
  console.log(`200 ${path}`);
}
for (const path of resources) {
  const response = await fetch(origin + path);
  if (response.status !== 200) throw new Error(`${path}: ${response.status}`);
}
console.log(`Verified ${resources.size} page links, scripts, styles, icons and images.`);
