#!/usr/bin/env python3
"""Make the claude.ai-hosted copy of the game: the host supplies <html>/<head>, so keep only
the title, style and body, and size the canvas to the space the page actually gets.
Usage: python3 tools/make_artifact.py OUTPUT.html"""
import re, sys
s = open('index.html', encoding='utf-8').read()
head, body = s.split('<body>', 1)
title = re.search(r'<title>.*?</title>', head).group(0)
style = re.search(r'<style>.*?</style>', head, re.S).group(0)
style = style.replace("html, body {\n    margin: 0; height: 100%;", "html, body {\n    margin: 0; height: 100%; box-sizing: border-box;")
style = style.replace("<style>", "<style>\n  :root { color-scheme: dark; }")
body = body.replace('</body>\n</html>', '').rstrip()
body = body.replace("const aspect = window.innerWidth / window.innerHeight;",
    "const vw = document.body.clientWidth || window.innerWidth, vh = document.body.clientHeight || window.innerHeight;\n  const aspect = vw / vh;")
body = body.replace("const scale = Math.min(window.innerWidth / W, window.innerHeight / H);", "const scale = Math.min(vw / W, vh / H);")
# the claude.ai copy has no sw.js next to it, so leave out the offline-cache registration
body = re.sub(r"// Installable app:.*?\n\}\n", "", body, flags=re.S)
assert 'const vw =' in body and 'Math.min(vw / W' in body and 'serviceWorker' not in body
open(sys.argv[1], 'w', encoding='utf-8').write(title + '\n' + style + '\n' + body + '\n')
