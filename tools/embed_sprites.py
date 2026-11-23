#!/usr/bin/env python3
"""Embed every PNG/WebP in sprites/ into index.html as a data URI (keeps the game a single file).

Usage (from the repo root):  python3 tools/embed_sprites.py
"""
import base64, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
html_path = os.path.join(ROOT, 'index.html')
sprite_dir = os.path.join(ROOT, 'sprites')

lines = []
for name in sorted(os.listdir(sprite_dir)):
    ext = name.rsplit('.', 1)[-1].lower()
    if ext not in ('png', 'webp'):
        continue
    with open(os.path.join(sprite_dir, name), 'rb') as f:
        data = base64.b64encode(f.read()).decode('ascii')
    lines.append("  %s: 'data:image/%s;base64,%s'," % (name.rsplit('.', 1)[0], ext, data))

html = open(html_path, encoding='utf-8').read()
block = '/*EMBED_START*/\n' + '\n'.join(lines) + '\n/*EMBED_END*/'
html, n = re.subn(r'/\*EMBED_START\*/.*?/\*EMBED_END\*/', lambda m: block, html, flags=re.S)
assert n == 1, 'embed markers not found'
open(html_path, 'w', encoding='utf-8').write(html)
print('embedded %d sprites' % len(lines))
