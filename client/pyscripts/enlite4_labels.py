#!/usr/bin/env python3

import sys
import json
import urllib.request
from pathlib import Path

outdir = Path('.' if len(sys.argv) < 2 else sys.argv[1])
outdir.mkdir(parents=True, exist_ok=True)

link = ('https://raw.githubusercontent.com/onnx/models/main/vision'
        '/classification/efficientnet-lite4/dependencies/labels_map.txt')

with urllib.request.urlopen(link) as fp:
    label_map = json.load(fp)

with open(outdir / 'enlite4_labels.txt', 'w') as fp:
    fp.write('\n'.join(label_map.values()))
