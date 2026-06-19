#!/bin/bash
curl -s http://localhost:6333/collections | python3 -c "
import sys, json, urllib.request
cols = json.load(sys.stdin)['result']['collections']
for c in cols:
    info = json.loads(urllib.request.urlopen(f'http://localhost:6333/collections/{c[\"name\"]}').read())['result']
    print(f'{c[\"name\"]}: {info[\"points_count\"]} points, {info[\"segments_count\"]} segments, {info[\"config\"][\"params\"][\"vectors\"][\"size\"]}dim')
"
