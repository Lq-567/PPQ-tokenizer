"""为完整运行生成或检查 SHA256 清单；不改变训练或评估。"""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--verify', action='store_true')
a = p.parse_args()
root = a.directory.resolve()
manifest = root / 'OUTPUT_SHA256.json'
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
if a.verify:
    files = json.loads(manifest.read_text())
    missing = [name for name, value in files.items() if not (root / name).is_file() or digest(root / name) != value]
    if missing: raise SystemExit('校验失败：' + ', '.join(missing))
    print(f'校验通过：{len(files)} 个文件')
else:
    assert (root / 'COMPLETE.json').is_file()
    files = {str(f.relative_to(root)): digest(f) for f in sorted(root.rglob('*')) if f.is_file() and f != manifest}
    manifest.write_text(json.dumps(files, indent=2) + '\n')
    print(f'已登记 {len(files)} 个文件')
