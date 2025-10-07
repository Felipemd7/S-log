from pathlib import Path
path = Path('clog/CLog_main.py')
text = path.read_text()
marker = 'for TIME_INTERVAL_ in ["300s"]:\r\n'
if marker not in text:
    marker = 'for TIME_INTERVAL_ in ["300s"]:\n'
if marker not in text:
    raise SystemExit('marker still not found')
prefix, suffix = text.split(marker, 1)
Path('tmp_debug.py').write_text(f"PREFIX END: {prefix[-60:]}\nSUFFIX START: {suffix[:120]!r}\n")
