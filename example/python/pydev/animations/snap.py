"""Screenshot an animated SVG at given times with headless Chrome (pauseAnimations + setCurrentTime).

    python snap.py out/qs_strebelle_50.svg out/qs_check 13.4 40

writes out/qs_check_t13.4.png, ... (to check an SVG without watching it; Chrome path for macOS).
"""
import os, subprocess, sys
svg_path, out_prefix, *times = sys.argv[1:]
out_prefix = os.path.abspath(out_prefix)
svg = open(svg_path).read()
chrome = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
for t in times:
    html = f"""<!doctype html><html><body style="margin:0">{svg}<script>
const s=document.querySelector('svg'); s.pauseAnimations(); s.setCurrentTime({float(t)});
</script></body></html>"""
    hp = f"{out_prefix}_t{t}.html"
    open(hp, "w").write(html)
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1920,1080",
                    f"--screenshot={out_prefix}_t{t}.png", "--virtual-time-budget=3000", "file://" + hp],
                   capture_output=True, timeout=120)
    os.remove(hp)
    print(out_prefix + f"_t{t}.png", os.path.exists(f"{out_prefix}_t{t}.png"))
