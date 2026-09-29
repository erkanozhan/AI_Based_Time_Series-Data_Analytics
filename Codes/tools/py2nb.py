"""Percent formatındaki (# %% / # %% [markdown]) bir .py dosyasını Jupyter notebook'a çevirir.

Kullanım (depo kök dizininden): python Codes/tools/py2nb.py Codes/python/ch07_sarima_airpassengers.py
Kural:
  - '# %% [markdown]' ile başlayan hücrenin satırları '# ' önekinden arındırılıp markdown hücresi olur.
  - '# %%' ile başlayan hücre kod hücresidir.
  - '# COLAB: <komut>' satırları yalnızca notebook'ta '%<komut>' olarak kod hücresine yazılır
    (ör. '# COLAB: pip install -q pmdarima'), .py olarak çalıştırmada yorum satırıdır.
  - İlk '# %%' işaretinden önceki kısım (modül docstring vb.) ilk kod hücresine eklenir.
"""
import json, os, re, sys


def convert(src, dst=None):
    if dst is None:
        base = os.path.splitext(os.path.basename(src))[0]
        root = os.path.dirname(os.path.dirname(os.path.abspath(src)))  # Codes/
        dst = os.path.join(root, "notebooks", base + ".ipynb")
    lines = open(src, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
    cells, cur, kind = [], [], "code"

    def flush():
        nonlocal cur
        while cur and not cur[-1].strip():
            cur.pop()
        while cur and not cur[0].strip():
            cur.pop(0)
        if cur:
            if kind == "markdown":
                text = [re.sub(r"^# ?", "", l) for l in cur]
                cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(text)})
            else:
                code = [("%" + l.split("COLAB:", 1)[1].strip()) if l.lstrip().startswith("# COLAB:") else l for l in cur]
                cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                              "outputs": [], "source": "\n".join(code)})
        cur = []

    for l in lines:
        m = re.match(r"^# %%(.*)$", l)
        if m:
            flush()
            kind = "markdown" if "[markdown]" in m.group(1) else "code"
            continue
        cur.append(l)
    flush()
    for c in cells:  # nbformat: kaynak satır listesi
        src_lines = c["source"].split("\n")
        c["source"] = [s + "\n" for s in src_lines[:-1]] + [src_lines[-1]]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for k, c in enumerate(cells):
        c["id"] = f"c{k:03d}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    return dst, len(cells)


if __name__ == "__main__":
    print(convert(*sys.argv[1:3]))
