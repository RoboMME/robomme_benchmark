import re
COLORS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black', 'white', 'pink', 'brown', 'gray', 'grey', 'cyan']

def classify_rgb(c):
    r, g, b = [float(x) for x in c]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx < 80: return 'black'
    if mn > 140 and mx - mn < 40: return 'white'
    if r > 120 and g < 70 and b < 70 and r > 2 * g: return 'red'
    if g > r + 30 and g > b + 40: return 'green'
    if b > r + 40 and b > g + 30: return 'blue'
    if r > 120 and b > 100 and g < 0.7 * min(r, b): return 'purple'
    if r > 140 and g > 140 and b < 90: return 'yellow'
    if r > 140 and 70 <= g < 160 and b < 70: return 'orange'
    return 'other'

def parse_instruction(text):
    """Return ordered list of (color, all_flag)."""
    t = text.lower()
    t = t.split('then', 1)[1] if 'then' in t else t
    out = []
    pat = re.compile(r'(all\s+(?:the\s+)?containers?|the\s+(?:container|one)s?|containers?)\s+hiding\s+(?:the\s+|a\s+|an\s+)?(\w+)\s+cubes?')
    for m in pat.finditer(t):
        col = m.group(2)
        if col == 'grey': col = 'gray'
        out.append((col, m.group(1).startswith('all')))
    if not out:
        for w in re.findall(r'\b(' + '|'.join(COLORS) + r')\b', t):
            out.append((w, False))
    return out
