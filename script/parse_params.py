import re
import argparse
import json
from typing import Dict, Any, List

# 运行逻辑概述：
# 1) 程序从命令行接收一个输入文件路径（MITgcm 风格的参数文件），可选的 --json-out 指定输出 JSON 文件路径。
# 2) 读取整个文件文本，split_namelists() 去掉注释并按照 &NAME ... & 的区块分割出每个 namelist（返回 name->content）。
# 3) 对每个 namelist，用 parse_namelist_content() 提取 key = value 对：
#    - 使用正则找到 key = value 的片段；
#    - split_csv() 在非引号区域按逗号拆分数组元素，保持引号内内容完整；
#    - convert_token() 将标量字符串转换为布尔、整型、浮点、字符串或保持原样；
#    - 单一元素被解析为标量，多元素被解析为列表。
# 4) parse_file() 聚合所有 namelist 的解析结果为一个嵌套 dict 并返回。
# 5) main() 打印解析结果（可选）并在提供 --json-out 时将结果写入 JSON 文件。
# 6) 错误处理：文本解析为尽量宽容的类型转换；命令行参数修正为可选输出路径以避免解析错误。

def split_namelists(text: str) -> Dict[str, str]:
    namelists = {}
    current = None
    buffer: List[str] = []
    for raw in text.splitlines():
        # remove comments starting with '#'
        line = raw.split('#', 1)[0]
        line_strip = line.strip()
        if not line_strip:
            continue
        if line_strip.startswith('&') and len(line_strip) > 1:
            # start namelist: &NAME
            current = line_strip.lstrip('&').strip()
            buffer = []
            continue
        if line_strip == '&':
            # end of current namelist
            if current:
                namelists[current] = ' '.join(buffer)
                current = None
                buffer = []
            continue
        if current:
            buffer.append(line_strip)
    return namelists

def split_csv(s: str) -> List[str]:
    parts = []
    cur = ''
    in_sq = False
    in_dq = False
    for ch in s:
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            cur += ch
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
            cur += ch
        elif ch == ',' and not in_sq and not in_dq:
            parts.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip() != '':
        parts.append(cur.strip())
    return parts

def convert_token(tok: str) -> Any:
    t = tok.strip()
    if t == '':
        return None
    low = t.lower()
    if low in ('.true.', 'true'):
        return True
    if low in ('.false.', 'false'):
        return False
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        return t[1:-1]
    # Try integer (no dot, no exponent)
    try:
        if '.' not in t and 'e' not in t.lower():
            return int(t)
    except Exception:
        pass
    # Try float (handle trailing dots like "27.")
    try:
        return float(t)
    except Exception:
        return t

def parse_namelist_content(s: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    # find key = value blocks; non-greedy until next key= or end
    pattern = re.compile(r'(\w+)\s*=\s*(.*?)(?=\s*\b\w+\s*=|$)', re.S)
    for m in pattern.finditer(s):
        key = m.group(1)
        val = m.group(2).strip()
        # remove trailing commas/spaces
        val = val.rstrip(',').strip()
        # split by commas outside quotes
        parts = split_csv(val)
        parts = [p for p in parts if p != '']
        if len(parts) == 0:
            params[key] = None
        elif len(parts) == 1:
            params[key] = convert_token(parts[0])
        else:
            params[key] = [convert_token(p) for p in parts]
    return params

def parse_file(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
    namelists = split_namelists(txt)
    parsed = {}
    for name, content in namelists.items():
        parsed[name] = parse_namelist_content(content)
    return parsed

def main():
    ap = argparse.ArgumentParser(description='Parse MITgcm-style parameter file into a nested dict.')
    ap.add_argument('input', help='Path to parameter file to parse')
    ap.add_argument('--json-out', help='JSON output file path (optional)')
    args = ap.parse_args()
    parsed = parse_file(args.input)
    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as fo:
            json.dump(parsed, fo, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(parsed, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
