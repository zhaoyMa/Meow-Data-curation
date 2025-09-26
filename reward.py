import json
import ast
import io
import re
import tokenize
from ref_reward import article_reward

def _convert_string_tokens_to_json_style(s: str) -> str:
    """
    将输入 s 中的所有字符串字面量（无论单引号还是双引号）转换为 JSON 风格（双引号且正确转义）。
    其它 token 原样保留。返回转换后的字符串（应该是有效 JSON）。
    """
    out_parts = []
    # tokenize.tokenize 需要 bytes 的 readline
    b = io.BytesIO(s.encode('utf-8'))
    try:
        tokgen = tokenize.tokenize(b.readline)
    except Exception:
        # 在极端异常下直接返回原始字符串
        return s

    for tok in tokgen:
        toknum = tok.type
        tokval = tok.string

        # 跳过编码声明 token（在开始处），以及末尾的 ENDMARKER
        if toknum == tokenize.ENCODING or toknum == tokenize.ENDMARKER:
            continue

        if toknum == tokenize.STRING:
            # tokval 是比如 "'a\\\"b\\n'" 或 "\"a\\\"b\\n\"" 等字面量
            try:
                # 安全地把字面量评估为 Python 字符串值
                py_str = ast.literal_eval(tokval)
            except Exception:
                # 若失败，保留原 token（降级处理）
                out_parts.append(tokval)
                continue
            # 然后用 json.dumps 生成 JSON 风格的双引号字符串（会正确转义）
            json_str = json.dumps(py_str, ensure_ascii=False)
            out_parts.append(json_str)
        else:
            # 对于其它 token，直接使用原始文本（保持格式，如逗号、方括号、大括号、名字、数值等）
            out_parts.append(tokval)
    return "".join(out_parts)


def parse_list_string(s: str):
    """
    尝试把形如 "[{...}, {...}]" 的字符串解析为 Python 列表（或其它容器）。
    支持：
      - 严格 JSON 字符串（使用双引号） -> json.loads
      - Python 字面量（可用单引号、转义、\\n 等） -> ast.literal_eval
      - 混合或不完全 JSON 的情况下，tokenize -> 转换再 json.loads
    失败时抛出 ValueError。
    """
    if not isinstance(s, str):
        raise TypeError("输入必须是字符串")

    s_strip = s.strip()

    # 1) 先尝试 JSON（最快）
    try:
        return json.loads(s_strip)
    except Exception:
        pass

    # 2) 再尝试 ast.literal_eval（支持单引号、Python 字符串转义等）
    try:
        return ast.literal_eval(s_strip)
    except Exception:
        pass

    # 3) 最后用 tokenize 把所有字符串字面量转换为 JSON 风格，然后 json.loads
    converted = _convert_string_tokens_to_json_style(s_strip)
    try:
        return json.loads(converted)
    except Exception as e:
        # 如果仍失败，给出包含原始错误信息的异常，便于 debug
        raise ValueError(f"无法解析输入字符串为列表：{e}\n原始/转换后字符串：\n{converted}") from e

# 这个函数对输入字符串打分
def reward(item):
    try:
        pred_text = item.get("predict", None)
        pred = parse_list_string(delete_think(pred_text))

        true_text = item.get("label", None)
        true = parse_list_string(delete_think(true_text))
        R_article, details = article_reward(gen_sections = pred , true_sections = true)
        return R_article
    except:
        return -2

def delete_think(s: str) -> str:
    # 删除从开头到第一个 </think>（包含 </think>）
    text = re.sub(r'^.*?</think>', '', s, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()

import json

def main():
    jsonl_file = "mix100.jsonl"  # 你的 JSONL 文件路径

    total = 0       # 总行数
    valid = 0       # 成功计算 reward 的行数
    s_sum = 0.0     # s 的累加和

    with open(jsonl_file, "r", encoding="utf-8") as fin:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                s = reward(item)
                
                if s != -2:
                    print(s)
                    s_sum += s
                    valid += 1

            except Exception as e:
                print(f"[{total}] 无法解析 JSON 行：{e}")
                continue

    if valid > 0:
        print(f"\n共有 {valid} 条有效数据，s 的均值为 {s_sum / valid:.4f}")
    else:
        print("没有有效的 s 值")




if __name__ == "__main__":
    main()