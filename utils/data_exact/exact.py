# %% [markdown]
# # arXiv Paper Structure Extraction
# 
# This notebook processes arXiv papers to extract:
# 1. Paper metadata from `filtered_survey_papers.jsonl`
# 2. Paper outlines (section structure) from TeX files
# 3. Reference metadata from BibTeX files
# 
# The final output is a structured JSON for each paper containing all three components.

# 运行这个脚本需要额外的 src 文件，该文件包含 metadata_path 中所有文章的压缩包

# %%
# Import required libraries
import os
import re
import json
import glob
import shutil
import pathlib
from collections import defaultdict
import pandas as pd
from typing import Tuple, Dict, List
# %% [markdown]
# ## 2. Load Paper Metadata
# 
# Load paper metadata from filtered_survey_papers.jsonl and create a dictionary mapping paper IDs to their metadata.

# %%
# Load metadata from filtered_survey_papers.jsonl
metadata_path = 'filtered_bpm_2020.jsonl'
paper_metadata = {}

with open(metadata_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        paper_id = data.get('id')
        if paper_id:
            paper_metadata[paper_id] = data

# Extract paper IDs directly from metadata
paper_ids = list(paper_metadata.keys())
print(f"Loaded metadata for {len(paper_metadata)} papers")
print(f"Sample IDs: {paper_ids[:5]}")

# Show a sample of the metadata
if paper_metadata:
    sample_id = next(iter(paper_metadata))
    print(f"Sample metadata for {sample_id}:")
    print(json.dumps(paper_metadata[sample_id], indent=2))

import re
#章节结构
def extract_sections(tex_content):
    """Extract hierarchical section structure from TeX content with improved robustness"""
    # 更健壮的正则表达式，支持可选短标题和更复杂的标题内容
    section_patterns = {
        'chapter': r'\\chapter(?:\*|\s*\[[^\]]*\])?\s*\{((?:[^{}]|(?:\{[^{}]*\}))*)\}',
        'section': r'\\section(?:\*|\s*\[[^\]]*\])?\s*\{((?:[^{}]|(?:\{[^{}]*\}))*)\}',
        'subsection': r'\\subsection(?:\*|\s*\[[^\]]*\])?\s*\{((?:[^{}]|(?:\{[^{}]*\}))*)\}',
        'subsubsection': r'\\subsubsection(?:\*|\s*\[[^\]]*\])?\s*\{((?:[^{}]|(?:\{[^{}]*\}))*)\}'
    }
    
    # 查找可能的自定义节命令 (如 \mynewsection{...})
    custom_section_pattern = r'\\((?:new|renew)command\s*\{\\([a-zA-Z]+section[a-zA-Z]*)\})'
    custom_sections = re.findall(custom_section_pattern, tex_content)
    for _, cmd in custom_sections:
        section_patterns[cmd] = fr'\\{cmd}(?:\*|\s*\[[^\]]*\])?\s*\{{((?:[^{{}}]|(?:\{{[^{{}}]*\}}))*)\}}'
    
    # 查找 documentclass 以确定文档类型
    doc_class = re.search(r'\\documentclass(?:\[[^\]]*\])?\{([^}]*)\}', tex_content)
    if doc_class:
        doc_type = doc_class.group(1)
        # 添加特定文档类的支持（如 book、report、article 等）
        if doc_type in ['book', 'report']:
            pass  # 已经包含了 chapter
        elif doc_type == 'article':
            # article 类文档没有 chapter，从 section 开始
            level_hierarchy = ['section', 'subsection', 'subsubsection', 'paragraph']
        else:
            # 默认层级
            level_hierarchy = ['chapter', 'section', 'subsection', 'subsubsection']
    else:
        # 默认层级
        level_hierarchy = ['chapter', 'section', 'subsection', 'subsubsection']
    
    # 查找文档开始位置，跳过前导代码
    begin_doc = re.search(r'\\begin\{document\}', tex_content)
    start_idx = begin_doc.end() if begin_doc else 0
    
    # 查找文档结束位置
    end_doc = re.search(r'\\end\{document\}', tex_content)
    end_idx = end_doc.start() if end_doc else len(tex_content)
    
    # 只处理文档主体部分
    content_to_process = tex_content[start_idx:end_idx]
    
    # 从文本中移除注释
    content_to_process = re.sub(r'%.*$', '', content_to_process, flags=re.MULTILINE)
    
    # 查找所有节标题，带行号
    sections = []
    lines = content_to_process.split('\n')
    
    # 检测复杂标题（可能跨多行）
    accumulated_line = ''
    accumulating = False
    brace_count = 0
    line_start = 0
    
    for i, line in enumerate(lines):
        if accumulating:
            accumulated_line += line
            
            # 计算括号平衡
            for char in line:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    
            # 如果括号已平衡，可以处理积累的行
            if brace_count == 0:
                accumulating = False
                line_to_process = accumulated_line
                process_line_for_sections(line_to_process, section_patterns, sections, line_start)
        else:
            # 检查此行是否开始一个新节
            for level, pattern in section_patterns.items():
                if re.search(fr'\\{level.split("_")[0]}', line):  # 简单检查，避免昂贵的正则
                    #print(f"\n发现可能的章节标记 ({level}): {line}")
                    # 计算此行的花括号平衡
                    brace_count = 0
                    for char in line:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                    
                    if brace_count > 0:  # 未闭合的括号，需要累积
                        accumulating = True
                        accumulated_line = line
                        line_start = i
                        break
                    else:  # 括号平衡，直接处理
                        process_line_for_sections(line, section_patterns, sections, i)
                        break
    
    # 按行号排序
    sections.sort(key=lambda x: x['line_num'])
    
    # 构建分层结构
    structured_outline = []
    section_stack = [{'children': structured_outline, 'level': 'root'}]
    
    for section in sections:
        #print(f"处理章节: {section} ")
        current_level = level_hierarchy.index(section['level']) if section['level'] in level_hierarchy else 999
        
        # 从堆栈中弹出，直到找到合适级别的父节点
        while len(section_stack) > 1:
            parent_level = level_hierarchy.index(section_stack[-1]['level']) if section_stack[-1]['level'] in level_hierarchy else -1
            if parent_level < current_level:
                break
            section_stack.pop()
        
        # 创建节点
        section_node = {
            'title': section['title'].strip(),
            'level': section['level'],
            'line_num': section['line_num'],
            'ref': [],
            'children': []
        }
        
        # 添加到父节点的子节点
        section_stack[-1]['children'].append(section_node)
        
        # 将此节点推入堆栈
        section_stack.append(section_node)
    
    return structured_outline
#单行标题
def process_line_for_sections(line, section_patterns, sections, line_num):
    """Process a line to find section headers"""
    for level, pattern in section_patterns.items():
        matches = re.finditer(pattern, line)
        for match in matches:
            #print(f"找到章节: {match} ")
            title = match.group(1)
            #print(f"章节标题: {title} ")
            # 清理标题中的 LaTeX 命令
            title = re.sub(r'\\[a-zA-Z]+{', '', title)  # 移除命令名
            title = re.sub(r'}', '', title)  # 移除花括号
            #print(f"清理后的标题: {title} ")
            sections.append({
                'level': level,
                'title': title,
                'line_num': line_num
            })

def extract_citations(tex_content, section_outline):
    """Extract citations from each section in the outline with improved accuracy"""
    if not section_outline:
        return section_outline
    
    # 从文本中移除注释
    tex_content = re.sub(r'%.*$', '', tex_content, flags=re.MULTILINE)
    
    level_hierarchy = ['chapter', 'section', 'subsection', 'subsubsection']
    
    # 更健壮的引用模式，支持更多类型的引用命令
    citation_patterns = [
        r'\\cite[a-zA-Z]*\{([^}]*)\}',                          # \cite{key}
        r'\\text(?:cite|cquote)[a-zA-Z]*\{([^}]*)\}\{',         # \textcite{key}{
        r'\\parencite[a-zA-Z]*\{([^}]*)\}',                     # \parencite{key}
        r'\\footcite[a-zA-Z]*\{([^}]*)\}',                      # \footcite{key}
        r'\\autocite[a-zA-Z]*\{([^}]*)\}'                       # \autocite{key}
    ]
    
    lines = tex_content.split('\n')
    
    # 标识所有节边界
    section_markers = []
    
    def collect_section_markers(sections, path=[]):
        # Fixed bug: Changed enumeration to use enumerate() function to get both index and section
        for i, section in enumerate(sections):  # This was the bug - needed enumerate() here
            current_path = path + [i]
            section_markers.append({
                'line': section['line_num'],
                'path': current_path.copy(),
                'section': section
            })
            if 'children' in section and section['children']:
                collect_section_markers(section['children'], current_path + ['children'])
    
    collect_section_markers(section_outline)
    
    # 按行号排序
    section_markers.sort(key=lambda x: x['line'])
    
    # 获取每节的引用
    for i, marker in enumerate(section_markers):
        start_line = marker['line'] + 1  # 从节标题的下一行开始
        
        # 确定当前节的级别
        current_section = marker['section']
        current_level_idx = level_hierarchy.index(current_section['level']) if current_section['level'] in level_hierarchy else 999
        
        # 找到下一个相同或更高级别的节作为边界
        end_line = len(lines)
        for j in range(i + 1, len(section_markers)):
            next_marker = section_markers[j]
            next_section = next_marker['section']
            next_level_idx = level_hierarchy.index(next_section['level']) if next_section['level'] in level_hierarchy else -1
            
            if next_level_idx <= current_level_idx:
                end_line = next_marker['line']
                break
        
        # 获取此节的文本（不包括子节）
        section_text = '\n'.join(lines[start_line:end_line])
        
        # 查找此节中的所有引用
        citations = []
        for pattern in citation_patterns:
            for match in re.finditer(pattern, section_text):
                cite_group = match.group(1)
                # 处理多个引用（用逗号分隔）
                for cite in cite_group.split(','):
                    cite = cite.strip()
                    if cite and cite not in citations:
                        citations.append(cite)
        
        # 更新节的引用
        current_section['ref'] = citations
    return section_outline


def process_tex_document(tex_content):
    """Process a complete TeX document and extract its structure with citations"""
    try:
        # 1. 提取节结构
        outline = extract_sections(tex_content)
        # 2. 如果没有找到任何节，尝试使用其他方法
        if not outline:
            print("Warning: No sections found using standard methods. Trying alternative approaches...")
            # 这里可以添加替代方法
        
        # 3. 提取每节的引用
        outline_with_citations = extract_citations(tex_content, outline)
        
        # 4. 简化验证：仅检查是否有结论章节
        has_conclusion = validate_outline_completeness(outline, tex_content)
        
        if not has_conclusion:
            print("WARNING: Document structure may be incomplete (no conclusion found)!")

        idx = next((i for i, item in enumerate(outline_with_citations)
            if item.get('level') == 'section'
            and item.get('title', '').strip().lower() == 'introduction'),
               None)
        if idx is None:
            return outline_with_citations
        return outline_with_citations[idx:]

    except Exception as e:
        print(f"Error processing TeX document: {e}")
        return []

def count_sections(outline):
    count = len(outline)
    for section in outline:
        if 'children' in section and section['children']:
            count += count_sections(section['children'])
    return count

def print_outline(outline, indent=0):
    """Print the document outline in a readable hierarchical format"""
    for section in outline:
        print("  " * indent + f"- {section['level']}: {section['title']}")
        if 'children' in section and section['children']:
            print_outline(section['children'], indent + 1)

def validate_outline_completeness(outline, tex_content):
    """
    验证提取的大纲是否完整
    检查是否存在 conclusion/summary/conclusions/acknowledgment 等章节
    返回 True 表示大纲可能完整，False 表示可能不完整
    """
    # 检查是否存在 conclusion/summary/acknowledgment 等章节
    has_ending_section = False
    
    def check_ending_section(sections):
        nonlocal has_ending_section
        for section in sections:
            if section['title']:
                title_lower = section['title'].lower()
                if any(keyword in title_lower for keyword in ['conclusion', 'summary', 'acknowledgment', 'acknowledgement', 'future work', 'discussion']):
                    has_ending_section = True
                    return True
            if 'children' in section and section['children']:
                if check_ending_section(section['children']):
                    return True
        return False
    
    check_ending_section(outline)
    
    # 输出验证信息
    if has_ending_section:
        print(f"Validation result: Found conclusion/summary/acknowledgment section")
    else:
        print(f"Validation result: Did not find conclusion/summary/acknowledgment section")
    
    return has_ending_section

def find_matching_brace(s: str, pos: int) -> int:
    """假设 s[pos] == '{'，返回匹配的 '}' 的索引；找不到返回 -1。"""
    assert 0 <= pos < len(s) and s[pos] == '{'
    depth = 0
    i = pos
    n = len(s)
    while i < n:
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1

def read_braced(s: str, pos: int) -> Tuple[str, int]:
    """读以 { 开头的值，返回 (内部字符串, 紧跟闭合'}'后的索引)。"""
    end = find_matching_brace(s, pos)
    if end == -1:
        return s[pos+1:], len(s)
    return s[pos+1:end], end+1

def read_quoted(s: str, pos: int) -> Tuple[str, int]:
    """读以 \" 开头的值，支持转义，返回 (内部字符串, 紧跟结束引号后的索引)。"""
    assert 0 <= pos < len(s) and s[pos] == '"'
    i = pos + 1
    n = len(s)
    out = []
    while i < n:
        ch = s[i]
        if ch == '\\' and i + 1 < n:
            out.append(s[i+1])
            i += 2
            continue
        if ch == '"':
            return ''.join(out), i+1
        out.append(ch)
        i += 1
    return ''.join(out), n

def parse_entry_fields(entry_content: str) -> Dict[str, str]:
    """
    逐字段解析 entry_content（不包含最外层大括号，也应已去掉开头的 citation key）。
    支持 {..}, "..." 和裸值（裸值读到同级逗号或条目末尾）。
    返回字段字典（作者会被拆成 authors 列表，pages 会标准化）。
    """
    i = 0
    n = len(entry_content)
    fields: Dict[str, str] = {}
    while i < n:
        # 跳过空白和逗号
        while i < n and entry_content[i].isspace():
            i += 1
        if i >= n:
            break
        # 匹配字段名 like author = ...
        m = re.match(r'\s*([A-Za-z0-9_-]+)\s*=', entry_content[i:])
        if not m:
            break
        field_name = m.group(1).lower()
        i += m.end()
        while i < n and entry_content[i].isspace():
            i += 1
        if i >= n:
            break
        ch = entry_content[i]
        if ch == '{':
            value, new_i = read_braced(entry_content, i)
        elif ch == '"':
            value, new_i = read_quoted(entry_content, i)
        else:
            # 裸值：读到同级逗号或末尾（跳过大括号/引号内部的逗号）
            start = i
            depth = 0
            in_quote = False
            while i < n:
                c = entry_content[i]
                if c == '{':
                    depth += 1
                elif c == '}':
                    if depth > 0:
                        depth -= 1
                elif c == '"':
                    in_quote = not in_quote
                elif c == ',' and depth == 0 and not in_quote:
                    break
                i += 1
            value = entry_content[start:i].strip()
            new_i = i
        # 跳过可能的空白和逗号
        while new_i < n and entry_content[new_i].isspace():
            new_i += 1
        if new_i < n and entry_content[new_i] == ',':
            new_i += 1
        val = value.strip()
        # 特殊处理常见字段
        if field_name == 'author':
            authors = [re.sub(r'[{}]', '', a).strip() for a in re.split(r'\s+and\s+', val)]
            fields['authors'] = authors
        elif field_name == 'pages':
            fields['pages'] = val.replace('--', '-')
        else:
            # 去掉可能残留的外层大括号（如果有）
            fields[field_name] = re.sub(r'^\s*{\s*|\s*}\s*$', '', val)
        i = new_i
    return fields

def parse_bibtex(bibtex_content: str) :
    """
    更鲁棒的 BibTeX 解析器，返回引用列表（每项为 dict）。
    处理条目格式：@type{key, ...fields...}
    支持字段顺序任意，支持嵌套大括号与引号，能处理裸值（尽量）。
    """
    references = []
    # 找到每个 @type{key, 开头
    for m in re.finditer(r'@(\w+)\s*{\s*([^,]+),', bibtex_content, re.IGNORECASE):
        entry_type = m.group(1)
        citation_key = m.group(2).strip()
        brace_pos = bibtex_content.find('{', m.start())
        if brace_pos == -1:
            continue
        end_pos = find_matching_brace(bibtex_content, brace_pos)
        if end_pos == -1:
            entry_content = bibtex_content[brace_pos+1:]
        else:
            entry_content = bibtex_content[brace_pos+1:end_pos]
        # 删掉开头的 citation key（entry_content 可能以 "key," 开头）
        first_comma = entry_content.find(',')
        if first_comma != -1:
            entry_content = entry_content[first_comma+1:].lstrip()
        # 跳过非文献类型
        if entry_type.lower() in ['comment', 'preamble', 'string']:
            continue
        ref = {'key': citation_key, 'type': entry_type.lower()}
        fields = parse_entry_fields(entry_content)
        ref.update(fields)
        references.append(ref)
    return references

def format_outline(sections, number_prefix=''):
    """Format sections with proper numbering"""
    formatted = []
    
    for i, section in enumerate(sections):
        # Create a copy of the section
        formatted_section = section.copy()
        
        # Add numbering to title
        section_number = f"{number_prefix}{i+1}" if number_prefix else f"{i+1}"
        
        # Use proper hierarchical numbering format
        if number_prefix and not number_prefix.endswith('.'):
            section_number = f"{number_prefix}.{i+1}"
        
        if 'level' in formatted_section:
            del formatted_section['level']
        if 'line_num' in formatted_section:
            del formatted_section['line_num']
            
        formatted_section['title'] = f"{section_number} {section['title']}"
        
        # Format children recursively with proper hierarchical numbering
        if 'children' in section and section['children']:
            formatted_section['children'] = format_outline(section['children'], f"{section_number}")
        
        formatted.append(formatted_section)
    
    return formatted

def format_outline_flat(sections, number_prefix='', level=1):
    """Flatten a nested section structure into a flat outline with numbering and levels"""
    outline = []

    for i, section in enumerate(sections):
        section_number = f"{number_prefix}{i+1}" if not number_prefix else f"{number_prefix}.{i+1}"

        # 基础字段
        outline_entry = {
            "level": level,
            "numbering": section_number,
            "title": section["title"]
        }

        # 如果有ref字段，也保留它
        if "ref" in section:
            outline_entry["ref"] = section["ref"]

        outline.append(outline_entry)

        # 如果有children，递归处理并扩展到当前outline中
        if "children" in section and section["children"]:
            child_outline = format_outline_flat(section["children"], section_number, level + 1)
            outline.extend(child_outline)

    return outline

def find_main_tex_file(tex_files):
    """Find the main tex file by looking for document structure indicators"""
    # Sort candidates by likelihood of being the main file
    main_candidates = []
    
    for tex_file in tex_files:
        # Skip likely auxiliary files
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Calculate a score based on main file indicators
            score = 0
            
            # Strong indicators
            if re.search(r'\\begin\s*\{\s*document\s*\}', content):
                score += 10
            if re.search(r'\\documentclass', content):
                score += 8
            if re.search(r'\\maketitle', content):
                score += 5
            if re.search(r'\\title', content):
                score += 4
            if re.search(r'\\author', content):
                score += 4
            if re.search(r'\\bibliography', content) or re.search(r'\\bibliographystyle', content):
                score += 3
            
            # Check if it includes other files (main files often include other files)
            include_count = len(re.findall(r'\\input\s*\{', content)) + len(re.findall(r'\\include\s*\{', content))
            score += min(include_count, 5)  # Cap at 5 to avoid overweighting
            
            
            # Add file as a candidate with its score
            main_candidates.append((tex_file, score))
            
        except Exception as e:
            print(f"  Error reading {tex_file}: {e}")
    
    # Sort candidates by score (descending)
    main_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Return the best candidate, or None if no candidates found
    return main_candidates[0][0] if main_candidates else None
def extract_input_files(content, base_dir):
    """Extract files that are included or input in the tex file"""
    input_patterns = [
        r'\\input\s*{\s*([^}]+)\s*}',
        r'\\include\s*{\s*([^}]+)\s*}'
    ]
    
    input_files = []
    
    for pattern in input_patterns:
        for match in re.finditer(pattern, content):
            filename = match.group(1).strip()
            
            # Handle relative paths and add .tex extension if missing
            if not filename.endswith('.tex'):
                filename += '.tex'
                
            # Create absolute path
            filepath = os.path.normpath(os.path.join(base_dir, filename))
            
            # Add to list of input files
            input_files.append(filepath)
    
    return input_files
def merge_tex_content(main_tex_file, visited=None):
    """Recursively merge content from included files into a single document"""
    if visited is None:
        visited = set()
        
    # Avoid circular includes
    if main_tex_file in visited:
        return ""
        
    visited.add(main_tex_file)
    
    try:
        # Read main file content
        with open(main_tex_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Extract base directory for resolving relative paths
        base_dir = os.path.dirname(main_tex_file)
        
        # Find all input/include statements
        input_patterns = [
            (r'\\input\s*{\s*([^}]+)\s*}', '\\input{'),
            (r'\\include\s*{\s*([^}]+)\s*}', '\\include{')
        ]
        
        # Process each pattern
        for pattern, command_prefix in input_patterns:
            # Find all matches in the current content
            offset = 0
            for match in re.finditer(pattern, content):
                filename = match.group(1).strip()
                
                # Handle relative paths and add .tex extension if missing
                if not filename.endswith('.tex'):
                    filename += '.tex'
                    
                # Create absolute path
                filepath = os.path.normpath(os.path.join(base_dir, filename))
                
                # Check if file exists
                if os.path.exists(filepath):
                    # Recursively process the included file
                    included_content = merge_tex_content(filepath, visited)
                    
                    # Replace the input/include command with the actual content
                    start_pos = match.start() + offset
                    end_pos = match.end() + offset
                    
                    # Add a comment to mark where content is inserted from
                    replacement = f"% START OF INCLUDED FILE: {filename}\n{included_content}\n% END OF INCLUDED FILE: {filename}"
                    
                    # Replace the command with the content
                    content = content[:start_pos] + replacement + content[end_pos:]
                    
                    # Update offset for subsequent replacements
                    offset += len(replacement) - (end_pos - start_pos)
                else:
                    print(f"  Warning: Included file not found: {filepath}")
        
        return content
    except Exception as e:
        print(f"  Error processing {main_tex_file}: {e}")
        return ""
def preprocess_latex(text):
    # 移除 \href 命令
    pattern = r'\\href\s*\{[^}]*\}'
    return re.sub(pattern, '', text)
def process_title(s: str) -> str:
    """
    检查并处理字符串 s 中的花括号内容：
      1. 如果没有任何成对闭合的 {}，返回原字符串；
      2. 如果只有一对（最外层）{}，删除这对括号及其内部所有字符，返回剩余部分；
      3. 如果有两对或以上（最外层），返回空字符串 ""。
    嵌套时只算最外层那一对。
    """
    depth = 0
    pairs = []  
    start_idx = None
    for i, ch in enumerate(s):
        if ch == '{':
            depth += 1
            if depth == 1:
                start_idx = i
        elif ch == '}':
            if depth == 1:
                pairs.append((start_idx, i))
                start_idx = None
            if depth > 0:
                depth -= 1
    if len(pairs) == 0:
        return s
    elif len(pairs) == 1:
        start, end = pairs[0]
        return s[:start] + s[end+1:]
    else:
        # 两对或以上，清空
        return ""
def parse_bbl(block_text: str):
    author = ""
    m_year = re.search(r'\d{4}', block_text)
    year = m_year.group(0) if m_year else ""
    title = ""
    pattern_block = r'\\(?:newblock|bibfield)'
    parts = re.split(pattern_block, block_text)
    pattern_author = re.compile(r'\n(?! )([A-Za-z\.,~ ]+?)\n')
    new_parts = []
    for s in parts:
        tmp = s.replace("\n ", "")
        au = pattern_author.search(tmp)
        if au:
            author = au.group(1)
            tmp = re.sub(r'\n[^\n]+\n', '', tmp)
        new_parts.append(tmp)
        #print(f" line:{tmp}")

    # 提取key    
    text = new_parts[0] # 标题
    # 方法1：^\\bibitem{…} 开头
    m = re.match(r'^\\bibitem\{([^}]*)\}', block_text)
    trimmed = text.rstrip(' \n')
    m2 = re.search(r'\{([^}]*)\}$', trimmed)
    if m:
        key = m.group(1)
        #print(m.group(1))
    # 方法2：尾部去除空格或换行
    elif m2:
        key = m2.group(1)
        #print(m2.group(1)) 
    else:
        key = ""
    if author:
        if len(new_parts)>1:
            title = new_parts[1]
    else:
        ma = re.search(r'\n(.*?),', text)
        if ma:
            author = ma.group(1)
            title_match = re.search(r'``(.*?)\'\'', text)
            if title_match:
                title = title_match.group(1).strip()  # 获取标题内容
                title = title.replace('\n', ' ')
                title = re.sub(r'\s+', ' ', title).strip()
            else:
                title = ""
        else:
            title_pattern = re.compile(r'\\showarticletitle\{(.*?)\}', re.DOTALL)
            title_match = title_pattern.search(block_text)
            if not title_match:
                title = ""
                author = ""
                year = ""
            else:
                title = title_match.group(1).strip()  # 获取标题内容
                title = title.replace('\n', ' ')
                title = re.sub(r'\s+', ' ', title).strip()
                
                # 提取作者信息
                author_pattern = re.compile(r'\\bibinfo\{person\}\{(.*?)\}', re.DOTALL)
                author_match = author_pattern.search(block_text)
                author = author_match.group(1).strip() if author_match else ""
                
                # 提取年份
                year_pattern = re.compile(r'\\bibinfo\{year\}\{(\d{4})\}', re.DOTALL)
                year_match = year_pattern.search(block_text)
                year = year_match.group(1) if year_match else ""
    title = process_title(title)
    title = title.replace('\n', '')
    while title and (title[-1].isdigit() or title[-1] in {',', '.', ' '}):
        title = title[:-1]
    m_year = re.search(r'\d{4}', key)
    year = m_year.group(0) if m_year else year
    title = "" if "/" in title  else title 
    result = {
        "key": key,
        "author": author,
        "title": title,
        "year": year,
    }
    return result 

    
def extract_references_from_bbl_new(bbl_path):
    with open(bbl_path, 'r', encoding='utf-8') as file:
        content = file.read()
    #print("content:", content)
    pattern = re.compile(
        r'(\\bibitem[\s\S]*?)(?=(?:\r?\n){2}|\\bibitem|\Z)',
        re.DOTALL
    )
    blocks = [m.group(1) for m in pattern.finditer(content)]
    return blocks

def extract_latex_sections(paper_id, temp_dir):
    """Extract TeX files, find the main file, and merge content for section extraction"""
    try:
        # Find all TeX files in the extracted directory
        tex_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.tex'):
                    tex_files.append(os.path.join(root, file))
        
        if not tex_files:
            print(f"  No .tex files found for {paper_id}")
            return None, False
            
        # Find the main TeX file
        main_tex_file = find_main_tex_file(tex_files)
        
        if not main_tex_file:
            print(f"  Could not determine main .tex file for {paper_id}, falling back to largest file")
            # Fall back to using the largest file as before
            main_tex_file = max(tex_files, key=lambda f: os.path.getsize(f), default=None)
            
        if not main_tex_file:
            print(f"  No valid main .tex file found for {paper_id}")
            return None, False
            
        print(f"  Using main TeX file: {os.path.basename(main_tex_file)}")
        
        # Merge content from all included files
        merged_content = merge_tex_content(main_tex_file)
        
        if not merged_content.strip():
            print(f"  Failed to merge TeX content for {paper_id}")
            # Fall back to just the main file content if merging failed
            with open(main_tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                merged_content = f.read()
        
        # Extract sections from the merged content
        outline = process_tex_document(merged_content)
        # Check if the paper has a conclusion section
        has_conclusion = False
        
        def check_for_conclusion(sections):
            for section in sections:
                if 'title' in section and section['title']:
                    title_lower = section['title'].lower()
                    if any(keyword in title_lower for keyword in ['conclusion', 'summary', 'acknowledgment', 'acknowledgement', 'future work', 'discussion']):
                        return True
                if 'children' in section and section['children']:
                    if check_for_conclusion(section['children']):
                        return True
            return False
        
        has_conclusion = check_for_conclusion(outline)
        has_conclusion = True
        #Find all BibTeX files
        bib_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.bib'):
                    bib_files.append(os.path.join(root, file))
        
        return {
            'merged_content': merged_content,
            'outline': outline,
            'main_tex_file': main_tex_file,
            'bib_files': bib_files
        }, has_conclusion
        
    except Exception as e:
        print(f"  Error in extract_latex_sections for {paper_id}: {e}")
        return None, False
from pathlib import Path

def cleanup_temp_dir(temp_dir):
    path = Path(temp_dir)
    for child in path.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
        except Exception as e:
            print(f"删除失败 {child}: {e}")
# %%
def process_paper_improved(paper_id, src_dir):
    """Process a single paper with improved TeX file handling"""
    # Path to the tar.gz file
    #tarball_path = os.path.join(src_dir, f"{paper_id}.tar.gz")
    tarball_path = os.path.join(src_dir, f"{paper_id}.tar.gz")
    if not os.path.exists(tarball_path):
        print(f"  Tarball not found: {tarball_path}")
        return None, False  # Return a tuple (result, has_conclusion)
    
    # Create a temporary directory for extraction
    import tempfile
    import tarfile
    import subprocess
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # First, try using external tar command which is more robust with different formats
            # Fall back to Python's tarfile if that fails
            try:
                # Use the system tar command which is more robust with various compression formats
                subprocess.run(['tar', '-xf', tarball_path, '-C', temp_dir], 
                               check=True, capture_output=True)
                print(f"  Successfully extracted using system tar command")
            except (subprocess.SubprocessError, FileNotFoundError):
                # Fall back to Python's tarfile with different modes
                extraction_successful = False
                
                # Try different opening modes
                for mode in ['r:gz', 'r:bz2', 'r:xz', 'r:']:
                    try:
                        with tarfile.open(tarball_path, mode) as tar:
                            tar.extractall(path=temp_dir)
                            extraction_successful = True
                            print(f"  Successfully extracted using mode: {mode}")
                            break
                    except Exception as e:
                        continue
                
                if not extraction_successful:
                    print(f"  Failed to extract {paper_id} with any archive format")
                    return None, False
            
            # Get paper metadata
            meta = paper_metadata.get(paper_id, {"paper_id": paper_id})
            #print("meta:", meta)
            title = meta.get('title', '')
            authors = meta.get('authors', [])
            year = meta.get('year', '')
            abstract = meta.get('abstract', '')
            
            # Use the improved function to extract LaTeX sections
            latex_data, has_conclusion = extract_latex_sections(paper_id, temp_dir)
            
            if not latex_data:
                print(f"  Failed to extract outline for {paper_id}")
                return None, False
                
            if not has_conclusion:
                print(f"  Paper {paper_id} has no conclusion/summary/acknowledgment section, skipping")
                return None, False
            
            # Format the outline with numbering
            formatted_outline = format_outline_flat(latex_data['outline'])

            ref_mate_bibtex = []
            extracted_titles = set() 
            for bib_file in latex_data['bib_files']:
                try:
                    # Count the number of lines in the bib file first
                    with open(bib_file, 'r', encoding='utf-8', errors='ignore') as f:
                        line_count = sum(1 for _ in f)
                    
                    # Skip if file has more than 8000 lines
                    if line_count > 8000:
                        print(f"  Skipping large BibTeX file: {os.path.basename(bib_file)} ({line_count} lines)")
                        continue
                        
                    # Process the file if it's not too large
                    with open(bib_file, 'r', encoding='utf-8', errors='ignore') as f:
                        bib_content = f.read()
                    parsed_entries = parse_bibtex(bib_content)
                    for entry in parsed_entries:
                        # 使用标题作为唯一标识符
                        if entry.get('title') and entry['title'] not in extracted_titles:
                            ref_mate_bibtex.append(entry)
                            extracted_titles.add(entry['title'])
                except Exception as e:
                    print(f"  Error processing BibTeX file {bib_file}: {e}")   
            
            # Extract reference metadata from BibTeX
            use_bbl = False
            use_bibtex = False 
            bbl_files = []
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.bbl'):
                        bbl_files.append(os.path.join(root, file))
            if not bbl_files:
                print(f"No .bbl files found for {paper_id}")
                use_bbl = False
            if use_bbl:
                selected_bbl = None
                if len(bbl_files) == 1:
                    selected_bbl = bbl_files[0]
                else:
                    main_bbls = [f for f in bbl_files if 'main' in os.path.basename(f).lower()]
                    if main_bbls:
                        selected_bbl = main_bbls[0]
                    else:
                        bbl_names = [os.path.basename(f) for f in bbl_files]
                        if len(set(bbl_names)) == 1: 
                            selected_bbl = bbl_files[0]
                        else:
                            selected_bbl = max(bbl_files, key=os.path.getmtime)
                refs = []
                refs = extract_references_from_bbl_new(selected_bbl)
                ref_meta = []
                print(f"  Found {len(refs)} references in {os.path.basename(selected_bbl)}")
                for idx, ref in enumerate(refs, 1):
                    result = parse_bbl(ref)
                    if result['title']:
                        ref_meta.append(result)

            # Create result object
            result = {
                'meta': meta,
                'outline': formatted_outline,
                'ref_meta': ref_mate_bibtex,
                'have_key': use_bibtex
            }
            return result, True, use_bibtex  # Return the result and True to indicate it has a conclusion
        
        except Exception as e:
            print(f"  Error extracting or processing {paper_id}: {e}")
            return None, False, False
        finally:
            cleanup_temp_dir(temp_dir)
        
output_file = './data_arxiv_math&phy.jsonl'


src_dir = 'your\\path\\to\\src'
processed_count = 0
skipped_count = 0
no_conclusion_count = 0
error_count = 0
pass_count = 0
fail_count = 0
have_ref = 0
ref_max = 0
total_bib = 0
# Adjust the number of papers to process here
papers_to_process = len(paper_ids)

is_first = True

for paper_id in papers_to_process:
    try:
        print(f"Processing paper {paper_id}...")
        result, has_conclusion, use_bib = process_paper_improved(paper_id, src_dir)

        if result:
            filtered_result = {
                'reflen': len(result['ref_meta']),
                'meta': result['meta'],
                'outline': result['outline'],
                'ref_meta': result['ref_meta']
            }

            if filtered_result['reflen'] > 1:
                have_ref += 1
            if filtered_result['reflen'] > ref_max:
                ref_max = filtered_result['reflen']
            if use_bib:
                total_bib += 1
            with open(output_file, 'a', encoding='utf-8') as f:
                if not is_first:
                    f.write('\n')
                else:
                    is_first = False
                json_str = json.dumps(filtered_result, ensure_ascii=False)
                f.write(json_str)

            processed_count += 1
            print(f"  Successfully processed {paper_id}, reflen = {filtered_result['reflen']}")
        else:
            skipped_count += 1
            if not has_conclusion:
                no_conclusion_count += 1
            else:
                error_count += 1
            print(f"  Skipped {paper_id}")
    except Exception as e:
        print(f"  Error processing {paper_id}: {e}")
        error_count += 1
        continue

print(f"\nProcessing summary:")
print(f"Successfully processed: {processed_count}")
print(f"Total skipped: {skipped_count}")
print(f"  - Skipped (no conclusion): {no_conclusion_count}")
print(f"  - Skipped (other errors): {error_count}")
print(f"Results have been written to {output_file}")
print(f"Total have_references found: {have_ref}")
print(f"Max references in a single paper: {ref_max}")
print(f"exact ref by bibtex: {total_bib}")

