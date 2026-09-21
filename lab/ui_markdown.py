"""Small, dependency-free Markdown renderer for the Tkinter Lab UI.

It intentionally renders only the readable subset used by public Agent output
and archived reports.  It never evaluates links, code, or embedded content.
"""
from __future__ import annotations

import re


def markdown_blocks(text: str) -> list[tuple[str, str]]:
    """Return a lightweight, testable block stream without a CommonMark dependency."""
    blocks: list[tuple[str, str]] = []
    in_code, code_language, code_lines = False, "", []
    for raw in (text or "").replace("\r\n", "\n").split("\n"):
        if raw.startswith("```"):
            if in_code:
                blocks.append(("code", "\n".join(code_lines)))
                in_code, code_language, code_lines = False, "", []
            else:
                in_code, code_language = True, raw[3:].strip()
            continue
        if in_code:
            code_lines.append(raw)
            continue
        if re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", raw): blocks.append(("separator", ""))
        elif match := re.match(r"^(#{1,3})\s+(.+)$", raw): blocks.append((f"h{len(match.group(1))}", match.group(2)))
        elif re.match(r"^\s*[-*+]\s+", raw): blocks.append(("bullet", re.sub(r"^\s*[-*+]\s+", "", raw)))
        elif match := re.match(r"^\s*(\d+)\.\s+(.+)$", raw): blocks.append(("ordered", f"{match.group(1)}. {match.group(2)}"))
        elif raw.startswith("> "): blocks.append(("quote", raw[2:]))
        elif raw.startswith("|") and raw.rstrip().endswith("|"): blocks.append(("table", raw))
        else: blocks.append(("paragraph", raw))
    if in_code: blocks.append(("code", "\n".join(code_lines)))
    return blocks


_INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|\[[^\]]+\]\([^)]*\))")


def render_markdown(widget, text: str, *, clear: bool = True) -> None:
    """Render public Markdown into a Tk Text widget using text tags only."""
    if clear:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
    widget.tag_configure("md_h1", font=("Microsoft YaHei UI", 15, "bold"), foreground="#17365d", spacing1=10, spacing3=6)
    widget.tag_configure("md_h2", font=("Microsoft YaHei UI", 13, "bold"), foreground="#1f4e79", spacing1=8, spacing3=4)
    widget.tag_configure("md_h3", font=("Microsoft YaHei UI", 11, "bold"), foreground="#305496", spacing1=6)
    widget.tag_configure("md_bold", font=("Microsoft YaHei UI", 10, "bold"))
    widget.tag_configure("md_italic", font=("Microsoft YaHei UI", 10, "italic"))
    widget.tag_configure("md_code", font=("Consolas", 10), background="#edf0f3", foreground="#7a1f1f")
    widget.tag_configure("md_code_block", font=("Consolas", 10), background="#edf0f3", foreground="#202020", lmargin1=12, lmargin2=12, rmargin=12, spacing1=5, spacing3=5)
    widget.tag_configure("md_quote", foreground="#5f6b73", lmargin1=14, lmargin2=14)
    widget.tag_configure("md_link", foreground="#0563c1", underline=True)
    widget.tag_configure("md_table", font=("Consolas", 9), background="#f4f6f8")
    widget.tag_configure("md_separator", foreground="#9aa5af")
    for kind, value in markdown_blocks(text):
        if kind == "separator":
            widget.insert("end", "────────────────────────────────────────\n", "md_separator")
        elif kind == "code":
            widget.insert("end", value + "\n", "md_code_block")
        elif kind == "table":
            widget.insert("end", value + "\n", "md_table")
        elif kind in {"h1", "h2", "h3"}:
            widget.insert("end", value + "\n", "md_" + kind)
        elif kind == "bullet":
            widget.insert("end", "• ")
            _insert_inline(widget, value)
            widget.insert("end", "\n")
        elif kind == "ordered":
            _insert_inline(widget, value)
            widget.insert("end", "\n")
        elif kind == "quote":
            widget.insert("end", "│ ", "md_quote")
            _insert_inline(widget, value, "md_quote")
            widget.insert("end", "\n")
        else:
            _insert_inline(widget, value)
            widget.insert("end", "\n")
    widget.configure(state="disabled")
    # Public Agent answers may be long.  Schedule after all Markdown blocks
    # were inserted so the reader lands at the actual final line.
    try:
        widget.after_idle(lambda: widget.see("end-1c"))
    except Exception:
        pass


def _insert_inline(widget, value: str, base_tag: str | None = None) -> None:
    for fragment in _INLINE.split(value):
        if not fragment:
            continue
        tag = base_tag
        if fragment.startswith("**") and fragment.endswith("**"):
            fragment, tag = fragment[2:-2], "md_bold"
        elif fragment.startswith("`") and fragment.endswith("`"):
            fragment, tag = fragment[1:-1], "md_code"
        elif fragment.startswith("*") and fragment.endswith("*"):
            fragment, tag = fragment[1:-1], "md_italic"
        elif fragment.startswith("["):
            fragment, tag = fragment[1:fragment.index("]")], "md_link"
        widget.insert("end", fragment, tag)
