#!/usr/bin/env python3
"""
OmniTabular Standard Library Documentation Generator (Next.js Edition)
Extracts API signatures and docstrings using `inspect` and generates `page.tsx`.
"""
import sys
import os
import inspect
import html

# Append parent dir so we can import omnitabular
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import omnitabular

def escape(text):
    if not text:
        return ""
    # html.escape escapes <, >, &, ", '. We also need to escape { and } for JSX
    escaped = html.escape(str(text))
    return escaped.replace("{", "&#123;").replace("}", "&#125;")

def parse_docstring(doc):
    if not doc:
        return "<p>No documentation provided.</p>"
    
    html_lines = []
    lines = doc.strip().split('\n')
    in_code = False
    in_section = False
    
    for line in lines:
        stripped = line.strip()
        
        # Headings
        if stripped.endswith("---") or stripped.endswith("==="):
            continue
            
        if stripped in ["Parameters", "Attributes", "Examples", "Workflow"]:
            html_lines.append(f"<h4 className='doc-section-title'>{escape(stripped)}</h4>")
            in_section = True
            continue
            
        if in_section and stripped and not stripped.startswith(" ") and not ':' in stripped and not '>>>' in stripped:
             in_section = False
            
        if stripped.startswith(">>>"):
            if not in_code:
                html_lines.append("<pre><code>")
                in_code = True
            html_lines.append(escape(line))
            continue
        elif in_code and not stripped.startswith(">>>") and stripped != "":
            if line.startswith(" "):
                html_lines.append(escape(line))
                continue
            else:
                html_lines.append("</code></pre>")
                in_code = False
                
        if not stripped:
            if in_code:
                html_lines.append("</code></pre>")
                in_code = False
            html_lines.append("<br />")
            continue
            
        if " :" in line or "\t:" in line or (":" in line and line.split(":")[0].strip().isalpha()):
            parts = line.split(":", 1)
            html_lines.append(f"<strong>{escape(parts[0].strip())}</strong>: {escape(parts[1].strip())}<br />")
        else:
            html_lines.append(escape(line) + "<br />")
            
    if in_code:
        html_lines.append("</code></pre>")
        
    return "".join([l + "\n" if "<pre>" not in l else l for l in html_lines])

def generate_class_html(cls):
    content = f"<div className='api-card glass reveal' id='{cls.__name__}'>\n"
    content += f"<h2>{cls.__name__}</h2>\n"
    
    content += f"<div className='docstring'>\n{parse_docstring(cls.__doc__)}\n</div>\n"
    
    try:
        sig = inspect.signature(cls.__init__)
        content += f"<h3>__init__<span className='signature'>{escape(str(sig))}</span></h3>\n"
    except ValueError:
        pass
        
    content += "<div className='methods-container'>\n"
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith("_"):
            try:
                sig = inspect.signature(method)
            except ValueError:
                sig = "(...)"
            
            content += f"<div className='method-block'>\n"
            content += f"<h4>{name}<span className='signature'>{escape(str(sig))}</span></h4>\n"
            content += f"<div className='method-doc'>\n{parse_docstring(method.__doc__)}\n</div>\n"
            content += f"</div>\n"
    content += "</div>\n"
    content += "</div>\n"
    return content

def build():
    classes_to_doc = [
        omnitabular.OmniTabular,
        omnitabular.HillClimbingEnsemble,
        omnitabular.WeightedEnsemble,
    ]
    
    # We output a TSX file returning a React component
    tsx_content = [
        "'use client';",
        "",
        "import { useEffect } from 'react';",
        "import Link from 'next/link';",
        "",
        "export default function ApiReference() {",
        "  useEffect(() => {",
        "    function reveal() {",
        "      const reveals = document.querySelectorAll('.reveal');",
        "      for (let i = 0; i < reveals.length; i++) {",
        "        const windowHeight = window.innerHeight;",
        "        const elementTop = reveals[i].getBoundingClientRect().top;",
        "        const elementVisible = 100;",
        "        if (elementTop < windowHeight - elementVisible) {",
        "          reveals[i].classList.add('active');",
        "        }",
        "      }",
        "    }",
        "    window.addEventListener('scroll', reveal);",
        "    reveal();",
        "    return () => window.removeEventListener('scroll', reveal);",
        "  }, []);",
        "",
        "  return (",
        "    <>",
        "      <div className='blob-bg'>",
        "        <div className='blob blob-1'></div>",
        "        <div className='blob blob-2'></div>",
        "        <div className='blob blob-3'></div>",
        "      </div>",
        "",
        "      <nav className='navbar glass'>",
        "        <div className='logo'>OmniTabular</div>",
        "        <ul className='nav-links'>",
        "          <li><Link href='/'>Home</Link></li>",
        "          <li><Link href='/#documentation'>Documentation</Link></li>",
        "          <li><a href='https://github.com/omnitabular/omnitabular' target='_blank' rel='noreferrer' className='btn primary-btn'>GitHub</a></li>",
        "        </ul>",
        "      </nav>",
        "",
        "      <main style={{ padding: '150px 10%' }}>",
        "        <h1 style={{ textAlign: 'center', fontSize: '4rem', marginBottom: '4rem' }}>",
        "          API <span className='gradient-text'>Reference</span>",
        "        </h1>",
    ]
    
    for cls in classes_to_doc:
        tsx_content.append(generate_class_html(cls))
        
    tsx_content.extend([
        "      </main>",
        "    </>",
        "  );",
        "}"
    ])
    
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'website', 'src', 'app', 'api-reference', 'page.tsx'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(tsx_content))
    print(f"Next.js Page generated successfully at {output_path}")

if __name__ == "__main__":
    build()
