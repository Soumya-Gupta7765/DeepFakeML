import os
import markdown
import subprocess
import pypdf

with open('report.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

css = """
@page {
    size: A4;
    margin: 12mm 12mm 12mm 12mm;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    font-size: 8.5pt;
    line-height: 1.35;
    color: #1e293b;
    margin: 0;
}
h1 {
    font-size: 13pt;
    margin: 0 0 3px 0;
    color: #0f172a;
    border-bottom: 2px solid #2563eb;
    padding-bottom: 2px;
}
h2 {
    font-size: 10pt;
    margin: 8px 0 3px 0;
    color: #1e40af;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 2px;
}
h3 {
    font-size: 8.5pt;
    margin: 5px 0 2px 0;
    color: #334155;
}
p {
    margin: 3px 0;
}
ul {
    margin: 3px 0;
    padding-left: 16px;
}
li {
    margin: 1.5px 0;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 5px 0;
    font-size: 7.5pt;
}
th, td {
    border: 1px solid #cbd5e1;
    padding: 3px 5px;
    text-align: left;
}
th {
    background-color: #f1f5f9;
    font-weight: 600;
}
tr:nth-child(2n) {
    background-color: #f8fafc;
}
code {
    font-family: Consolas, monospace;
    font-size: 7.5pt;
    background-color: #f1f5f9;
    padding: 1px 2px;
    border-radius: 2px;
}
pre {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 5px;
    font-size: 6.8pt;
    line-height: 1.25;
    margin: 3px 0;
}
hr {
    border: none;
    height: 1px;
    background-color: #e2e8f0;
    margin: 6px 0;
}
"""

full_html = f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>{css}</style></head><body>{html_body}</body></html>'

temp_html = 'temp_short_report.html'
with open(temp_html, 'w', encoding='utf-8') as f:
    f.write(full_html)

chrome = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
pdf_out = os.path.abspath('report.pdf')
cmd = [chrome, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={pdf_out}', os.path.abspath(temp_html)]
subprocess.run(cmd, capture_output=True)

if os.path.exists(temp_html):
    os.remove(temp_html)

reader = pypdf.PdfReader('report.pdf')
print('report.pdf generated! Total pages:', len(reader.pages))

# Copy to submission_package/report.pdf as well
import shutil
shutil.copy2('report.pdf', 'submission_package/report.pdf')
print('Synced to submission_package/report.pdf!')
