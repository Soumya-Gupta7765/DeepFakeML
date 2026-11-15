import os
import base64
import re
import markdown
import subprocess

def convert_md_to_pdf():
    with open('TECHNICAL_REPORT.md', 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown image paths to base64 data URIs so they are fully embedded
    def img_replacer(match):
        alt = match.group(1)
        img_path = match.group(2)
        if not os.path.isabs(img_path):
            resolved = os.path.join(os.getcwd(), img_path)
        else:
            resolved = img_path
        if os.path.exists(resolved):
            with open(resolved, 'rb') as img_f:
                b64_data = base64.b64encode(img_f.read()).decode('utf-8')
            ext = os.path.splitext(resolved)[1].lower().replace('.', '')
            if ext == 'jpg':
                ext = 'jpeg'
            return f'![{alt}](data:image/{ext};base64,{b64_data})'
        return match.group(0)

    md_content_b64 = re.sub(r'!\[(.*?)\]\((.*?)\)', img_replacer, md_content)

    html_body = markdown.markdown(
        md_content_b64,
        extensions=['tables', 'fenced_code']
    )

    css_style = """
    @page {
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        font-size: 10pt;
        line-height: 1.5;
        color: #24292e;
        background-color: #ffffff;
    }
    h1 {
        font-size: 18pt;
        border-bottom: 2px solid #eaecef;
        padding-bottom: 8px;
        margin-top: 0;
        color: #0b2265;
    }
    h2 {
        font-size: 13pt;
        border-bottom: 1px solid #eaecef;
        padding-bottom: 6px;
        margin-top: 22px;
        margin-bottom: 10px;
        color: #1a365d;
        page-break-after: avoid;
    }
    h3 {
        font-size: 11pt;
        margin-top: 16px;
        margin-bottom: 6px;
        color: #2d3748;
        page-break-after: avoid;
    }
    p {
        margin: 6px 0;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 12px 0;
        font-size: 8.5pt;
        page-break-inside: avoid;
    }
    th, td {
        border: 1px solid #dfe2e5;
        padding: 6px 9px;
        text-align: left;
    }
    th {
        background-color: #f1f5f9;
        font-weight: 600;
        color: #0f172a;
    }
    tr:nth-child(2n) {
        background-color: #f8fafc;
    }
    code {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        font-size: 8.5pt;
        background-color: #f1f5f9;
        padding: 1.5px 4px;
        border-radius: 3px;
        color: #0f172a;
    }
    pre {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 5px;
        padding: 10px;
        font-size: 8pt;
        line-height: 1.35;
        overflow-x: auto;
        page-break-inside: avoid;
    }
    pre code {
        background-color: transparent;
        padding: 0;
    }
    img {
        max-width: 95%;
        height: auto;
        display: block;
        margin: 14px auto;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        page-break-inside: avoid;
    }
    hr {
        height: 1px;
        background-color: #e2e8f0;
        border: none;
        margin: 18px 0;
    }
    blockquote {
        border-left: 3px solid #2563eb;
        margin: 10px 0;
        padding: 4px 12px;
        color: #334155;
        background-color: #f0fdf4;
    }
    """

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Technical Report — Deepfake ML Hackathon</title>
<style>{css_style}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    temp_html_path = 'temp_technical_report.html'
    with open(temp_html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)

    chrome_path = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
    pdf_out = os.path.abspath(r'submission_package\TECHNICAL_REPORT.pdf')
    html_abs = os.path.abspath(temp_html_path)

    cmd = [
        chrome_path,
        '--headless=new',
        '--disable-gpu',
        '--no-pdf-header-footer',
        f'--print-to-pdf={pdf_out}',
        html_abs
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    if os.path.exists(temp_html_path):
        os.remove(temp_html_path)

    if os.path.exists(pdf_out):
        size_kb = os.path.getsize(pdf_out) / 1024
        print(f"[+] Successfully exported: {pdf_out} ({size_kb:.1f} KB)")
        # Also copy to root directory
        root_pdf = 'TECHNICAL_REPORT.pdf'
        import shutil
        shutil.copy2(pdf_out, root_pdf)
        print(f"[+] Synced to root: {root_pdf}")
    else:
        print("[-] Error generating PDF")

if __name__ == '__main__':
    convert_md_to_pdf()
