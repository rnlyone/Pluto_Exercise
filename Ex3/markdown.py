from markdownify import markdownify as md


def markdown(html_content, txt_file):
    with open(html_content, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    markdown_text = md(html_content)
    plain_text = markdown_text.strip()
    
    with open(txt_file, 'w', encoding='utf-8') as file:
        file.write(plain_text)
        
    print(f'Text has been successfully exported to {txt_file}')
    
    return plain_text



if __name__ == "__main__":
    html = "NLPWIKI.html"
    txt_target = "nlpwiki.txt"
    markdown(html, txt_target)