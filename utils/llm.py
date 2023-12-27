

def parse_llm_response(resp: str, llm_notes: list[str]) -> tuple[str, str]:
    template = ''
    if 'hcl' in resp:
        fields = resp.split('```hcl')
        preface = fields[0]
        fields = fields[1].split('```')
        template = fields[0]
        if len(fields) >= 2:
            llm_notes.append(fields[1])
    else:
        fields = resp.split('```')
        preface = fields[0]
        if len(fields) >= 2:
            template = fields[1]
        if len(fields) >= 3:
            llm_notes.append(fields[2])

    template = template.replace('\\n', '\n')
    template = template.replace('\\"', '\"')
    return (preface, template)


def parse_llm_response_mult_lang(resp: str, langs: list[str]) -> list[str]:
    return [(resp.split('```' + lang)[1]).split('```')[0] for lang in langs if lang in resp]
