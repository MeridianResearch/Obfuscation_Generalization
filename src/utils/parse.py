import re


def extract_xml_answer(text: str) -> str:
    xml_pattern = r'<answer>\s*(.*?)\s*</answer>'
    xml_match = re.search(xml_pattern, text, re.DOTALL)
    if xml_match:
        return xml_match.group(1).strip()

    answer_pattern = r'Answer:\s*\(([A-Za-z])\)'
    answer_match = re.search(answer_pattern, text)
    if answer_match:
        return answer_match.group(1)

    return "No answer tags found"


def extract_third_email_decision(text: str) -> str:
    answer_content = extract_xml_answer(text)
    if answer_content == "No answer tags found":
        return "No answer tags found"

    decisions = [d.strip().upper() for d in answer_content.split(',')]
    decisions = [d for d in decisions if d]

    if len(decisions) == 3:
        return decisions[2]
    if len(decisions) == 1:
        return decisions[0]
    return answer_content


