import re


class SecurityAgent:
    PATTERNS = [
        ("api_key", re.compile(r"\b(?:sk|dashscope)-[A-Za-z0-9_-]{12,}\b", re.I)),
        ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
        ("phone", re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")),
        ("asin", re.compile(r"\bB0[A-Z0-9]{8}\b", re.I)),
        ("url", re.compile(r"https?://\S+", re.I)),
        ("account_id", re.compile(r"(?i)(账号|账户|account\s*id)\s*[:：=]?\s*[A-Za-z0-9_-]{5,}")),
    ]

    def __init__(self, custom_terms=None):
        self.custom_terms = [term for term in (custom_terms or []) if term]

    def sanitize(self, text):
        sanitized = text
        findings = []
        for label, pattern in self.PATTERNS:
            if pattern.search(sanitized):
                findings.append(label)
                sanitized = pattern.sub("[已脱敏:{}]".format(label), sanitized)
        for term in self.custom_terms:
            if term in sanitized:
                findings.append("custom_term")
                sanitized = sanitized.replace(term, "[已脱敏:名称]")
        return sanitized, sorted(set(findings))

