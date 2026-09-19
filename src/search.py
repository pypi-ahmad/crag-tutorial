"""Optional keyless web snippets. Failures never become fabricated evidence."""
from dataclasses import dataclass
from ddgs import DDGS
from src.config import MODEL_NAME

@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str

@dataclass
class ExternalSearchReport:
    original_query: str
    search_query: str
    results: list
    external_context: str
    source_attribution: list

class ExternalWebSearcher:
    def __init__(self, model_name=MODEL_NAME, max_results=4):
        if model_name != MODEL_NAME:
            raise ValueError("Only agnes-3.0-flash is supported.")
        self.max_results = max_results

    def search_and_extract(self, question):
        try:
            raw = DDGS().text(question, max_results=self.max_results)
            results = [SearchResult(row.get("title", ""), row.get("body", ""), row.get("href", ""))
                       for row in raw if row.get("body") and row.get("href", "").startswith(("https://", "http://"))]
        except Exception:
            raise RuntimeError("Optional web search failed; no substitute evidence was created.") from None
        return ExternalSearchReport(question, question, results, "\n\n".join(r.snippet for r in results), [r.url for r in results])
