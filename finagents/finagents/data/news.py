"""News documents from Google News RSS.

Filings answer "what did the company report"; they do not answer "why is the
stock moving today", so the news lens needs its own source. Google News RSS is
used because it needs no API key and returns per-ticker query results.

Two honest limitations, both surfaced in the document metadata rather than
hidden: RSS gives headlines and lead paragraphs, not full articles, and the
publisher is the *original* outlet rather than Google. Citing "aggregated via
Google News" is therefore the only accurate provenance, and every document says
so, because a source string that implies first-party reporting would be a lie.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional, Sequence
from xml.etree import ElementTree

import httpx

from ..schemas import Document, EvidenceType

logger = logging.getLogger(__name__)

RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Aggregators reprint wire copy verbatim, so a story is often attributed to a
# dozen outlets. Credibility therefore has to be judged from the outlet name, and
# a wire service is a better source than an unattributed blog.
_SOURCE_WEIGHTS: Sequence[tuple] = (
    (r"reuters", 0.80),
    (r"bloomberg", 0.80),
    (r"financial times|\bft\b", 0.80),
    (r"wall street journal|\bwsj\b", 0.78),
    (r"associated press|\bap\b", 0.75),
    (r"cnbc", 0.72),
    (r"barron", 0.72),
    (r"the economist", 0.72),
    (r"marketwatch", 0.65),
    (r"investor'?s business daily|\bibd\b", 0.65),
    (r"yahoo finance", 0.58),
    (r"motley fool", 0.45),
    (r"zacks", 0.45),
    (r"benzinga", 0.42),
    (r"simply wall st", 0.40),
    (r"seeking alpha", 0.45),
)

_DEFAULT_RELIABILITY = 0.55
_HEADLINE_CHARS = 420


class NewsClient:
    """Fetches recent headlines for a ticker as retrievable documents."""

    def __init__(
        self,
        timeout_s: float = 20.0,
        limit: int = 12,
        min_body_chars: int = 120,
    ) -> None:
        self.limit = limit
        self.min_body_chars = min_body_chars
        self._client = httpx.Client(
            headers={"User-Agent": "Mozilla/5.0 (compatible; finagents/0.1)"},
            timeout=timeout_s,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "NewsClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch(self, ticker: str, limit: Optional[int] = None) -> List[Document]:
        symbol = ticker.upper()
        url = RSS_URL.format(query=f"{symbol} stock when:30d")
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("news fetch failed for %s: %s", symbol, exc)
            return []
        return self.parse(response.text, symbol, limit=limit)

    def parse(
        self, xml: str, ticker: str, limit: Optional[int] = None
    ) -> List[Document]:
        """Parse an RSS payload into documents.

        Kept separate from the fetch so it can be tested against a fixture
        instead of the live endpoint.
        """
        symbol = ticker.upper()
        try:
            root = ElementTree.fromstring(xml)
        except ElementTree.ParseError as exc:
            logger.warning("unparseable news feed for %s: %s", symbol, exc)
            return []

        docs: List[Document] = []
        seen: set = set()
        for item in root.iterfind(".//item"):
            if len(docs) >= (limit or self.limit):
                break
            title = _clean(_text(item, "title"))
            link = _text(item, "link").strip()
            if not title:
                continue
            # Google titles are "Headline - Publisher"; the tail is the outlet.
            source = _text(item, "source").strip()
            if not source and " - " in title:
                title, _, source = title.rpartition(" - ")
            source = source or "Google News"
            # The outlet is already recorded separately, so leaving it on the end
            # of the title just prints the same name twice in the citation.
            if source and title.rstrip().endswith(f" - {source}"):
                title = title.rstrip()[: -(len(source) + 3)].rstrip()
            summary = _clean(_text(item, "description"))
            published = _parse_pubdate(_text(item, "pubDate"))
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)
            body = f"{title}. {summary}" if summary else title
            if len(body) < self.min_body_chars:
                # A headline with no lead paragraph carries no verifiable detail.
                continue
            docs.append(
                Document(
                    doc_id=f"news-{symbol.lower()}-{abs(hash(key)) % 10**10:010d}",
                    title=title[:_HEADLINE_CHARS],
                    text=body[:2000],
                    doc_type=EvidenceType.NEWS,
                    source=f"{source} (via Google News)",
                    published_at=published,
                    url=link or None,
                    reliability=_reliability_for(source),
                    metadata={
                        "tickers": [symbol],
                        "source_kind": "news",
                        "outlet": source,
                        # One collection per outlet, so the retriever's
                        # per-collection cap allows two stories from Reuters and
                        # two from CNBC instead of two from "news" in total. The
                        # cap would otherwise fall back to bucketing every wire
                        # story together, which is the opposite of diversity.
                        "collection": f"news-{symbol.lower()}-{_slug(source)}",
                        "perspectives": ["news", "risk"],
                        # Stated plainly: this is an aggregator, and only a
                        # headline and lead paragraph were read, not the article.
                        "provenance": (
                            "aggregated via Google News: headline and lead paragraph only, "
                            "not the full article"
                        ),
                    },
                )
            )
        return docs


def _text(item: ElementTree.Element, tag: str) -> str:
    node = item.find(tag)
    return (node.text or "") if node is not None else ""


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text or "")).strip()


def _parse_pubdate(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _reliability_for(outlet: str) -> float:
    lowered = (outlet or "").lower()
    for pattern, weight in _SOURCE_WEIGHTS:
        if re.search(pattern, lowered):
            return weight
    return _DEFAULT_RELIABILITY


def _slug(text: str) -> str:
    """A filename-safe fragment of an outlet name, for use in a collection id."""
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "unknown"


__all__ = ["NewsClient", "RSS_URL"]
