"""SEC EDGAR client.

EDGAR is the authoritative free source for US filings and for the XBRL financial
facts extracted from them. Two properties make it the backbone of an online-only
version of this system:

* **Filings are primary documents.** A number in a 10-K is a number the company
  filed with a regulator, which is a stronger evidentiary basis than anything a
  news aggregator can offer.
* **XBRL facts are machine-readable.** The verifier can check an agent's figure
  against the tagged value rather than against prose, which removes most
  sentence-scraping error from the numeric check.

EDGAR requires a declared User-Agent and asks for no more than 10 requests per
second, so this client rate-limits itself and caches responses on disk: repeated
research on the same company should not re-download megabytes of filings.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..schemas import Document, EvidenceType

logger = logging.getLogger(__name__)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{filename}"
)

# EDGAR's fair-access limit is 10 requests/second. Stay well under it: this
# client is a research tool, not a bulk harvester, and the archive does throttle.
_MIN_INTERVAL_S = 0.34

# EDGAR endpoints occasionally drop a read on a busy or flaky link. One retry
# turns a transient timeout into a normal request; without it a single dropped
# response on company_tickers.json leaves the whole run unable to resolve any
# ticker at all.
_MAX_ATTEMPTS = 2


def _ensure_cache_dir(cache_dir: Path) -> Path:
    """Create the cache directory, falling back if the path is unusable.

    A relative cache dir is resolved against the current working directory,
    which can vanish (a temp checkout, a deleted folder) and turn `mkdir` into
    `FileNotFoundError: [WinError 2]`. A missing cache is not worth failing a
    research run over, so fall back to the system temp directory.
    """

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / "finagents" / "sec"
        logger.warning("cache dir %s unusable (%s); using %s", cache_dir, exc, fallback)
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# A real identity, not a library string. The SEC asks for "sample company name
# admin contact name admin@example.com" and throttles callers that send
# something generic. Overridable with FINAGENTS_SEC_USER_AGENT.
DEFAULT_SEC_USER_AGENT = "finagents/0.1 (research; set FINAGENTS_SEC_USER_AGENT)"

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")
# Inline XBRL leaves facts in the HTML; they are noise once the text is read.
_XBRL_TAG_RE = re.compile(r"</?ix:[^>]*>", re.IGNORECASE)
# Machine-readable bookkeeping that is hidden in the rendered document but
# present in the bytes. A 10-Q carries 600+ <xbrli:context> blocks listing every
# fact's dimensions and periods; left in, they bury the prose under tens of
# thousands of characters of `us-gaap:FairValueInputsLevel2Member 2026-01-25`
# and consume the entire extraction budget before the first sentence of the MD&A
# is reached. Nothing a reader sees belongs to these blocks.
_XBRL_METADATA_RE = re.compile(
    r"<ix:header\b.*?</ix:header\s*>"
    r"|<xbrli:(?:context|unit|entity)\b.*?</xbrli:(?:context|unit|entity)\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Section headings in 10-K/10-Q/8-K filings. Splitting on these is what makes a
# filing retrievable: "Item 1A. Risk Factors" and "Item 7. Management's
# Discussion" answer different questions and should not score identically.
#
# Matching is by canonical SEC item title rather than by line position, because
# `html_to_text` collapses newlines: after flattening, a heading is no longer on
# a line of its own and a `^...$` pattern can never fire. Requiring the title to
# be one the SEC actually prescribes also avoids splitting mid-sentence, which a
# generic "Item N. <anything>" pattern would do.
_SECTION_TITLES: tuple[str, ...] = (
    "Management's Discussion and Analysis of Financial Condition",
    "Quantitative and Qualitative Disclosures About Market Risk",
    "Financial Statements and Supplementary Data",
    "Market for Registrant's Common Equity",
    "Submission of Matters to a Vote of Security Holders",
    "Submission of Matters to a Vote of Securityholders",
    "Changes in Registrant's Certifying Accountant",
    "Mine Safety Disclosures",
    "Exhibits and Financial Statement Schedules",
    "Exhibits, Financial Statement Schedules",
    "Notes to Condensed Consolidated Financial Statements",
    "Management's Discussion and Analysis",
    "Management's Discussion",
    "Risk Factors",
    "Business, Risk Factors",
    "Controls and Procedures",
    "Legal Proceedings",
    "Properties",
    "Business",
    "Other Information",
    "Financial Statements",
)
_SECTION_RE = re.compile(
    r"(?:ITEM\s+\d+[A-Z]?\.?\s+(?:"
    + "|".join(re.escape(t) for t in sorted(_SECTION_TITLES, key=len, reverse=True))
    + r")\b\.?)"
    r"|(?:PART\s+[IVX]{1,4}\b\.?)",
    re.IGNORECASE,
)

# Forms worth reading, and what each is good for. 8-K is included because
# material events (guidance changes, charges) show up there first.
INTERESTING_FORMS = {
    "10-K": (0.95, "annual report"),
    "10-Q": (0.90, "quarterly report"),
    "8-K": (0.78, "current report"),
    "DEF 14A": (0.60, "proxy statement"),
}

# XBRL metric name -> candidate tags, most-standard first. Which tag a filer
# uses changes over time: NVIDIA tagged ASC 606 revenue through FY2021 and then
# switched to the legacy `Revenues` tag, so ranking by tag priority and taking
# the first hit returns a stale $26.9bn figure. Candidates are therefore compared
# by reporting period, and the tag order only breaks ties between equal periods.
FACT_TAG_PRIORITY: Dict[str, List[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "research_and_development": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "eps_basic": ["EarningsPerShareBasic"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "shareholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capital_expenditure": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ],
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}

# Tags that are naturally percentages or ratios rather than USD amounts.
_RATIO_TAGS = {"eps_diluted", "eps_basic"}


class RateLimiter:
    """Minimum-interval limiter shared by all EDGAR endpoints."""

    def __init__(self, min_interval: float = _MIN_INTERVAL_S) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()


class SECClient:
    """Read-only EDGAR access: ticker resolution, filings and XBRL facts."""

    def __init__(
        self,
        user_agent: str = DEFAULT_SEC_USER_AGENT,
        cache_dir: Optional[Path] = None,
        timeout_s: float = 30.0,
        max_filings: int = 4,
        min_interval_s: float = _MIN_INTERVAL_S,
    ) -> None:
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        self.max_filings = max_filings
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/finagents/sec")
        self.cache_dir = _ensure_cache_dir(self.cache_dir)
        self.limiter = RateLimiter(min_interval_s)
        if user_agent == DEFAULT_SEC_USER_AGENT:
            # The SEC's fair-access policy requires a declared identity, and
            # throttles or blocks anonymous traffic. Say so once, loudly, at
            # startup rather than serving a 403 and failing the research run.
            logger.warning(
                "SEC requests are using the placeholder User-Agent %r. Set "
                "FINAGENTS_SEC_USER_AGENT to 'name email' - the SEC requires a "
                "declared contact and will throttle anonymous access.",
                user_agent,
            )
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout_s,
            follow_redirects=True,
        )
        self._ticker_map: Optional[Dict[str, Tuple[int, str]]] = None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SECClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- transport --------------------------------------------------------

    def _get(self, url: str, cache_key: str, use_cache: bool = True) -> Optional[Any]:
        path = self.cache_dir / f"{cache_key}.json"
        if use_cache and path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                path.unlink(missing_ok=True)
        last_error: Optional[Exception] = None
        for _ in range(_MAX_ATTEMPTS):
            self.limiter.wait()
            try:
                response = self._client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                last_error = exc
                continue
            try:
                payload = response.json()
            except ValueError:
                logger.warning("EDGAR returned non-JSON for %s", url)
                return None
            path.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        logger.warning("EDGAR request failed (%s): %s", url, last_error)
        if use_cache and path.exists():
            # Stale data beats no data for a filing index.
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _get_text(self, url: str, cache_key: str) -> Optional[str]:
        path = self.cache_dir / f"{cache_key}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        self.limiter.wait()
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("filing download failed (%s): %s", url, exc)
            return None
        # Filings are frequently windows-1252, not utf-8. Letting httpx guess
        # produces U+FFFD in the middle of sentences, which then propagates into
        # the evidence the verifier reads and can quote back.
        raw = response.content
        text = ""
        for encoding in ("utf-8", "cp1252", "latin-1"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if not text:
            text = raw.decode("utf-8", errors="replace")
        path.write_text(text, encoding="utf-8", errors="replace")
        return text

    # -- ticker resolution ------------------------------------------------

    def ticker_map(self) -> Dict[str, Tuple[int, str]]:
        """ticker -> (CIK, company name), from EDGAR's official mapping."""
        if self._ticker_map is None:
            payload = self._get(TICKER_MAP_URL, "company_tickers")
            if not isinstance(payload, dict):
                # Do NOT cache the failure. An empty map resolves every symbol to
                # None, which turns the whole run tickerless; leaving it unset
                # lets the next call retry, and the on-disk cache makes that free
                # once the link recovers.
                logger.warning(
                    "EDGAR ticker map unavailable; ticker resolution is degraded "
                    "until it can be fetched"
                )
                return {}
            mapping: Dict[str, Tuple[int, str]] = {}
            for row in payload.values():
                if not isinstance(row, dict):
                    continue
                symbol = str(row.get("ticker", "")).upper()
                if not symbol:
                    continue
                mapping[symbol] = (int(row["cik_str"]), str(row.get("title", "")))
            self._ticker_map = mapping
            logger.info("EDGAR ticker map: %d symbols", len(mapping))
        return self._ticker_map

    def is_known_ticker(self, symbol: str) -> bool:
        return symbol.upper() in self.ticker_map()

    def company_name(self, ticker: str) -> Optional[str]:
        entry = self.ticker_map().get(ticker.upper())
        return entry[1] if entry else None

    def cik(self, ticker: str) -> Optional[int]:
        entry = self.ticker_map().get(ticker.upper())
        return entry[0] if entry else None

    def ticker_for_company_name(self, name: str) -> Optional[str]:
        needle = name.strip().lower()
        if not needle:
            return None
        for symbol, (_cik, company) in self.ticker_map().items():
            if company.lower() == needle:
                return symbol
        return None

    # -- filings ----------------------------------------------------------

    def recent_filings(
        self, ticker: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Most recent interesting filings for a ticker, newest first."""
        cik = self.cik(ticker)
        if cik is None:
            return []
        payload = self._get(SUBMISSIONS_URL.format(cik=cik), f"submissions_{cik:010d}")
        if not isinstance(payload, dict):
            return []
        recent = payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", []) or []
        accessions = recent.get("accessionNumber", []) or []
        primaries = recent.get("primaryDocument", []) or []
        filing_dates = recent.get("filingDate", []) or []
        report_dates = recent.get("reportDate", []) or []
        out: List[Dict[str, Any]] = []
        for idx, form in enumerate(forms):
            if form not in INTERESTING_FORMS:
                continue
            if idx >= len(accessions) or idx >= len(primaries):
                continue
            out.append(
                {
                    "form": form,
                    "accession": accessions[idx],
                    "primary_document": primaries[idx],
                    "filing_date": (
                        filing_dates[idx] if idx < len(filing_dates) else None
                    ),
                    "report_date": (
                        report_dates[idx] if idx < len(report_dates) else None
                    ),
                }
            )
        out.sort(key=lambda f: f.get("filing_date") or "", reverse=True)
        return out[: limit or self.max_filings]

    def filing_text(
        self,
        ticker: str,
        accession: str,
        primary_document: str,
        max_chars: int = 400_000,
    ) -> Optional[str]:
        """Download a filing's primary document and reduce it to plain text."""
        cik = self.cik(ticker)
        if cik is None:
            return None
        url = ARCHIVE_URL.format(
            cik=cik,
            accession_nodash=accession.replace("-", ""),
            filename=primary_document,
        )
        raw = self._get_text(url, f"filing_{cik:010d}_{accession.replace('-', '')}")
        if raw is None:
            return None
        return html_to_text(raw, max_chars=max_chars)

    def filing_documents(
        self,
        ticker: str,
        limit: Optional[int] = None,
        max_chunks_per_filing: int = 24,
        max_chunks_per_section: int = 3,
        chunk_chars: int = 1600,
    ) -> List[Document]:
        """Recent filings as retrievable, section-scoped documents.

        A 10-Q runs to hundreds of thousands of characters, which is neither
        affordable in a prompt nor useful as a single retrievable unit: every
        query about margins would match the whole filing equally. Filings are
        therefore split at their own section headings ("Item 1A. Risk Factors")
        and then windowed, so retrieval can return the risk-factor section for a
        risk question and the revenue discussion for a fundamentals question.

        The chunk budget is then allocated by value, because filing order is not
        value order. "Item 1. Financial Statements" is the first numbered section
        of a 10-Q and by volume alone it consumes the entire budget with table
        fragments like "$ 5,602 $ 3,958" - pushing "Item 1A. Risk Factors" and
        the MD&A, which arrive later, out of the corpus altogether. So table-like
        chunks are dropped, each section is capped, and the figures that matter
        are already served by the tagged XBRL document.

        Reliability stays tied to form type, and every chunk carries its parent
        accession so the diversity pass can stop one verbose filing from taking
        every slot.
        """
        docs: List[Document] = []
        for filing in self.recent_filings(ticker, limit=limit):
            form = filing["form"]
            reliability, label = INTERESTING_FORMS.get(form, (0.6, "filing"))
            text = self.filing_text(
                ticker, filing["accession"], filing["primary_document"]
            )
            if not text or len(text) < 400:
                logger.info("skipping %s %s: no usable text", ticker, form)
                continue
            filed = filing.get("filing_date") or "undated"
            slug = filed.replace("-", "")
            collection = f"sec-{ticker.lower()}-{form.replace(' ', '').lower()}-{slug}"
            url = ARCHIVE_URL.format(
                cik=self.cik(ticker) or 0,
                accession_nodash=filing["accession"].replace("-", ""),
                filename=filing["primary_document"],
            )
            sections = _prioritise_sections(
                split_into_sections(text, max_chars=chunk_chars),
                per_section=max_chunks_per_section,
                total=max_chunks_per_filing,
            )
            for index, (heading, body) in enumerate(sections):
                docs.append(
                    Document(
                        doc_id=f"{collection}-c{index:02d}",
                        title=f"{ticker} {form} {label} - {heading or 'section ' + str(index + 1)}",
                        text=body,
                        doc_type=_doc_type_for_form(form),
                        source="SEC EDGAR",
                        published_at=_parse_date(filing.get("filing_date")),
                        url=url,
                        reliability=reliability,
                        metadata={
                            "tickers": [ticker.upper()],
                            "form": form,
                            "accession": filing["accession"],
                            "report_date": filing.get("report_date"),
                            "source_kind": "regulatory_filing",
                            "perspectives": _perspectives_for_form(form),
                            # `collection` groups the chunks of one filing so the
                            # retriever can cap them; it is not a citation field.
                            "collection": collection,
                            "parent_doc_id": collection,
                            "section": heading,
                            "chunk_index": index,
                        },
                    )
                )
            if len(sections) > max_chunks_per_filing:
                logger.info(
                    "%s %s: kept %d of %d sections",
                    ticker,
                    form,
                    max_chunks_per_filing,
                    len(sections),
                )
        return docs

    # -- XBRL facts -------------------------------------------------------

    def company_facts(self, ticker: str) -> Dict[str, Any]:
        cik = self.cik(ticker)
        if cik is None:
            return {}
        payload = self._get(COMPANY_FACTS_URL.format(cik=cik), f"facts_{cik:010d}")
        return payload if isinstance(payload, dict) else {}

    def key_financials(self, ticker: str) -> Dict[str, float]:
        """Latest annual value for each mapped XBRL metric. See `annual_facts`."""
        return self.annual_facts(ticker)[0]

    def annual_facts(self, ticker: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Return (metrics, period) for the most recent annual figures.

        Annual (`FY`, 10-K) frames are preferred over quarterly because a margin
        or revenue figure quoted without a period is nearly always a full-year
        figure, and mixing the two is how a report ends up comparing a quarterly
        margin against an annual one.

        The period travels with the numbers rather than being recomputed
        elsewhere, because a document that states "$215.9bn revenue" without
        naming the fiscal year invites the reader to assume it is the current
        one - and when the two diverge, that assumption is the whole error.
        Metrics may come from different tags with different period ends, so
        `period` reports the newest and `per_metric_periods` records the rest.
        """
        facts = self.company_facts(ticker)
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        out: Dict[str, float] = {}
        ends: Dict[str, str] = {}
        filed: Optional[str] = None
        for name, tags in FACT_TAG_PRIORITY.items():
            best: Optional[Tuple[str, float, int]] = None
            for rank, tag in enumerate(tags):
                if tag not in us_gaap:
                    continue
                found = _latest_annual(us_gaap[tag].get("units", {}))
                if found is None:
                    continue
                end, value = found
                # Latest reporting period wins; the earlier tag breaks ties.
                if best is None or (end, -rank) > (best[0], -best[2]):
                    best = (end, value, rank)
            if best is not None:
                out[name] = best[1]
                ends[name] = best[0]
                filed = _latest_filed(us_gaap, tags)
        # Derived ratios need both inputs, and a margin outside 0-100% means the
        # inputs disagree about the period. Reporting a 570% gross margin would
        # be worse than reporting no margin, so the ratio is dropped and the
        # inconsistency is logged.
        for name, numerator in (
            ("gross_margin", "gross_profit"),
            ("operating_margin", "operating_income"),
            ("net_margin", "net_income"),
        ):
            if numerator in out and out.get("revenue"):
                pct = 100.0 * out[numerator] / out["revenue"]
                if 0.0 <= pct <= 100.0:
                    out[name] = round(pct, 4)
                else:
                    logger.warning(
                        "%s: dropping %s (%.1f%%) because it implies revenue of %s "
                        "against %s of %s; the tags disagree on period",
                        ticker,
                        name,
                        pct,
                        f"{out['revenue']:,.0f}",
                        numerator,
                        f"{out[numerator]:,.0f}",
                    )
        if not out:
            return out, {}
        newest = max(ends.values())
        period: Dict[str, Any] = {
            "period_end": newest,
            "fiscal_year": _fiscal_year_from_end(newest),
            "form": "10-K",
            "filed": filed,
            "per_metric_periods": dict(sorted(set(ends.items()), key=lambda kv: kv[0])),
        }
        return out, period

    def financials_document(self, ticker: str) -> Optional[Document]:
        """Render the XBRL facts as a document.

        This is the highest-trust document in the corpus: every figure is a
        value the issuer tagged in a filing, so the verifier can check a claim
        about revenue or EPS against an exact number rather than a sentence.
        """
        metrics, period = self.annual_facts(ticker)
        if not metrics:
            return None
        name = self.company_name(ticker) or ticker
        facts = self.company_facts(ticker)
        entity = facts.get("entityName", name)
        period_end = period.get("period_end")
        fiscal_year = period.get("fiscal_year")
        when = f" for fiscal year {fiscal_year}" if fiscal_year else ""
        lines = [
            f"{entity} ({ticker.upper()}) reported figures as tagged in its XBRL filings "
            f"with the SEC. Values are the most recent annual figures available{when}"
            + (f", for the period ended {period_end}." if period_end else "."),
            "",
        ]
        for key, value in sorted(metrics.items()):
            if key in _RATIO_TAGS:
                lines.append(f"{key.replace('_', ' ')}: {value:,.2f} per diluted share")
            elif key.endswith(("_margin",)) or key == "gross_margin":
                lines.append(f"{key.replace('_', ' ')}: {value:,.1f} percent")
            elif abs(value) >= 1_000_000:
                lines.append(
                    f"{key.replace('_', ' ')}: {value:,.0f} USD ({value / 1e9:,.2f} billion)"
                )
            else:
                lines.append(f"{key.replace('_', ' ')}: {value:,.0f} USD")
        # Where the tags disagree on period, say so rather than presenting a
        # blended picture as one fiscal year.
        period_ends = set(period.get("per_metric_periods", {}).values())
        if len(period_ends) > 1:
            lines.append("")
            lines.append(
                "Note: individual tags carry different period ends ("
                + ", ".join(sorted(period_ends))
                + "); the figures above are each the latest annual value available for "
                "that tag."
            )
        filed = _parse_date(period.get("filed"))
        return Document(
            doc_id=f"sec-{ticker.lower()}-xbrl-facts",
            title=(
                f"{entity} ({ticker.upper()}) XBRL financial facts"
                + (f", FY{fiscal_year}" if fiscal_year else "")
            ),
            text="\n".join(lines),
            doc_type=EvidenceType.FILING,
            source="SEC EDGAR XBRL company facts",
            # The date the figures became public, not the date this code ran.
            published_at=filed or datetime.now(timezone.utc),
            url=COMPANY_FACTS_URL.format(cik=self.cik(ticker) or 0),
            reliability=0.95,
            metadata={
                "tickers": [ticker.upper()],
                "source_kind": "xbrl_facts",
                "perspectives": ["fundamental", "risk"],
                "metrics": metrics,
                "collection": f"sec-{ticker.lower()}-xbrl-facts",
                "period_end": period.get("period_end"),
                "fiscal_year": period.get("fiscal_year"),
                "period_form": period.get("form"),
                "filed": period.get("filed"),
                "per_metric_periods": period.get("per_metric_periods", {}),
            },
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _prose_density(text: str) -> float:
    """Fraction of tokens that are words, as opposed to digits and punctuation.

    A rendered balance sheet scores near zero ("$ 5,602 $ 3,958 Deferred revenue
    1,796 1,193"). A sentence of MD&A scores near one. This is the cheapest
    reliable way to tell narrative from table without parsing HTML structure that
    the inline-XBRL transformations have already rewritten.
    """
    tokens = re.findall(r"[A-Za-z]{2,}|\d+|[^\s\w]", text)
    if not tokens:
        return 0.0
    words = sum(1 for t in tokens if t[0].isalpha())
    return words / len(tokens)


# Below this, a chunk is a table fragment or a signature block: not useful as
# prose evidence, and cheap to re-derive from the tagged XBRL document.
_MIN_PROSE_DENSITY = 0.55


def _prioritise_sections(
    sections: List[Tuple[str, str]], per_section: int, total: int
) -> List[Tuple[str, str]]:
    """Drop table-like chunks, cap each section, and keep document order.

    Order is restored at the end so `chunk_index` still increases through the
    filing, and a reader following the citations moves forward rather than
    jumping around.
    """
    if not sections:
        return []
    narrative = [s for s in sections if _prose_density(s[1]) >= _MIN_PROSE_DENSITY]
    if not narrative:
        # A filing made of tables (an 8-K exhibit, a short press release) is
        # still evidence. Keep the densest few rather than nothing.
        narrative = sorted(sections, key=lambda s: -_prose_density(s[1]))[:per_section]
    kept: List[Tuple[str, str, int]] = []
    per_heading: Dict[str, int] = {}
    for order, (heading, body) in enumerate(narrative):
        count = per_heading.get(heading, 0)
        if count >= per_section:
            continue
        per_heading[heading] = count + 1
        kept.append((heading, body, order))
    kept.sort(key=lambda item: item[2])
    return [(h, b) for h, b, _ in kept[:total]]


def split_into_sections(text: str, max_chars: int = 1600) -> List[Tuple[str, str]]:
    """Split filing text into (heading, body) chunks at section boundaries.

    Sections longer than `max_chars` are windowed with a small overlap, because
    a risk factor that straddles a boundary still reads correctly with a little
    context carried over, while one that loses its subject does not.
    """
    if not text:
        return []
    marks = list(_SECTION_RE.finditer(text))
    sections: List[Tuple[str, str]] = []
    if not marks:
        # 8-Ks and press releases have no numbered items. They are still worth
        # retrieving, so they are windowed whole rather than discarded.
        return _window("whole document", text, max_chars)
    if marks[0].start() > 0:
        sections.extend(_window("front matter", text[: marks[0].start()], max_chars))
    for index, match in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        heading = " ".join(match.group(0).split())
        sections.extend(_window(heading, text[match.end() : end], max_chars))
    return [s for s in sections if len(s[1].strip()) > 200]


def _window(
    heading: str, body: str, max_chars: int, overlap: int = 200
) -> List[Tuple[str, str]]:
    # A heading match that stops before its own trailing period leaves the body
    # starting ". ", which then gets quoted as if it were the filing's wording.
    body = body.lstrip(". \t").strip()
    if not body:
        return []
    if len(body) <= max_chars:
        return [(heading, body)]
    out: List[Tuple[str, str]] = []
    start = 0
    step = max(1, max_chars - overlap)
    while start < len(body):
        piece = body[start : start + max_chars]
        # Prefer to break on a sentence end so a chunk is not cut mid-clause.
        cut = piece.rfind(". ")
        if cut > max_chars * 0.6:
            piece = piece[: cut + 1]
        out.append((heading, piece.strip()))
        if start + max_chars >= len(body):
            break
        # Overlap must not resume mid-word: "hether the registrant" reads as
        # corruption, and the agent may quote it.
        nxt = start + step
        space = body.find(" ", nxt)
        start = space + 1 if 0 <= space < nxt + 40 else nxt
    return out


def html_to_text(html: str, max_chars: int = 400_000) -> str:
    """Reduce a filing to readable text.

    Filings are 1-15 MB of HTML with inline-XBRL markup. The goal is a
    searchable, readable wall of text, not a faithful rendering: tags go, script
    and style blocks go, the hidden XBRL metadata goes, entities are decoded,
    and the result is truncated to a budget that keeps prompts affordable.

    Inline-XBRL spans are *unwrapped*, not deleted. In a 10-Q the visible figure
    lives inside `<ix:nonFraction>215.9</ix:nonFraction>`, so removing the span
    with its contents throws away every tagged number in the document - and the
    tagged numbers are the only part of the filing that is machine-verifiable.

    The character budget is applied to the extracted text, not to the download.
    A 10-Q's markup runs about 1.5 MB but only a fraction of that is prose, and
    truncating the markup first cuts mid-tag and leaves the hidden XBRL header as
    the only readable content.
    """
    text = _XBRL_METADATA_RE.sub(" ", html)
    # Unwrap: drop the ix: tags, keep everything between them.
    text = _XBRL_TAG_RE.sub("", text)
    text = _SCRIPT_RE.sub(" ", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|tr|h\d|li)>", "\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub(" ", text)
    for entity, char in (
        ("&nbsp;", " "),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&rsquo;", "'"),
        ("&lsquo;", "'"),
        ("&mdash;", "-"),
        ("&ndash;", "-"),
        ("&hellip;", "..."),
        ("&#160;", " "),
    ):
        text = text.replace(entity, char)
    text = re.sub(
        r"&#(\d+);",
        lambda m: chr(int(m.group(1))) if int(m.group(1)) < 0x110000 else " ",
        text,
    )
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    text = text.strip()
    if len(text) > max_chars:
        # Prefer the front of a filing, which holds the financials and the risk
        # factors, over the middle, which holds exhibits and signatures.
        text = text[:max_chars].rsplit(" ", 1)[0] + " [truncated]"
    return text


def _doc_type_for_form(form: str) -> EvidenceType:
    if form in ("10-K", "10-Q", "DEF 14A"):
        return EvidenceType.FILING
    if form == "8-K":
        return EvidenceType.REGULATORY
    return EvidenceType.FILING


def _perspectives_for_form(form: str) -> List[str]:
    if form in ("10-K", "10-Q"):
        return ["fundamental", "risk"]
    if form == "8-K":
        return ["news", "risk"]
    return ["risk"]


def _latest_annual(units: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    """Most recent FY 10-K value as (period end, value), across available units."""
    best: Optional[Tuple[str, float]] = None
    for unit_name in ("USD", "USD/shares", "shares", "pure"):
        rows = units.get(unit_name)
        if not isinstance(rows, list) or not rows:
            continue
        annual = [
            row
            for row in rows
            if row.get("form") in ("10-K", "10-K/A")
            and row.get("fp") == "FY"
            and row.get("end")
            and row.get("val") is not None
        ]
        if not annual:
            continue
        annual.sort(key=lambda row: (row["end"], row.get("filed", "")))
        end, value = annual[-1]["end"], float(annual[-1]["val"])
        if best is None or end > best[0]:
            best = (end, value)
    return best


def _latest_filed(us_gaap: Dict[str, Any], tags: List[str]) -> Optional[str]:
    """Newest `filed` date across the tags used, i.e. when the figures went public."""
    best: Optional[str] = None
    for tag in tags:
        rows = us_gaap.get(tag, {}).get("units", {}).get("USD")
        if not isinstance(rows, list):
            continue
        for row in rows:
            filed = row.get("filed")
            if filed and (best is None or filed > best):
                best = filed
    return best


def _fiscal_year_from_end(period_end: str) -> Optional[int]:
    """SEC period-end dates are `YYYY-MM-DD`; the fiscal year is the calendar
    year they fall in, which is correct for the January-year-end filers that
    dominate the large-cap universe and close enough elsewhere to be labelled
    as an approximation."""
    try:
        return int(period_end[:4])
    except (TypeError, ValueError):
        return None


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None


__all__ = [
    "ARCHIVE_URL",
    "COMPANY_FACTS_URL",
    "RateLimiter",
    "SECClient",
    "SUBMISSIONS_URL",
    "TICKER_MAP_URL",
    "html_to_text",
    "split_into_sections",
]

# Kept as an alias because the archive URL is the canonical filing permalink.
SEC_TICKER_URL = ARCHIVE_URL
