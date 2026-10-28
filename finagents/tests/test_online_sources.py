"""The live data clients, tested without touching the network.

`sec.py` and `news.py` are the two places where untrusted third-party bytes
become citable evidence, so the parsing is tested against fixtures shaped like
the real payloads. The properties pinned here are the ones that silently corrupt
a financial answer when they break: a stale XBRL figure presented as current, a
filing that chunks to one unretrievable blob, an aggregator presented as if it
were the original publisher, and a currency/encoding error that turns a quote
into unreadable text.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from finagents.data.news import NewsClient
from finagents.data.sec import (
    COMPANY_FACTS_URL,
    SUBMISSIONS_URL,
    SECClient,
    html_to_text,
    split_into_sections,
)
from finagents.schemas import EvidenceType

# --------------------------------------------------------------------------
# Fixtures shaped like real payloads
# --------------------------------------------------------------------------

TICKER_MAP = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "2": {"cik_str": 2488, "ticker": "AMD", "title": "ADVANCED MICRO DEVICES, INC."},
    # Class and preferred shares. EDGAR's map holds 543 non-alphabetic symbols
    # out of 10,431; a `[A-Z]{1,5}` gate rejects every one of them.
    "3": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
    "4": {"cik_str": 1067983, "ticker": "BRK-A", "title": "BERKSHIRE HATHAWAY INC"},
    "5": {"cik_str": 70858, "ticker": "BAC-PB", "title": "Bank of America Corp"},
}

SUBMISSIONS = {
    "cik": "0001045810",
    "name": "NVIDIA CORP",
    "filings": {
        "recent": {
            "accessionNumber": [
                "0001045810-26-000123",
                "0001045810-26-000099",
                "0001045810-26-000050",
            ],
            "filingDate": ["2026-09-03", "2026-08-26", "2026-08-25"],
            "reportDate": ["2026-09-03", "2026-08-02", "2026-07-27"],
            "form": ["8-K", "10-Q", "4"],
            "primaryDocument": ["nvda-20260903.htm", "nvda-20260826.htm", "form4.xml"],
        }
    },
}

# Revenue is deliberately present under three tags, and the tags disagree. The
# freshest tag is the correct answer; picking a preferred tag instead returns a
# figure that is two years stale while looking entirely plausible.
COMPANY_FACTS = {
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        {
                            "end": "2026-01-25",
                            "val": 215_900_000_000,
                            "form": "10-K",
                            "fp": "FY",
                            "filed": "2026-02-25",
                            "fy": 2026,
                        },
                        {
                            "end": "2024-01-28",
                            "val": 60_922_000_000,
                            "form": "10-K",
                            "fp": "FY",
                            "filed": "2024-02-21",
                            "fy": 2024,
                        },
                    ]
                }
            },
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "end": "2023-01-29",
                            "val": 26_974_000_000,
                            "form": "10-K",
                            "fp": "FY",
                            "filed": "2023-02-22",
                            "fy": 2023,
                        },
                    ]
                }
            },
            "GrossProfit": {
                "units": {
                    "USD": [
                        {
                            "end": "2026-01-25",
                            "val": 153_500_000_000,
                            "form": "10-K",
                            "fp": "FY",
                            "filed": "2026-02-25",
                            "fy": 2026,
                        },
                    ]
                }
            },
            "OperatingIncomeLoss": {
                "units": {
                    "USD": [
                        {
                            "end": "2026-01-25",
                            "val": 130_200_000_000,
                            "form": "10-K",
                            "fp": "FY",
                            "filed": "2026-02-25",
                            "fy": 2026,
                        },
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {
                            "end": "2026-01-25",
                            "val": 121_400_000_000,
                            "form": "10-K",
                            "fp": "FY",
                            "filed": "2026-02-25",
                            "fy": 2026,
                        },
                    ]
                }
            },
            "EarningsPerShareDiluted": {
                "units": {
                    "USD/shares": [
                        {
                            "end": "2026-01-25",
                            "val": 4.90,
                            "form": "10-K",
                            "fp": "FY",
                            "filed": "2026-02-25",
                            "fy": 2026,
                        },
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        {
                            "end": "2026-01-25",
                            "val": 206_800_000_000,
                            "form": "10-K",
                            "fp": "FY",
                            "filed": "2026-02-25",
                            "fy": 2026,
                        },
                    ]
                }
            },
        }
    }
}

FILING_HTML = """
<html><body>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
<p>PART I</p>
<p>Item 1. Financial Statements</p>
<p>The accompanying unaudited condensed consolidated financial statements have
been prepared in accordance with accounting principles generally accepted in the
United States of America and reflect the results of operations for the quarter
ended August 2, 2026.</p>
<p>Item 2. Management's Discussion and Analysis of Financial Condition</p>
<p>Revenue increased year over year on higher volumes of data center
accelerators. Gross margin declined sequentially as the mix shifted toward
large configurations with more networking content, and we expect the pace of
orders to remain volatile given customer capital expenditure patterns.</p>
<p>Item 1A. Risk Factors</p>
<p>Demand for accelerated computing products is concentrated in a small number
of hyperscale customers. Export controls restrict sales of advanced products to
certain regions and may be tightened further, and supply depends on third-party
foundries and advanced packaging capacity that we do not control.</p>
</body></html>
"""

NEWS_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item>
  <title>Nvidia sets biggest-ever buyback plan - Reuters</title>
  <link>https://news.google.com/rss/articles/abc123</link>
  <description>Nvidia said it would authorise $150 billion of share repurchases, the
largest such plan for the company, while analysts questioned the durability of
demand as custom silicon competition increased. Shares rose in after-hours
trading following the announcement.</description>
  <pubDate>Mon, 28 Sep 2026 21:10:00 GMT</pubDate>
  <source url="https://www.reuters.com">Reuters</source>
</item>
<item>
  <title>Nvidia announces $150 billion buyback - CNBC</title>
  <link>https://news.google.com/rss/articles/def456</link>
  <description>The chipmaker said it would return capital to shareholders through a
record authorisation even as competition in AI silicon intensifies from custom
accelerator designs built by the largest cloud customers.</description>
  <pubDate>Mon, 28 Sep 2026 22:00:00 GMT</pubDate>
  <source url="https://www.cnbc.com">CNBC</source>
</item>
<item>
  <title>Nvidia - Motley Fool</title>
  <link>https://news.google.com/rss/articles/ghi789</link>
  <description>A short headline with no lead paragraph at all, so there is nothing
verifiable in it.</description>
  <pubDate>Mon, 28 Sep 2026 23:00:00 GMT</pubDate>
  <source url="https://fool.com">Motley Fool</source>
</item>
<item>
  <title>Duplicate story - Reuters</title>
  <link>https://news.google.com/rss/articles/jkl000</link>
  <description>This headline is a repeat of an earlier item and should not be
counted twice in the corpus for the same ticker on the same day.</description>
  <pubDate>Mon, 28 Sep 2026 21:10:00 GMT</pubDate>
  <source url="https://www.reuters.com">Reuters</source>
</item>
</channel></rss>
"""


class _StubClient(SECClient):
    """An SEC client whose HTTP layer is replaced by a fixture table.

    Everything above `_get`/`_get_text` - ticker resolution, filing selection,
    sectioning, XBRL extraction - runs for real.
    """

    def __init__(self, cache_dir=None):
        super().__init__(user_agent="test-agent test@example.com", cache_dir=cache_dir)
        self.json_responses = {
            "company_tickers": TICKER_MAP,
            "submissions:1045810": SUBMISSIONS,
            "companyfacts:1045810": COMPANY_FACTS,
        }
        self.text_responses = {"filing": FILING_HTML}
        self.requested: list = []

    def _get(self, url, cache_key, use_cache=True):
        self.requested.append(url)
        if "company_tickers" in url:
            return self.json_responses["company_tickers"]
        if "submissions" in url:
            return self.json_responses["submissions:1045810"]
        if "companyfacts" in url:
            return self.json_responses["companyfacts:1045810"]
        return None

    def _get_text(self, url, cache_key):
        self.requested.append(url)
        return self.text_responses.get("filing")


@pytest.fixture
def sec(tmp_path):
    return _StubClient(cache_dir=str(tmp_path / "sec"))


# --------------------------------------------------------------------------
# Ticker resolution
# --------------------------------------------------------------------------


class TestTickerResolution:
    def test_resolves_a_known_ticker(self, sec):
        assert sec.cik("NVDA") == 1045810
        assert sec.cik("nvda") == 1045810, "symbols are case-insensitive"

    def test_pads_the_cik_to_ten_digits(self, sec):
        """A 9-digit CIK in a URL silently 404s; the format spec is load-bearing."""
        cik = sec.cik("AMD")
        assert cik == 2488
        assert COMPANY_FACTS_URL.format(cik=cik) == (
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000002488.json"
        )
        assert SUBMISSIONS_URL.format(cik=1045810) == (
            "https://data.sec.gov/submissions/CIK0001045810.json"
        )

    def test_unknown_ticker_returns_none(self, sec):
        assert sec.cik("ZZZZ") is None
        assert sec.is_known_ticker("ZZZZ") is False

    def test_company_name_round_trips_to_a_ticker(self, sec):
        assert sec.ticker_for_company_name("NVIDIA CORP") == "NVDA"
        assert sec.ticker_for_company_name("not a company") is None

    def test_class_and_preferred_shares_resolve(self, sec):
        """`BRK-B` is not an exotic listing; it is the most-held US stock.

        543 of EDGAR's 10,431 symbols contain a class separator. A plain-letters
        rule rejects all of them, so this is a mainstream-company bug, not an
        edge case.
        """
        from finagents.data.providers import OnlineProvider

        provider = OnlineProvider(sec=sec)
        assert provider.resolve_ticker("BRK-B") == "BRK-B"
        assert provider.resolve_ticker("BRK-A") == "BRK-A"
        assert provider.resolve_ticker("BAC-PB") == "BAC-PB"

    def test_share_class_separators_are_normalised(self, sec):
        """People write `BRK.B`; EDGAR writes `BRK-B`."""
        from finagents.data.providers import OnlineProvider

        provider = OnlineProvider(sec=sec)
        for spelling in ("BRK.B", "BRK/B", "BRK B", "brk-b", "$BRK-B", " BRK-A "):
            assert provider.resolve_ticker(spelling) in {"BRK-B", "BRK-A"}, spelling

    def test_a_bare_class_letter_does_not_resolve(self, sec):
        from finagents.data.providers import OnlineProvider

        provider = OnlineProvider(sec=sec)
        assert provider.resolve_ticker("BRK") is None
        assert provider.resolve_ticker("") is None
        assert provider.resolve_ticker("ZZZZ") is None

    def test_public_entry_points_canonicalise_before_fetching(self, sec):
        """A caller writes `BRK.B`; the fetch must use `BRK-B`.

        EDGAR returns None for the CIK of `BRK.B` and Yahoo rejects the dotted
        form, so an unresolved symbol degrades to news-only results and the query
        looks like it succeeded.
        """
        from finagents.data.providers import OnlineProvider

        provider = OnlineProvider(sec=sec, news_per_ticker=0)
        assert provider.canonical_tickers(["BRK.B", "nvda", "BRK-B"]) == [
            "BRK-B",
            "NVDA",
        ]

    def test_a_symbol_with_no_canonical_match_is_kept(self, sec):
        """Yahoo-only tickers should still be priced rather than dropped."""
        from finagents.data.providers import OnlineProvider

        provider = OnlineProvider(sec=sec)
        assert provider.canonical_tickers(["BRK.B", "ZZZZ"]) == ["BRK-B", "ZZZZ"]

    def test_canonical_tickers_defaults_to_the_watchlist(self, sec):
        from finagents.data.providers import OnlineProvider

        provider = OnlineProvider(sec=sec)
        assert provider.canonical_tickers(None) == provider.supported_tickers()


class TestProviderSettings:
    """`build_provider` must be the only place network policy is decided."""

    def test_settings_reach_the_sec_client(self, tmp_path):
        from finagents.data import build_provider
        from finagents.schemas import Settings

        settings = Settings(
            sec_user_agent="Jane Doe jane@example.com",
            cache_dir=str(tmp_path / "cache"),
            filings_per_ticker=7,
            news_per_ticker=3,
            http_timeout_s=5.0,
        )
        provider = build_provider(settings)
        try:
            assert provider.sec.user_agent == "Jane Doe jane@example.com"
            assert provider.sec.cache_dir == tmp_path / "cache" / "sec"
            assert provider.sec.timeout_s == 5.0
            assert provider.filings_per_ticker == 7
            assert provider.news_per_ticker == 3
            assert provider.news._client.timeout.read == 5.0
        finally:
            provider.close()

    def test_the_placeholder_user_agent_is_shared_with_the_data_layer(self):
        """The two defaults cannot be imported across modules; hold them equal."""
        from finagents.data.sec import DEFAULT_SEC_USER_AGENT
        from finagents.schemas import Settings

        assert Settings().sec_user_agent == DEFAULT_SEC_USER_AGENT

    def test_the_placeholder_user_agent_warns(self, tmp_path, caplog):
        from finagents.data.sec import SECClient

        with caplog.at_level("WARNING"):
            SECClient(cache_dir=str(tmp_path)).close()
        assert any("FINAGENTS_SEC_USER_AGENT" in r.message for r in caplog.records)

    def test_a_real_user_agent_does_not_warn(self, tmp_path, caplog):
        from finagents.data.sec import SECClient

        with caplog.at_level("WARNING"):
            SECClient(
                user_agent="Jane Doe jane@example.com", cache_dir=str(tmp_path)
            ).close()
        assert not [
            r for r in caplog.records if "FINAGENTS_SEC_USER_AGENT" in r.message
        ]


# --------------------------------------------------------------------------
# Filings
# --------------------------------------------------------------------------


class TestFilings:
    def test_only_interesting_forms_are_kept(self, sec):
        forms = [f["form"] for f in sec.recent_filings("NVDA", limit=10)]
        assert "10-Q" in forms and "8-K" in forms
        assert "4" not in forms, "insider Form 4s are not research evidence"

    def test_limit_is_respected(self, sec):
        assert len(sec.recent_filings("NVDA", limit=1)) == 1

    def test_filings_are_newest_first(self, sec):
        dates = [f["filing_date"] for f in sec.recent_filings("NVDA", limit=5)]
        assert dates == sorted(dates, reverse=True)

    def test_documents_are_split_at_section_headings(self, sec):
        docs = sec.filing_documents("NVDA", limit=5, max_chunks_per_filing=40)
        sections = {d.metadata.get("section") for d in docs}
        assert any("Risk Factors" in (s or "") for s in sections), sections
        assert any(
            "Discussion" in (s or "") for s in sections
        ), "a fundamentals query must be able to reach the MD&A section"

    def test_risk_factors_text_really_contains_the_risk(self, sec):
        """A section that is split at the wrong place is worse than no split."""
        docs = sec.filing_documents("NVDA", limit=5, max_chunks_per_filing=40)
        risk = next(
            d for d in docs if "Risk Factors" in (d.metadata.get("section") or "")
        )
        assert "export controls" in risk.text.lower()
        assert "Item 2." not in risk.text, "the risk section bled into the next item"

    def test_chunks_of_one_filing_share_a_collection(self, sec):
        docs = sec.filing_documents("NVDA", limit=5, max_chunks_per_filing=40)
        tenq = [d for d in docs if d.metadata.get("form") == "10-Q"]
        collections = {d.metadata["collection"] for d in tenq}
        assert (
            len(collections) == 1
        ), "one filing must be one collection, or the cap cannot work"
        assert {d.metadata["parent_doc_id"] for d in tenq} == collections

    def test_chunk_index_is_sequential(self, sec):
        docs = [
            d
            for d in sec.filing_documents("NVDA", limit=5)
            if d.metadata.get("form") == "10-Q"
        ]
        indices = sorted(d.metadata["chunk_index"] for d in docs)
        assert indices == list(range(len(indices)))

    def test_every_document_cites_edgar_with_a_permalink(self, sec):
        for doc in sec.filing_documents("NVDA", limit=5):
            assert doc.source == "SEC EDGAR"
            assert doc.url and doc.url.startswith(
                "https://www.sec.gov/Archives/edgar/data/"
            )
            assert doc.reliability >= 0.6

    def test_chunk_cap_is_applied(self, sec):
        docs = sec.filing_documents("NVDA", limit=3, max_chunks_per_filing=2)
        tenq = [d for d in docs if d.metadata.get("form") == "10-Q"]
        assert len(tenq) <= 2

    def test_doc_type_reflects_the_form(self, sec):
        docs = sec.filing_documents("NVDA", limit=5)
        eight_k = [d for d in docs if d.metadata.get("form") == "8-K"]
        assert eight_k and all(d.doc_type == EvidenceType.REGULATORY for d in eight_k)


# --------------------------------------------------------------------------
# XBRL facts
# --------------------------------------------------------------------------


class TestSectionBudget:
    """Filing order is not value order, and the budget must not follow it."""

    @staticmethod
    def _filing(sections):
        """Build a 10-Q-like document: a long table section, then MD&A, then risks."""
        table = " ".join(
            f"Total current liabilities {i},{i+1} {i+2},{i+3} Deferred revenue (2) {i} {i+1}"
            for i in range(60)
        )
        parts = ["PART I Item 1. Financial Statements. " + table * 6]
        parts.append(
            "Item 2. Management's Discussion and Analysis of Financial Condition. "
            + "Revenue increased year over year on higher volumes of accelerators. "
            * 40
        )
        parts.append(
            "Item 1A. Risk Factors. "
            + "Demand is concentrated in a small number of hyperscale customers. " * 40
        )
        return " ".join(parts)

    def test_table_fragments_are_dropped(self, sec):
        """Balance-sheet rows are re-derivable from XBRL and crowd out prose."""
        from finagents.data.sec import _prose_density

        assert _prose_density("$ 5,602 $ 3,958 Deferred revenue (2) 1,796 1,193") < 0.3
        assert (
            _prose_density(
                "Revenue increased year over year on higher volumes of accelerators."
            )
            > 0.8
        )

    def test_a_verbose_table_section_cannot_exhaust_the_budget(self, sec):
        docs = sec.filing_documents("NVDA", limit=5, max_chunks_per_filing=8)
        sections = {d.metadata.get("section") for d in docs}
        assert any("Risk Factors" in (s or "") for s in sections), sections
        assert any("Discussion" in (s or "") for s in sections), sections

    def test_each_section_is_capped(self, sec):
        from collections import Counter

        docs = sec.filing_documents("NVDA", limit=5, max_chunks_per_filing=20)
        counts = Counter(d.metadata.get("section") for d in docs)
        assert max(counts.values()) <= 3, counts

    def test_chunk_index_still_increases_through_the_filing(self, sec):
        docs = [
            d
            for d in sec.filing_documents("NVDA", limit=5)
            if d.metadata.get("form") == "10-Q"
        ]
        assert [d.metadata["chunk_index"] for d in docs] == list(range(len(docs)))

    def test_a_filing_with_no_prose_is_still_evidence(self, sec):
        """An all-table exhibit must not be discarded for lacking narrative."""
        from finagents.data.sec import _prioritise_sections

        tables = [("Item 1. Financial Statements", "$ 1 $ 2 $ 3 $ 4 $ 5 $ 6 $ 7 $ 8")]
        kept = _prioritise_sections(tables, per_section=3, total=5)
        assert kept, "a table-only filing is still a citable document"


class TestFinancialFacts:
    def test_picks_the_most_recent_fiscal_year(self, sec):
        """The whole point: three revenue tags, only the freshest is current.

        A preferred-tag rule would return the 2023 `Revenues` figure of
        $27.0bn, or the 2024 one, and present it as the latest year - a wrong
        number that looks entirely reasonable.
        """
        facts = sec.key_financials("NVDA")
        assert facts["revenue"] == pytest.approx(215_900_000_000)

    def test_derived_margins(self, sec):
        """Margins are percentages, matching the renderer's "(%)" unit suffix."""
        facts = sec.key_financials("NVDA")
        assert facts["gross_margin"] == pytest.approx(100 * 153.5 / 215.9, rel=1e-3)
        assert 0.0 < facts["net_margin"] < 100.0

    def test_per_share_values_are_not_scaled(self, sec):
        facts = sec.key_financials("NVDA")
        assert facts["eps_diluted"] == pytest.approx(4.90)

    def test_facts_document_states_the_period_and_provenance(self, sec):
        """A figure without its fiscal year is the easiest way to mislead.

        The corpus must never present a number that the reader could reasonably
        take for the current year when it is not.
        """
        doc = sec.financials_document("NVDA")
        assert doc is not None
        assert "SEC EDGAR XBRL" in doc.source
        assert (
            doc.reliability >= 0.9
        ), "tagged XBRL is the highest-trust document we have"
        assert doc.metadata["period_end"] == "2026-01-25"
        assert doc.metadata["fiscal_year"] == 2026
        assert "fiscal year 2026" in doc.text
        assert "2026-01-25" in doc.text
        assert "215" in doc.text or "216" in doc.text

    def test_facts_document_dated_when_figures_went_public(self, sec):
        """`published_at` is the filing date, not the time this code ran."""
        doc = sec.financials_document("NVDA")
        assert doc.published_at == datetime(2026, 2, 25, tzinfo=doc.published_at.tzinfo)

    def test_conflicting_period_ends_are_disclosed(self, sec):
        """When tags disagree on period, the document has to say so."""
        facts = sec.company_facts("NVDA")
        # Give one tag a different, older period end.
        facts["facts"]["us-gaap"]["Assets"]["units"]["USD"][0]["end"] = "2024-01-28"
        sec.json_responses["companyfacts:1045810"] = facts
        doc = sec.financials_document("NVDA")
        assert doc is not None
        assert "different period ends" in doc.text

    def test_unknown_ticker_yields_no_document(self, sec):
        assert sec.financials_document("ZZZZ") is None
        assert sec.key_financials("ZZZZ") == {}

    def test_missing_facts_do_not_crash(self, sec):
        sec.json_responses["companyfacts:1045810"] = {"facts": {"us-gaap": {}}}
        assert sec.key_financials("NVDA") == {}
        assert sec.financials_document("NVDA") is None


# --------------------------------------------------------------------------
# Sectioning and HTML
# --------------------------------------------------------------------------


class TestSectioning:
    def test_splits_on_item_headings(self):
        text = (
            "Item 1. Financial Statements. " + ("alpha " * 60) + "\n"
            "Item 1A. Risk Factors. " + ("beta " * 60) + "\n"
            "Item 2. Management's Discussion. " + ("gamma " * 60)
        )
        sections = split_into_sections(text, max_chars=2000)
        headings = [h for h, _ in sections]
        assert any("Risk Factors" in h for h in headings)
        assert any("Discussion" in h for h in headings)

    def test_each_section_body_excludes_the_next_heading(self):
        text = "Item 1A. Risk Factors. " + (
            "risk " * 80
        ) + "\n" "Item 2. Management's Discussion. " + ("discussion " * 80)
        sections = split_into_sections(text, max_chars=4000)
        risk = next(body for h, body in sections if "Risk Factors" in h)
        assert "discussion" not in risk.lower()

    def test_long_sections_are_windowed_with_overlap(self):
        body = " ".join(f"word{i}" for i in range(2000))
        sections = split_into_sections(f"Item 1A. Risk Factors. {body}", max_chars=400)
        assert len(sections) > 1
        assert all(len(b) <= 400 for _, b in sections)

    def test_windows_do_not_start_mid_word(self):
        """A chunk opening on "hether the registrant" is a corrupted quote."""
        words = [f"sentence{i}" for i in range(2000)]
        body = " ".join(words)
        sections = split_into_sections(f"Item 1A. Risk Factors. {body}", max_chars=300)
        assert len(sections) > 1
        for _, text in sections[1:]:
            first = text.split(" ", 1)[0]
            assert first in words, f"chunk opens on a fragment: {first!r}"

    def test_text_without_headings_still_retrievable(self):
        """An 8-K has no Items; it must not be discarded for lacking structure."""
        sections = split_into_sections("word " * 2000, max_chars=500)
        assert sections and all(len(b) <= 500 for _, b in sections)

    def test_empty_input(self):
        assert split_into_sections("") == []

    def test_tiny_sections_are_dropped(self):
        assert split_into_sections("Item 1. Financial Statements. x") == []


class TestHtmlToText:
    def test_strips_tags_and_scripts(self):
        html = "<html><body><script>var x=1;</script><p>Revenue rose.</p></body></html>"
        text = html_to_text(html)
        assert "Revenue rose." in text
        assert "<p>" not in text and "var x" not in text

    def test_unwraps_xbrl_spans(self):
        """Inline XBRL wraps text in <ix:> spans; leaving them in loses content."""
        html = (
            "<p><ix:nonFraction>Total revenue was 215.9 billion</ix:nonFraction>.</p>"
        )
        assert "215.9 billion" in html_to_text(html)
        assert "ix:" not in html_to_text(html)

    def test_collapses_whitespace_but_keeps_paragraphs(self):
        """Runs of spaces collapse; block boundaries survive as one blank line."""
        text = html_to_text("<p>a</p>\n\n\n<p>b</p>")
        assert "   " not in text and "\n\n\n" not in text
        assert text.split() == ["a", "b"]

    def test_body_does_not_begin_with_a_stray_period(self):
        """A heading match that eats its own trailing period would leave '. x'."""
        text = html_to_text(
            "<p>PART I</p><p>Item 1A. Risk Factors. "
            + "Export controls may tighten further. " * 20
            + "</p><p>Item 2. Management's Discussion and Analysis of Financial Condition. "
            + "Revenue increased year over year on higher volumes. " * 20
            + "</p>"
        )
        sections = split_into_sections(text)
        assert len(sections) >= 2
        for _, body in sections:
            assert not body.startswith("."), repr(body[:40])

    def test_decodes_entities(self):
        assert "&amp;" not in html_to_text("<p>R&amp;D</p>")
        assert "R&D" in html_to_text("<p>R&amp;D</p>")


# --------------------------------------------------------------------------
# News
# --------------------------------------------------------------------------


class TestNewsParsing:
    @pytest.fixture
    def client(self):
        return NewsClient()

    def test_parses_items_into_documents(self, client):
        docs = client.parse(NEWS_RSS, "NVDA")
        assert docs
        assert all(d.doc_type == EvidenceType.NEWS for d in docs)

    def test_outlet_is_recovered_and_aggregator_is_disclosed(self, client):
        """`source` must not imply first-party reporting by Google."""
        doc = client.parse(NEWS_RSS, "NVDA")[0]
        assert doc.source == "Reuters (via Google News)"
        provenance = doc.metadata["provenance"]
        assert "aggregated" in provenance
        assert "not the full article" in provenance

    def test_reliability_is_weighted_by_outlet(self, client):
        by_outlet = {
            d.metadata["outlet"]: d.reliability for d in client.parse(NEWS_RSS, "NVDA")
        }
        assert by_outlet["Reuters"] > by_outlet["CNBC"]

    def test_each_outlet_is_its_own_collection(self, client):
        """Two stories per outlet, not two stories from the wire in total.

        Without a per-outlet collection the retriever's diversity cap buckets
        every article into one group, so a 12-item feed yields 2 retrievable
        items no matter how many distinct outlets it came from.
        """
        docs = client.parse(NEWS_RSS, "NVDA")
        collections = {d.metadata["collection"] for d in docs}
        outlets = {d.metadata["outlet"] for d in docs}
        assert len(collections) == len(outlets)
        assert all(c.startswith("news-nvda-") for c in collections)

    def test_headline_only_items_are_dropped(self, client):
        """A headline with no lead paragraph carries no verifiable detail."""
        docs = client.parse(NEWS_RSS, "NVDA")
        assert not any(d.metadata["outlet"] == "Motley Fool" for d in docs)

    def test_titles_are_not_duplicated_into_the_body(self, client):
        """The citation prints title and source; a repeated outlet is noise."""
        for doc in client.parse(NEWS_RSS, "NVDA"):
            assert not doc.title.endswith(f" - {doc.metadata['outlet']}")
            assert doc.metadata["outlet"] not in doc.title

    def test_ticker_is_attached(self, client):
        for doc in client.parse(NEWS_RSS, "NVDA"):
            assert doc.metadata["tickers"] == ["NVDA"]
            assert doc.doc_id.startswith("news-nvda-")

    def test_publication_date_is_parsed(self, client):
        doc = client.parse(NEWS_RSS, "NVDA")[0]
        assert isinstance(doc.published_at, datetime)
        assert doc.published_at.year == 2026

    def test_malformed_xml_degrades_to_empty(self, client):
        assert client.parse("<rss><channel><item>", "NVDA") == []
        assert client.parse("", "NVDA") == []

    def test_limit_is_respected(self, client):
        assert len(client.parse(NEWS_RSS, "NVDA", limit=1)) == 1

    def test_news_perspectives_are_declared(self, client):
        doc = client.parse(NEWS_RSS, "NVDA")[0]
        assert "news" in doc.metadata["perspectives"]
        assert "risk" in doc.metadata["perspectives"]


# --------------------------------------------------------------------------
# Cache and rate limiting
# --------------------------------------------------------------------------


class TestCacheAndLimits:
    def test_rate_limiter_enforces_a_minimum_interval(self):
        from finagents.data.sec import RateLimiter

        limiter = RateLimiter(min_interval=0.05)
        import time

        start = time.perf_counter()
        limiter.wait()
        assert time.perf_counter() - start < 0.05, "first call must not sleep"
        limiter.wait()
        assert time.perf_counter() - start >= 0.05

    def test_a_failed_request_does_not_poison_the_cache(self, tmp_path):
        """A transient 500 must not be cached as if it were the filing text."""
        calls = []

        class Flaky(SECClient):
            def _get_text(self, url, cache_key):
                calls.append(url)
                if len(calls) == 1:
                    return None  # first attempt "fails"
                return "<p>Revenue was 215.9 billion.</p>"

        client = Flaky(user_agent="t t@example.com", cache_dir=str(tmp_path / "sec"))
        assert client.filing_text("NVDA", "0001045810-26-000001", "a.htm") is None
        assert client.filing_text("NVDA", "0001045810-26-000001", "a.htm") is not None
        assert len(calls) == 2

    def test_declared_user_agent_is_sent(self, sec):
        assert "test@example.com" in sec._client.headers.get("User-Agent", "")
        """The SEC requires a contact; an anonymous agent gets throttled or blocked."""
