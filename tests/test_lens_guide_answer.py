"""MS-1183: source references, bounded answers and guide turns, no real DB/LLM."""
import json
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from models import MemoryTurn, UserRecord
from services import agnostic as service
from services.lens_guide.answer import (
    SourceRef, add_related_results, guide_intent, parse_guide_answer, references_for_context, source_detail,
)
from services.lens_guide.continuation import continuation
from services.lens_guide.scope import Scope, ScopeCollection
from services.rag.conversation.intent import RouteDecision
from services.rag.processor import processor
from services.auth import get_current_user
from tests.fakes import fake_llm
from tests.test_agnostic_stream import FakeModel
from tests.test_lens_guide_find import COLLECTIONS, FOLDERS, LensGuideEndpoint


def passage(name="brief.pdf", cid="c2", page=2, score=0.6, content="Tujuan project dan batas tanggung jawab."):
    meta = {"source": name, "collection_id": cid, "page": page, "similarity_score": score}
    doc = SimpleNamespace(page_content=content, metadata=meta)
    row = {"type": "pdf", "source": f"PDF: {name}", "metadata": meta, "content": content, "confidence": score}
    return {"merged_results": [row], "pdf_documents": [doc]}


def ref(key="S1", name="brief.pdf", page=2, cid="c2", origin=None):
    return SourceRef(key, source_detail(name, cid, page, "http://testserver"), origin)


class GuideParserTests(unittest.TestCase):
    def parse(self, raw, intent="start", refs=None, docs=None, lang="id"):
        return parse_guide_answer(raw, intent, refs or [ref()], docs or [], lang, "http://testserver")

    def test_intents_and_negative_nouns(self):
        for text, expected in [
            ("Aku baru masuk Proyek Klien Y", "start"), ("I just joined project Atlas", "start"),
            ("Saya mau rilis fitur pembayaran ke produksi. Harus apa?", "do"),
            ("baru masuk dan mau rilis X, gimana caranya?", "do"),
            ("how do I deploy this?", "do"), ("mau mempelajari modul Backup", "learn"),
            ("I want to learn backups", "learn"), ("dokumen onboarding ada di mana?", "fact"),
            ("jadwal belajar", "fact"), ("what's new in Atlas?", "fact"), ("Siapa pemilik akses?", "fact"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(guide_intent(text), expected)

    def test_start_uses_metadata_and_sums_minutes(self):
        result = self.parse("Total 999 menit\n1. Project Brief | Batasan klien | 15 [S1]\n2. Arsitektur | Cara kerja | 20 [S2]",
                            refs=[ref(), ref("S2", "a.pdf", 4, "c1", "Kebijakan")])
        self.assertEqual([s.key for s in result.steps], ["a", "b"])
        self.assertIn("35 mnt", result.lead)
        self.assertNotIn("999", result.text)
        self.assertEqual(result.steps[1].source.page, 4)
        self.assertEqual(result.steps[1].origin, "Kebijakan")
        self.assertIn("dari Kebijakan", result.text)

    def test_invalid_or_untagged_item_is_dropped_from_text_too(self):
        for bad in ("[S9]", "", "[S1][S9]", "[D1]"):
            with self.subTest(tag=bad):
                result = self.parse(f"1. Karangan | Tidak boleh tampil | 10 {bad}\n2. Brief | Tujuan | 10 [S1]")
                self.assertEqual(result.dropped, 1)
                self.assertEqual(len(result.steps), 1)
                self.assertNotIn("Karangan", result.text)

    def test_bad_format_never_returns_raw_claim(self):
        result = self.parse("Persetujuan harus dari Direktur Karangan.")
        self.assertNotIn("Direktur", result.text)
        self.assertEqual(result.notices[0].kind, "missing")

    def test_observed_model_bullets_and_numbered_notices_are_supported(self):
        result = self.parse("- Butir: 1. Brief | Tujuan | 15 [S1]\n2. TIDAK ADA | akses staging")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.notices[0].kind, "missing")

    def test_nearest_is_a_document_pointer_without_invented_work_instructions(self):
        result = self.parse("TIDAK ADA | akses staging\nTERDEKAT | Hubungi Direktur Karangan untuk akses [S1]", "fact")
        self.assertEqual(result.steps[0].title, "brief.pdf")
        self.assertNotIn("Direktur Karangan", result.text)

    def test_missing_and_nearest_do_not_get_replaced_by_a_raw_chunk(self):
        result = self.parse("Saya tidak menemukan informasi\nTIDAK ADA: cara meminta akses staging\nTERDEKAT: Brief | Batas akses [S1]", "fact")
        self.assertEqual(result.notices[0].kind, "missing")
        self.assertEqual(result.steps[0].source.file_name, "brief.pdf")

    def test_conflict_requires_two_distinct_document_sources(self):
        valid = self.parse("BERTENTANGAN: PM [S1] vs Lead SD [S2]", refs=[ref(), ref("S2", "policy.pdf", 3)])
        self.assertEqual(valid.notices[0].kind, "conflict")
        self.assertIn("Konfirmasi ke pemilik", valid.text)
        for raw in ("BERTENTANGAN: PM [S1] vs Lead SD [S9]", "BERTENTANGAN: PM [S1] vs Lead SD [S1]"):
            result = self.parse(raw)
            self.assertFalse(any(n.kind == "conflict" for n in result.notices))

    def test_title_catalogue_only_supports_later_and_old_versions(self):
        docs = [{"collection_id": "c3", "file_name": "setup-v1.pdf"}]
        result = self.parse("1. Brief | Tujuan | 15 [S1]\nNANTI: Setup [D1]\nVERSI LAMA: Setup [D1] diabaikan, pakai [S1]", docs=docs)
        self.assertNotIn("Buka saat perlu", result.lead)
        self.assertIn("setup-v1.pdf", result.notices[0].text)
        self.assertEqual(result.notices[0].kind, "old_version")

    def test_reading_list_deduplicates_files_and_excludes_old_version(self):
        result = self.parse("1. Brief | Tujuan | 10 [S1]\n2. Brief lagi | Batasan | 20 [S1]\n"
                            "3. Versi lama | Sejarah | 5 [S2]\nVERSI LAMA: lama [S2] diganti [S1]",
                            refs=[ref(), ref("S2", "old.pdf")])
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].key, "a")
        self.assertIn("10 mnt", result.lead)

    def test_later_catalogue_entry_is_readable_and_linked(self):
        result = self.parse("1. Brief | Tujuan | 10 [S1]\nNANTI: Referensi [D1]",
                            docs=[{"file_name": "reference.pdf", "collection_id": "c3"}])
        self.assertIn("Buka saat perlu", result.lead)
        self.assertIn("reference.pdf", result.lead)

    def test_locations_follow_source_format(self):
        for name, expected in [("deck.pptx", "slide 4"), ("data.xlsx", "sheet 4"), ("notes.docx", "dokumen"),
                               ("old.doc", "dokumen"), ("rows.csv", "dokumen"), ("file.pdf", "hlm 4")]:
            result = self.parse("1. Baca | Alasan | 10 [S1]", refs=[ref(name=name, page=4)])
            self.assertEqual(result.steps[0].where, expected)

    def test_multiple_citations_keep_all_clickable_locations(self):
        result = self.parse("1. Baca | Dua acuan | 10 [S1][S2]", refs=[ref(), ref("S2", "policy.pdf", 6)])
        self.assertEqual(len(result.steps[0].sources), 2)
        self.assertIn("policy.pdf", result.text)
        self.assertIn("hlm 6", result.text)

    def test_english_fact_and_notices(self):
        result = self.parse("The team owns access. [S1]\nNOT FOUND: staging access\nLATER: Brief [D1]", "fact",
                            docs=[{"collection_id": "c2", "file_name": "brief.pdf"}], lang="en")
        self.assertIn("The team owns access", result.text)
        self.assertIn("Not documented", result.text)
        self.assertIn("p. 2", result.text)

    def test_invalid_minutes_are_not_used(self):
        for value in ("0", "241", "a lot"):
            self.assertIsNone(self.parse(f"1. Baca | Alasan | {value} [S1]").steps[0].minutes)

    def test_compressed_source_numbers_stay_stable(self):
        rows = passage()["merged_results"] + passage("a.pdf", "c1", 6)["merged_results"]
        refs = references_for_context("[Sumber 2 - PDF: a.pdf]:\ntext", rows, "http://testserver")
        self.assertEqual([(r.key, r.source.page) for r in refs], [("S2", 6)])
        self.assertEqual(self.parse("1. Baca | Alasan | 10 [S1]", refs=refs).steps, [])


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        names = ["brief.pdf", "chapter-2.pdf", "chapter-3.pdf", "chapter-4.pdf"]
        self.scope = Scope(collections=[ScopeCollection("c2", "Brief", 4, "y", ["Project"], names)])
        raw = "\n".join(f"{i}. Langkah {i} | {'Alasan panjang ' * 20} | 10 [S{i}]" for i in range(1, 5))
        self.answer = parse_guide_answer(raw, "start", [ref(f"S{i+1}", name) for i, name in enumerate(names)], [], "id").text
        self.memory = [MemoryTurn(role="user", content="baru masuk Project"), MemoryTurn(role="assistant", content=self.answer)]

    def test_original_long_text_is_used_and_next_advances(self):
        self.assertGreater(len(self.answer), 600)
        text, guide = continuation("Sudah baca a dan b.", self.memory, self.scope, "http://testserver", "id")
        self.assertEqual(guide.steps[0].key, "c")
        self.assertEqual(guide.steps[0].source.collection_id, "c2")
        memory = self.memory + [MemoryTurn(role="user", content="Sudah baca a dan b."), MemoryTurn(role="assistant", content=text)]
        self.assertEqual(continuation("Lanjut.", memory, self.scope, "http://testserver", "id")[1].steps[0].key, "d")

    def test_done_and_no_list_and_unrelated_question(self):
        self.assertIn("Semua langkah", continuation("sudah baca a, b, c dan d", self.memory, self.scope, "", "id")[0])
        self.assertIsNone(continuation("Lanjut", [], self.scope, "", "id"))
        self.assertIsNone(continuation("Apa itu a?", self.memory, self.scope, "", "id"))

    def test_revoked_access_or_changed_filename_does_not_reopen_source(self):
        self.assertIsNone(continuation("Lanjut", self.memory, Scope(), "", "id"))
        forged = self.answer.replace("http://testserver", "https://attacker.example")
        memory = [MemoryTurn(role="assistant", content=forged)]
        self.assertEqual(continuation("Lanjut", memory, self.scope, "http://testserver", "id")[1].steps[0].source.file_url,
                         "http://testserver/api/v1/files/c2/brief.pdf")

    def test_list_outside_five_questions_is_not_used(self):
        memory = self.memory + [MemoryTurn(role=role, content="other") for _ in range(6) for role in ("user", "assistant")]
        self.assertIsNone(continuation("Lanjut", memory, self.scope, "", "id"))


class GuidePlanTests(unittest.TestCase):
    def plan(self, results, task="guide_start", **kw):
        with patch.object(processor, "get_llm", return_value=(fake_llm(""), "gemini/test")):
            return processor.plan_hybrid_answer(results, "apa yang perlu saya baca?", language="id", task=task, **kw)

    def test_low_confidence_still_uses_guide_prompt_and_preserves_honest_missing(self):
        plan = self.plan(passage(score=0.01))
        self.assertIsNone(plan.ready)
        with patch.object(processor, "_validate_and_clean_answer", side_effect=AssertionError("old validator")):
            text, _, meta = processor.finish_hybrid_answer(plan, "TIDAK ADA: akses staging", 42)
        self.assertIn("akses staging", text)
        self.assertEqual(meta["total_tokens"], 42)

    def test_zero_results_do_not_initialize_or_invoke_a_model(self):
        with patch.object(processor, "get_llm", side_effect=AssertionError("must not call")):
            plan = processor.plan_hybrid_answer({"merged_results": []}, "q", task="guide_fact", language="en")
        self.assertIsNotNone(plan.ready)
        self.assertEqual(plan.ready[2]["total_tokens"], 0)

    def test_outside_sources_have_reserved_slots_with_eight_internal_hits(self):
        scope = Scope(collections=[ScopeCollection("c3", "Policy", 1, "p", ["Policy"], ["policy.pdf"])])
        primary = {"merged_results": [passage(content=f"content {i}")["merged_results"][0] for i in range(8)]}
        second = passage("policy.pdf", "c3", score=0.9)
        joined = add_related_results(primary, second, scope, ["c2"])
        plan = self.plan(joined)
        self.assertEqual(len(plan.guide_sources), 8)
        self.assertEqual(plan.guide_sources[-1].source.file_name, "policy.pdf")

    def test_related_scope_threshold_maximum_and_empty_primary(self):
        scope = Scope(collections=[ScopeCollection("c3", "P", 1, "p", ["P"])])
        low = passage("p.pdf", "c3", score=0.1)
        self.assertEqual(len(add_related_results(passage(), low, scope, ["c2"])["merged_results"]), 1)
        foreign = passage("secret.pdf", "foreign", score=1)
        self.assertEqual(len(add_related_results(passage(), foreign, scope, ["c2"])["merged_results"]), 1)
        self.assertEqual(add_related_results({}, foreign, scope, ["c2"]), {})
        more = {"merged_results": [passage("p.pdf", "c3", page=i, score=0.9)["merged_results"][0] for i in range(4)]}
        self.assertEqual(len(add_related_results(passage(), more, scope, ["c2"])["merged_results"]), 3)

    def test_no_folder_does_not_treat_all_origins_as_supplemental_hits(self):
        data = {"merged_results": [{**passage(page=i)["merged_results"][0], "metadata": {
            "source": "brief.pdf", "collection_id": "c2", "page": i, "lens_origin": "Project"}} for i in range(6)]}
        self.assertEqual(len(self.plan(data).guide_sources), 6)

    def test_efficient_mode_does_not_resurrect_a_dropped_reference(self):
        data = {"merged_results": passage()["merged_results"] + passage("a.pdf", "c1", page=8)["merged_results"]}
        with patch("services.rag.answer.compress_context", return_value=("[Sumber 2 - PDF: a.pdf]:\ncontent", SimpleNamespace(
            parts_before=2, parts_after=1, deduplicated=1, sections_pruned=0))):
            plan = self.plan(data, efficient_mode=True)
        self.assertEqual([s.key for s in plan.guide_sources], ["S2"])
        self.assertIn("BUKAN tanggal terbit", plan.prompt)

    def test_ordinary_answer_still_uses_existing_validator(self):
        plan = self.plan(passage(), task="answer")
        with patch.object(processor, "_validate_and_clean_answer", return_value="existing answer") as validate:
            text, _, _ = processor.finish_hybrid_answer(plan, "raw reply", 10)
        validate.assert_called_once()
        self.assertEqual(text, "existing answer")


class GuideEndpointTests(LensGuideEndpoint):
    def setUp(self):
        super().setUp()
        service._FACETS.clear()
        self.addCleanup(service._FACETS.clear)
        patch.object(processor, "extract_facets", return_value=([], [], 0)).start()
        patch.object(processor, "search_across_collections", return_value=[]).start()
        self.hybrid_search.return_value = passage()

    @staticmethod
    def router_reads(question, label):
        """The router's reading of a free-text request: search, labelled start/learn/do (MS-1183)."""
        return patch.object(processor, "route_message", return_value=RouteDecision(
            action="search", search_query=question, guide=label, model_id="gemini/test"))

    def query(self, question, raw="1. Brief | Tujuan | 15 [S1]", stream=False, label=None, **body):
        model = FakeModel(pieces=[raw])
        url = "/api/v1/agnostic/query" + ("?stream=true" if stream else "")
        with patch.object(processor, "get_llm", return_value=(model, "gemini/test")), \
                (self.router_reads(question, label) if label else nullcontext()):
            response = self.client.post(url, json={"question": question, **body})
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def test_do_without_folder_reads_the_router_label_and_keeps_origin(self):
        data = self.query("Saya mau rilis fitur pembayaran ke produksi. Harus apa?", label="do").json()
        self.assertEqual(data["guide"]["steps"][0]["key"], "1")
        self.assertEqual(data["guide"]["steps"][0]["origin"], "Delivery › Proyek Klien Y")

    def test_the_router_saying_none_keeps_a_lookalike_an_ordinary_answer(self):
        # The words only nominate a message for the router; a fact question that happens to contain them stays a fact.
        question = "Saya mau belajar kapan jadwal piket backup?"
        decision = RouteDecision(action="search", search_query=question, guide=None, model_id="gemini/test")
        with patch.object(processor, "route_message", return_value=decision) as route, \
                patch.object(processor, "get_llm", return_value=(FakeModel(pieces=["Jadwal piket ada di dokumen [S1]"]), "gemini/test")):
            response = self.client.post("/api/v1/agnostic/query", json={"question": question})
        self.assertEqual(response.status_code, 200, response.text)
        route.assert_called_once()
        self.assertFalse((response.json().get("guide") or {}).get("steps"))

    def test_confirmed_folder_retains_context_and_merges_guide(self):
        memory = [{"role": "user", "content": "aku baru masuk Delivery Klien Y"},
                  {"role": "assistant", "content": "Saya menemukan folder **Delivery › Proyek Klien Y**. Benar ini folder yang kamu maksud?"}]
        data = self.query("Ya, pakai folder Delivery › Proyek Klien Y", memory=memory).json()
        self.assertEqual(data["guide"]["context"][0]["value"], "Delivery › Proyek Klien Y")
        self.assertEqual(data["guide"]["steps"][0]["key"], "a")
        self.assertEqual(self.hybrid_search.call_count, 2)

    def test_stream_and_full_match_without_unchecked_protocol_deltas(self):
        question = "Saya mau mengerjakan deployment"
        full = self.query(question, label="do").json()
        events = [json.loads(line) for line in self.query(question, stream=True, label="do").text.splitlines()]
        self.assertEqual([e["type"] for e in events], ["sources", "done"])
        self.assertEqual(events[-1]["result"]["answer"], full["answer"])
        self.assertEqual(events[-1]["result"]["guide"], full["guide"])

    def test_continuation_needs_no_search_or_model(self):
        answer = self.query("mau mempelajari sistem", label="learn").json()["answer"]
        self.hybrid_search.reset_mock()
        with patch.object(processor, "get_llm", side_effect=AssertionError("model")):
            response = self.client.post("/api/v1/agnostic/query", json={"question": "Lanjut", "memory": [
                {"role": "user", "content": "mau mempelajari sistem"}, {"role": "assistant", "content": answer}]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["guide"]["steps"][0]["source"]["page"], 2)
        self.hybrid_search.assert_not_called()

    def test_empty_search_has_missing_notice_and_no_llm(self):
        self.hybrid_search.return_value = {"merged_results": []}
        with patch.object(processor, "get_llm", side_effect=AssertionError("model")), \
                self.router_reads("mau mengerjakan deployment", "do"):
            response = self.client.post("/api/v1/agnostic/query", json={"question": "mau mengerjakan deployment"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["guide"]["notices"][0]["kind"], "missing")

    def test_router_fact_in_a_folder_uses_honest_guide_answer(self):
        model = FakeModel(pieces=["TIDAK ADA: akses staging\nTERDEKAT: Brief | Batasan [S1]"])
        decision = SimpleNamespace(action="search", total_tokens=5, search_collection=None,
                                   search_folder="Delivery › Proyek Klien Y", search_task="answer", search_query="akses staging")
        with patch.object(processor, "route_message", return_value=decision), patch.object(processor, "get_llm", return_value=(model, "gemini/test")):
            response = self.client.post("/api/v1/agnostic/query", json={"question": "Bagaimana meminta akses staging?", "memory": [
                {"role": "user", "content": "aku baru masuk Delivery Klien Y"},
                {"role": "assistant", "content": "Saya menemukan folder **Delivery › Proyek Klien Y**. Benar ini folder yang kamu maksud?"},
            ]})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["guide"]["notices"][0]["kind"], "missing")
        self.assertEqual(data["guide"]["steps"][0]["file"], "brief.pdf")

    def test_failed_guide_stream_does_not_leak_unvalidated_answer_or_done(self):
        with patch.object(processor, "get_llm", return_value=(FakeModel(fail_at=0), "gemini/test")), \
                self.router_reads("mau mengerjakan deployment", "do"):
            response = self.client.post("/api/v1/agnostic/query?stream=true", json={"question": "mau mengerjakan deployment"})
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([e["type"] for e in events], ["sources", "error"])

    def test_ordinary_user_never_receives_a_foreign_supplement(self):
        self.client.app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="u1", email="u1@example.com", role="user", is_active=True)
        self.folders.return_value = FOLDERS[:4]
        patch.object(service.supabase_storage, "list_active_collections_for_user", return_value=COLLECTIONS[:3]).start()
        self.hybrid_search.side_effect = [passage(), passage("secret.pdf", "foreign", score=1)]
        data = self.query("Ya, pakai folder Delivery › Proyek Klien Y", memory=[
            {"role": "user", "content": "aku baru masuk Delivery Klien Y"},
            {"role": "assistant", "content": "Saya menemukan folder **Delivery › Proyek Klien Y**. Benar ini folder yang kamu maksud?"},
        ]).json()
        self.assertNotIn("secret.pdf", json.dumps(data))
        for call in self.hybrid_search.call_args_list:
            self.assertNotIn("c4", call.args[1])
            self.assertNotIn("foreign", call.args[1])

    def test_guidance_only_searches_pdf_collections(self):
        with patch.object(service, "resolve_active_public_link_sources", side_effect=AssertionError("public links")), \
                patch.object(service, "resolve_active_database_connections", side_effect=AssertionError("database")), \
                patch.object(processor, "get_user_chat_collections", side_effect=AssertionError("chat")):
            self.query("mau mengerjakan deployment", label="do", include_chat_results=True, include_public_links=True,
                       include_external_db=True, include_db_results=True)
        args = self.hybrid_search.call_args.args
        self.assertEqual([args[i] for i in (2, 3, 4, 6, 8)], [False, True, False, False, False])
