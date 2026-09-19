"""Offline regression checks; no provider calls or production-store writes."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import httpx
import datasets
from openai import RateLimitError
from qdrant_client import QdrantClient
from src import crag, pipeline, qdrant_store, retrieve, data_hotpot, comparison
from src.agnes_client import chat, count_requests, ModelRequestError

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for module in (crag, pipeline, data_hotpot, comparison):
            p = patch.object(module, "CACHE_DIR", self.root)
            p.start()
            self.addCleanup(p.stop)

    def test_json_braces_fences_and_invalid_preamble(self):
        self.assertEqual(crag.extract_json_object('junk {invalid} ```json\n{"why":"brace } inside", "score":0.5}\n```')["score"], .5)
        with self.assertRaises(ValueError):
            crag.extract_json_object("none")

    def test_scores_and_boundaries(self):
        for bad in (-1, 2, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                crag.decide_action([bad])
        for scores, expected in (([], "Incorrect"), ([.3], "Incorrect"), ([.7], "Correct"), ([.5], "Ambiguous")):
            self.assertEqual(crag.decide_action(scores), expected)

    def test_bad_evaluation_not_cached_then_good_cached(self):
        doc = {"title": "T", "text": "Source."}
        with patch.object(crag, "chat", side_effect=['{"score":2,"label":"relevant","why":"x"}', '{"score":0.8,"label":"relevant","why":"support"}']) as call:
            with self.assertRaises(ValueError):
                crag.evaluate_documents("Q", [doc])
            self.assertFalse(list(self.root.rglob("*.json")))
            self.assertEqual(crag.evaluate_documents("Q", [doc])[0]["score"], .8)
            crag.evaluate_documents("Q", [doc])
            self.assertEqual(call.call_count, 2)

    def test_strip_ids_and_no_invented_text(self):
        with patch.object(crag, "chat", return_value='{"kept_ids":[99]}'):
            with self.assertRaises(ValueError):
                crag.refine_strips("Q", "Yes. A longer source sentence.")
        with patch.object(crag, "chat", return_value='{"kept_ids":[1]}'):
            self.assertEqual(crag.refine_strips("Q", "Yes. A longer source sentence."), ["Yes."])

    def test_generation_failure_and_empty_abstention(self):
        with patch.object(crag, "chat", side_effect=ModelRequestError("failure")) as call:
            self.assertEqual(crag.generate_answer("Q", []), crag.ABSTENTION)
            call.assert_not_called()
            with self.assertRaises(ModelRequestError):
                crag.generate_answer("Q", ["Source."])
            self.assertFalse(list(self.root.rglob("*.json")))

    def test_kept_titles_require_kept_strips(self):
        doc = {"title": "T", "text": "Text."}
        with patch.object(pipeline, "evaluate_documents", return_value=[{"score": .9, "label": "relevant", "why": "x", "doc": doc}]), patch.object(pipeline, "refine_strips", return_value=[]):
            result = pipeline.run_crag("Q", docs=[doc])
        self.assertEqual(result.kept_titles, [])
        self.assertEqual(result.n_llm_calls, 0)

    def test_literal_substring_direction(self):
        self.assertFalse(pipeline.answer_contains_gold("", "reference"))
        self.assertFalse(pipeline.answer_contains_gold("short", "shorter"))
        self.assertFalse(pipeline.answer_contains_gold("anything", ""))
        self.assertTrue(pipeline.answer_contains_gold("not supported", "no"))

    def test_title_recall_is_not_any_gold_hit_rate(self):
        record = {"id":"test", "question":"Synthetic", "gold_answer":"A", "gold_titles":["T1", "T2"]}
        result = pipeline.CRAGResult(hits=[{"title":"T1"}], evaluations=[], answer="A", action="Correct", strips=["A"], kept_titles=["T1"], n_llm_calls=0)
        row = pipeline._row(record, result)
        self.assertTrue(row["gold_in_hits"])
        self.assertEqual(row["gold_title_recall_at_k"], .5)

    def test_naive_rag_uses_full_ranked_evidence_without_correction(self):
        docs = [{"title":"First", "text":"Full first paragraph."}, {"title":"Second", "text":"Full second paragraph."}]
        with patch.object(comparison, "generate_answer", return_value="Answer") as generate:
            result = comparison.run_naive_rag("Question", docs)
        self.assertEqual(result["evidence"], ["Full first paragraph.", "Full second paragraph."])
        self.assertEqual(result["evidence_titles"], ["First", "Second"])
        generate.assert_called_once_with("Question", result["evidence"], model="agnes-3.0-flash")

    def test_comparison_retrieves_once_and_shares_hits(self):
        records = [{"id":str(i), "question":"Q", "gold_answer":"A", "gold_titles":["T"], "question_type":"bridge"} for i in range(50)]
        hit = {"title":"T", "text":"Evidence.", "score":.8, "payload":{"is_gold":True}}
        crag_result = pipeline.CRAGResult(answer="A", action="Correct", evaluations=[], kept_titles=["T"], strips=["Evidence."], n_llm_calls=0)
        with patch.object(comparison, "search", return_value=[hit]) as search_call, patch.object(comparison, "run_naive_rag", return_value={"answer":"A", "evidence":["Evidence."], "evidence_titles":["T"], "n_llm_calls":0}) as naive_call, patch.object(comparison, "run_crag", return_value=crag_result) as crag_call:
            rows = comparison.run_comparison_50(records, max_retries=1)
        self.assertEqual(search_call.call_count, 50)
        self.assertEqual(naive_call.call_args_list[0].args[1], [hit])
        self.assertEqual(crag_call.call_args_list[0].kwargs["docs"], [hit])
        self.assertEqual(len(rows), 50)

    def test_manual_reviews_are_bound_to_exact_outputs(self):
        row = {"id":"x", "question":"Q", "gold_answer":"A", "hits":[{"title":"T", "text":"E"}],
               "naive":{"answer":"A", "evidence":["E"]}, "crag":{"answer":"A", "strips":["E"]}}
        row["comparison_fingerprint"] = comparison.comparison_fingerprint(row)
        review = {"id":"x", "comparison_fingerprint":row["comparison_fingerprint"],
                  "naive_correctness":"correct", "naive_support":"fully_supported",
                  "crag_correctness":"correct", "crag_support":"fully_supported",
                  "preference":"tie_both_good", "error_types":["none"], "rationale":"Both are supported."}
        path = self.root / "reviews.json"
        path.write_text(json.dumps({"reviews":[review]}), encoding="utf-8")
        valid, stale = comparison.load_manual_reviews([row], path)
        self.assertEqual(len(valid), 1)
        self.assertEqual(stale, [])
        row["crag"]["answer"] = "Changed"
        valid, stale = comparison.load_manual_reviews([row], path)
        self.assertEqual(valid, [])
        self.assertEqual(stale, ["x"])

    def test_429_retry_counts_actual_attempts(self):
        response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid"))
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="PONG"), finish_reason="stop")])
        client = Mock()
        client.chat.completions.create.side_effect = [RateLimitError("redacted", response=response, body=None), completion]
        with patch("src.agnes_client.time.sleep"), count_requests() as count:
            self.assertEqual(chat([{"role":"user","content":"ping"}], client=client), "PONG")
        self.assertEqual(count["attempts"], 2)

    def test_benchmark_resume_and_settings_invalidation(self):
        records = [{"id": str(i), "question": "Q", "gold_answer": "A", "gold_titles": ["T"], "context_paragraphs": []} for i in range(2)]
        result = pipeline.CRAGResult(hits=[{"title":"T"}], evaluations=[], answer="A", action="Correct", strips=["A"], kept_titles=["T"], n_llm_calls=1)
        with patch.object(pipeline, "run_crag", side_effect=[result, ValueError("bad")]):
            with self.assertRaises(RuntimeError):
                pipeline.run_slice_50_benchmark(records, max_retries=1)
        with patch.object(pipeline, "run_crag", return_value=result) as call:
            rows, metrics, _ = pipeline.run_slice_50_benchmark(records, max_retries=1)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(rows[0]["n_llm_calls"], 0)
            self.assertEqual(metrics["gold_title_recall@k"], 1)
            pipeline.run_slice_50_benchmark(records, k=4, max_retries=1)
            self.assertEqual(call.call_count, 3)

    def test_smoke_never_uses_ids_outside_slice(self):
        records = [{"id": str(i), "question_type": "bridge"} for i in range(8)]
        ids = data_hotpot._smoke(records, self.root / "smoke.json")
        self.assertEqual(len(ids), 8)
        self.assertEqual(set(ids), {r["id"] for r in records})

    def test_seed_provenance_and_unknown_labels(self):
        fixture = datasets.Dataset.from_list([{"id":str(i), "question":"Synthetic test question", "answer":"Test answer", "context":{"title":["Test title"], "sentences":[["Test sentence."]]}, "type":"bridge"} for i in range(4)])
        with patch.object(data_hotpot, "download_or_load", return_value=fixture) as load:
            first, ids = data_hotpot.build_slice(n=2, seed=42)
            self.assertTrue(all(r["context_paragraphs"][0]["is_gold"] is None for r in first))
            data_hotpot.build_slice(n=2, seed=42)
            self.assertEqual(load.call_count, 1)
            data_hotpot.build_slice(n=2, seed=43)
            self.assertEqual(load.call_count, 2)

    def test_benchmark_counts_failed_attempt_before_retry(self):
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")])
        client = Mock()
        client.chat.completions.create.return_value = completion
        attempts = []
        def run(*args, **kwargs):
            with count_requests():
                chat([{"role":"user", "content":"test"}], client=client)
                attempts.append(1)
                if len(attempts) == 1:
                    raise ValueError("invalid schema")
                return pipeline.CRAGResult(hits=[], evaluations=[], answer=crag.ABSTENTION, action="Incorrect", strips=[], kept_titles=[], n_llm_calls=1)
        record = {"id":"test", "question":"Synthetic", "gold_answer":"A", "gold_titles":["T"], "context_paragraphs":[]}
        with patch.object(pipeline, "run_crag", side_effect=run), patch.object(pipeline.time, "sleep"):
            rows, metrics, _ = pipeline.run_slice_50_benchmark([record], max_retries=2)
        self.assertEqual(rows[0]["n_llm_calls"], 2)

    def test_evaluator_prompt_excludes_gold_metadata(self):
        doc = {"title":"T", "text":"Source.", "payload":{"is_gold":True, "gold_answer":"DO_NOT_SEND"}}
        with patch.object(crag, "chat", return_value='{"score":0.5,"label":"relevant","why":"support"}') as call:
            crag.evaluate_documents("Q", [doc])
            messages = json.dumps(call.call_args.args[0])
        self.assertNotIn("DO_NOT_SEND", messages)
        self.assertNotIn("is_gold", messages)

    def test_optional_web_failure_never_fabricates_evidence(self):
        from src.search import ExternalWebSearcher
        with patch("src.search.DDGS") as provider:
            provider.return_value.text.side_effect = OSError("offline")
            with self.assertRaisesRegex(RuntimeError, "no substitute evidence"):
                ExternalWebSearcher().search_and_extract("Synthetic test question")

    def test_model_cannot_be_overridden(self):
        with self.assertRaises(ValueError):
            chat([], model="other-model", client=Mock())
        with self.assertRaises(ValueError):
            crag.evaluate_documents("Q", [], model="other-model")

    def test_index_change_missing_encoder_and_shrink(self):
        encoder_path = self.root / "encoder.joblib"
        manifest_path = self.root / "index.json"
        records = [{"id":"q", "context_paragraphs":[{"title":str(i),"text":text,"is_gold":i == 0} for i, text in enumerate(("Alpha apple fruit.", "Beta banana yellow.", "Gamma grapes green."))]}]
        client = QdrantClient(":memory:")
        self.addCleanup(client.close)
        # Explicit paths avoid default-argument binding to production paths.
        original_save, original_load = qdrant_store.TFIDFSVDEncoder.save, qdrant_store.TFIDFSVDEncoder.load
        with patch.object(qdrant_store, "ENCODER_CACHE_PATH", encoder_path), patch.object(qdrant_store, "INDEX_MANIFEST", manifest_path), patch.object(qdrant_store.TFIDFSVDEncoder, "save", lambda obj, path=None: original_save(obj, encoder_path)), patch.object(qdrant_store.TFIDFSVDEncoder, "load", lambda path=None: original_load(encoder_path)):
            self.assertEqual(qdrant_store.index_slice(records, client=client), 3)
            with patch.object(retrieve, "ENCODER_CACHE_PATH", encoder_path), patch.object(retrieve, "_cached_encoder", None), patch.object(retrieve, "_encoder_stamp", None):
                self.assertEqual(len(retrieve.search("fruit", 3, "q", client=client)), 3)
                self.assertEqual(retrieve.search("fruit", 3, "missing", client=client), [])
                self.assertEqual(len(retrieve.search("fruit", 3, client=client)), 3)
            with patch.object(client, "upsert", wraps=client.upsert) as call:
                qdrant_store.index_slice(records, client=client)
                call.assert_not_called()
            encoder_path.unlink()
            qdrant_store.index_slice(records, client=client)
            self.assertTrue(encoder_path.exists())
            records[0]["context_paragraphs"][0]["text"] = "Updated apples red fruit."
            qdrant_store.index_slice(records, client=client)
            point = client.retrieve("hotpot_slice", [0])[0]
            self.assertEqual(point.payload["text"], "Updated apples red fruit.")
            records[0]["context_paragraphs"].pop()
            qdrant_store.index_slice(records, client=client)
            self.assertEqual(client.count("hotpot_slice").count, 2)

if __name__ == "__main__":
    unittest.main()
