"""Unit tests for the clean-index evaluation runner.

All tests are offline. They use the frozen index only for read-only structural
checks and Mocks for API/inference behavior.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.clean_index_evaluation import (
    APICallRecorder,
    DEFAULT_EXPECTED_FINGERPRINT,
    OUTPUT_CONTRACT_FILES,
    CleanIndexSnapshot,
    ConsecutiveFailureLimit,
    FailureTracker,
    IndexFingerprintChanged,
    RunContext,
    RecordedLLM,
    aggregate_retrieval_metrics,
    build_expected_chunk_mapping,
    build_retrieval_metrics_bundle,
    evaluate_retrieval_result,
    initialize_output_contract,
    load_and_verify_index,
    read_jsonl,
    resolve_no_answer_cases,
    summarize_usage,
    upsert_jsonl_by_question_id,
    validate_evaluation_dataset,
    validate_explicit_index_paths,
    validate_no_answer_cases,
    validate_resume,
    verify_no_answer_absence_in_clean_index,
    write_evaluation_outputs,
)
from evaluation.no_answer_cases import NO_ANSWER_CASES
from src.settings import OpenAICompatible
from src.config import Settings as AppSettings


ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = (
    ROOT / "evaluation" / "index_audit" / "clean_index" / "storage" / "chroma"
)
DOCSTORE_DIR = (
    ROOT / "evaluation" / "index_audit" / "clean_index" / "storage" / "docstore"
)
MANIFEST = ROOT / "evaluation" / "index_audit" / "clean_index" / "index_manifest.json"
DATASET = ROOT / "evaluation" / "datasets" / "rag_eval_50.jsonl"
NODE_DIFF = (
    ROOT
    / "evaluation"
    / "index_audit"
    / "clean_index"
    / "comparison"
    / "node_diff.csv"
)


class CleanIndexPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = [
            json.loads(line)
            for line in DATASET.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.snapshot = load_and_verify_index(
            CHROMA_DIR,
            DOCSTORE_DIR,
            MANIFEST,
            DEFAULT_EXPECTED_FINGERPRINT,
        )
        cls.mapping_rows, cls.mapping_summary = build_expected_chunk_mapping(
            cls.dataset,
            NODE_DIFF,
            cls.snapshot,
        )

    def test_clean_index_path_isolation(self) -> None:
        result = validate_explicit_index_paths(CHROMA_DIR, DOCSTORE_DIR)
        self.assertEqual(
            Path(result["chroma_dir"]),
            CHROMA_DIR.resolve(),
        )
        self.assertEqual(
            Path(result["docstore_dir"]),
            DOCSTORE_DIR.resolve(),
        )
        with self.assertRaises(RuntimeError):
            validate_explicit_index_paths(
                ROOT / "storage" / "chroma",
                ROOT / "storage" / "docstore",
            )

    def test_logical_fingerprint_validation(self) -> None:
        self.assertEqual(
            self.snapshot.fingerprint,
            DEFAULT_EXPECTED_FINGERPRINT,
        )
        with self.assertRaises(IndexFingerprintChanged):
            load_and_verify_index(
                CHROMA_DIR,
                DOCSTORE_DIR,
                MANIFEST,
                "0" * 64,
            )

    def test_dataset_and_mapping_completeness(self) -> None:
        summary = validate_evaluation_dataset(self.dataset)
        self.assertEqual(summary["record_count"], 50)
        self.assertEqual(summary["unique_ids"], 50)
        self.assertEqual(self.mapping_summary["expected_old_ids"], 40)
        self.assertEqual(self.mapping_summary["mapped_clean_ids"], 40)
        self.assertEqual(
            self.mapping_summary["evidence_complete_questions"], 50
        )
        self.assertEqual(self.mapping_summary["status"], "passed")
        self.assertTrue(
            all(row["question_evidence_complete"] for row in self.mapping_rows)
        )
        self.assertTrue(
            all(
                row["old_text_sha256_normalized"]
                and row["new_text_sha256_normalized"]
                for row in self.mapping_rows
            )
        )

    def test_model09_and_cross09_evidence(self) -> None:
        for question_id in ("MODEL-09", "CROSS-09"):
            result = self.mapping_summary["focus_review"][question_id]
            self.assertTrue(result["evidence_complete"])
            self.assertTrue(result["expected_chunks_clean"])
            self.assertTrue(
                all(item["present"] for item in result["evidence_fragments"])
            )
            self.assertTrue(
                all(
                    row["focus_review"]
                    for row in self.mapping_rows
                    if row["question_id"] == question_id
                )
            )

    def test_top4_production_configuration(self) -> None:
        cfg = AppSettings()  # type: ignore[call-arg]
        self.assertEqual(cfg.rerank_top_n, 4)

    def test_no_answer_ten_cases(self) -> None:
        cases, source = resolve_no_answer_cases()
        summary = validate_no_answer_cases(cases)
        canonical_summary = validate_no_answer_cases(
            [dict(case) for case in NO_ANSWER_CASES]
        )
        self.assertEqual(summary["record_count"], 10)
        self.assertEqual(summary["sha256"], canonical_summary["sha256"])
        self.assertTrue(summary["matches_canonical_source"])
        self.assertIn(
            source,
            (
                str(
                    (
                        ROOT
                        / "evaluation"
                        / "results"
                        / "full_evaluation_20261112_234856"
                        / "no_answer_eval_10.jsonl"
                    ).resolve()
                ),
                "evaluation/no_answer_cases.py",
            ),
        )
        absence = verify_no_answer_absence_in_clean_index(
            cases,
            self.snapshot,
        )
        self.assertEqual(absence["case_count"], 10)
        self.assertTrue(absence["all_verified_unanswerable"])


class RetrievalMetricTests(unittest.TestCase):
    def test_required_all_metrics(self) -> None:
        question = {
            "question_id": "MODEL-01",
            "category": "跨产品型号归属辨析",
            "expected_chunk": "a",
            "expected_chunks": ["a", "b"],
        }
        result = evaluate_retrieval_result(
            question,
            ["x", "b", "y", "a"],
            ["a", "b"],
        )
        self.assertEqual(result["evidence_mode"], "required_all")
        self.assertEqual(result["hit_at_1"], 0.0)
        self.assertEqual(result["hit_at_3"], 1.0)
        self.assertEqual(result["recall_at_3"], 0.5)
        self.assertEqual(result["recall_at_4"], 1.0)
        self.assertEqual(result["mrr_at_4"], 0.5)

    def test_alternative_evidence_metrics(self) -> None:
        question = {
            "question_id": "TROUBLE-01",
            "category": "故障排查",
            "expected_chunk": "a",
            "expected_chunks": ["a", "b"],
        }
        result = evaluate_retrieval_result(
            question,
            ["x", "b", "y", "z"],
            ["a", "b"],
        )
        self.assertEqual(result["evidence_mode"], "alternative")
        self.assertEqual(result["recall_at_3"], 1.0)
        self.assertEqual(result["mrr_at_4"], 0.5)

    def test_api_failure_counts_in_primary_denominator(self) -> None:
        question = {
            "question_id": "PARAM-01",
            "category": "参数查询",
            "expected_chunk": "a",
            "expected_chunks": ["a"],
        }
        failed = evaluate_retrieval_result(
            question,
            [],
            ["a"],
            retrieval_success=False,
        )
        bundle, _, _ = build_retrieval_metrics_bundle([failed])
        self.assertEqual(bundle["overall"]["primary_denominator"], 1)
        self.assertEqual(bundle["overall"]["hit_at_3"], 0.0)
        self.assertEqual(bundle["overall"]["successful_questions"], 0)


class UsageAndRecoveryTests(unittest.TestCase):
    def test_multiple_usage_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sink = Path(temp) / "api_usage.jsonl"
            recorder = APICallRecorder("test-run", sink)
            first = recorder.start(
                question_id="Q1",
                scope="answer",
                provider="test",
                model="mock",
                call_type="llm",
                streaming=True,
            )
            recorder.finish(
                first,
                status="success",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            )
            fallback = recorder.start(
                question_id="Q1",
                scope="answer",
                provider="test",
                model="mock",
                call_type="llm",
                streaming=False,
                retry=True,
                fallback=True,
            )
            recorder.finish(
                fallback,
                status="success",
                usage={
                    "prompt_tokens": 8,
                    "completion_tokens": 4,
                    "total_tokens": 12,
                },
            )
            summary = summarize_usage(recorder.records)
            self.assertEqual(summary["llm_calls"], 2)
            self.assertEqual(summary["total_tokens"], 27)
            self.assertTrue(summary["usage_complete"])
            persisted = read_jsonl(sink)
            self.assertEqual(len(persisted), 2)
            self.assertTrue(persisted[0]["streaming"])
            self.assertTrue(persisted[1]["fallback"])
            self.assertTrue(persisted[1]["retry"])

    def test_stream_empty_then_nonstream_fallback_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sink = Path(temp) / "api_usage.jsonl"
            recorder = APICallRecorder("stream-test", sink)
            llm = RecordedLLM(
                recorder=recorder,
                provider_secrets=[],
                model="mock-llm",
                api_key="mock-key",
                api_base="http://127.0.0.1:1/v1",
                max_tokens=32,
            )
            llm.begin_question("Q1", "answer")
            with patch.object(
                OpenAICompatible,
                "stream_chat",
                return_value=iter([]),
            ):
                self.assertEqual(list(llm.stream_chat(messages=[])), [])
            llm.configure_next_call(
                "nonstream_fallback",
                retry=True,
                fallback=True,
            )
            fake_response = SimpleNamespace(
                additional_kwargs={
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                }
            )
            with patch.object(
                OpenAICompatible,
                "chat",
                return_value=fake_response,
            ):
                response = llm.chat(messages=[])
            self.assertIs(response, fake_response)
            records = read_jsonl(sink)
            self.assertEqual(len(records), 2)
            self.assertTrue(records[0]["streaming"])
            self.assertFalse(records[0]["fallback"])
            self.assertFalse(records[1]["streaming"])
            self.assertTrue(records[1]["retry"])
            self.assertTrue(records[1]["fallback"])
            self.assertEqual(summarize_usage(records)["total_tokens"], 6)

    def test_consecutive_failure_limit(self) -> None:
        tracker = FailureTracker(3)
        tracker.record(True)
        tracker.record(False)
        tracker.record(True)
        tracker.record(True)
        self.assertFalse(tracker.should_pause())
        tracker.record(True)
        self.assertTrue(tracker.should_pause())
        with self.assertRaises(ConsecutiveFailureLimit):
            if tracker.should_pause():
                raise ConsecutiveFailureLimit("threshold reached")

    def test_resume_validation_and_no_duplicate_append(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            paths = initialize_output_contract(run_dir, "run-test")
            preflight = {
                "dataset": {"sha256": "dataset"},
                "no_answer": {"sha256": "noanswer"},
                "index": {"fingerprint": DEFAULT_EXPECTED_FINGERPRINT},
            }
            context = RunContext(
                run_dir=run_dir,
                run_id="run-test",
                dataset=[
                    {"question_id": "Q1"},
                    {"question_id": "Q2"},
                ],
                no_answer_cases=[{"question_id": "NA-01"}],
                mapping_rows=[],
                mapping_summary={"status": "passed"},
                preflight=preflight,
                config_fingerprint="config-hash",
                answer_rows=[
                    {
                        "question_id": "Q1",
                        "had_api_failure": False,
                    }
                ],
                abstention_rows=[
                    {
                        "question_id": "NA-01",
                        "had_api_failure": False,
                    }
                ],
            )
            append_rows = {
                "question_id": "Q1",
                "execution_status": "completed",
            }
            upsert_jsonl_by_question_id(
                paths["answer_results.jsonl"],
                append_rows,
            )
            upsert_jsonl_by_question_id(
                paths["answer_results.jsonl"],
                append_rows,
            )
            self.assertEqual(
                len(read_jsonl(paths["answer_results.jsonl"])),
                1,
            )
            upsert_jsonl_by_question_id(
                paths["abstention_results.jsonl"],
                {
                    "question_id": "NA-01",
                    "execution_status": "completed",
                },
            )
            context.persist("running", "answer_50")
            answers, abstentions = validate_resume(
                run_dir,
                expected_config_fingerprint="config-hash",
            )
            self.assertEqual([row["question_id"] for row in answers], ["Q1"])
            self.assertEqual(
                [row["question_id"] for row in abstentions],
                ["NA-01"],
            )
            with self.assertRaises(RuntimeError):
                validate_resume(
                    run_dir,
                    expected_config_fingerprint="changed-config",
                )


class OutputContractTests(unittest.TestCase):
    def test_all_required_output_paths_and_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            paths = initialize_output_contract(run_dir, "output-test")
            question = {
                "question_id": "PARAM-01",
                "category": "参数查询",
                "question": "减少多少脂肪？",
                "expected_answer": "减少 90% 的脂肪量。",
                "evaluation_points": ["减少 90% 脂肪量"],
                "expected_chunk": "old-1",
                "expected_chunks": ["old-1"],
                "source_document": "空气炸锅.pdf",
                "evidence": "减少 90% 的脂肪量。",
            }
            retrieval_eval = evaluate_retrieval_result(
                question,
                ["new-1", "new-2"],
                ["new-1"],
            )
            answer_rows = [
                {
                    "question_id": "PARAM-01",
                    "category": "参数查询",
                    "question": question["question"],
                    "answer": "可减少 90% 的脂肪量 [1]。",
                    "retrieved": [
                        {
                            "node_id": "new-1",
                            "document": "空气炸锅.pdf",
                            "page_label": "6",
                            "text": "同时减少 90% 的脂肪量。",
                        }
                    ],
                    "retrieved_node_ids": ["new-1", "new-2"],
                    "retrieval_evaluation": retrieval_eval,
                    "execution_status": "completed",
                    "embedding_ms": 10.0,
                    "vector_ms": 5.0,
                    "bm25_ms": 4.0,
                    "rrf_ms": 2.0,
                    "rerank_ms": 30.0,
                    "retrieval_ms": 51.0,
                    "llm_first_token_ms": 100.0,
                    "llm_generation_ms": 500.0,
                    "total_ms": 551.0,
                    "api_error_message": None,
                }
            ]
            abstention_rows = [
                {
                    "question_id": "NA-01",
                    "category": "无答案",
                    "question": "不存在的问题？",
                    "answer": "资料中未提到。",
                    "retrieved": [],
                    "retrieved_node_ids": [],
                    "execution_status": "completed",
                    "classification": "correct_refusal",
                    "embedding_ms": 10.0,
                    "vector_ms": 5.0,
                    "bm25_ms": 4.0,
                    "rrf_ms": 2.0,
                    "rerank_ms": 30.0,
                    "retrieval_ms": 51.0,
                    "llm_first_token_ms": 100.0,
                    "llm_generation_ms": 500.0,
                    "total_ms": 551.0,
                    "api_error_message": None,
                }
            ]
            append_jsonl_rows = [
                {"question_id": answer_rows[0]["question_id"]}
            ]
            paths["answer_results.jsonl"].write_text(
                "\n".join(json.dumps(row) for row in append_jsonl_rows) + "\n",
                encoding="utf-8",
            )
            paths["abstention_results.jsonl"].write_text(
                json.dumps({"question_id": "NA-01"}) + "\n",
                encoding="utf-8",
            )
            api_records = [
                {
                    "call_type": "embedding",
                    "status": "success",
                    "streaming": False,
                    "retry": False,
                    "fallback": False,
                },
                {
                    "call_type": "reranker",
                    "status": "success",
                    "streaming": False,
                    "retry": False,
                    "fallback": False,
                },
                {
                    "call_type": "llm",
                    "status": "success",
                    "streaming": True,
                    "retry": False,
                    "fallback": False,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            ]
            mapping_rows = [
                {
                    "question_id": "PARAM-01",
                    "old_node_id": "old-1",
                    "new_node_id": "new-1",
                    "mapping_status": "passed",
                }
            ]
            preflight = {
                "index": {"fingerprint": DEFAULT_EXPECTED_FINGERPRINT},
                "dataset": {"sha256": "dataset"},
                "no_answer": {"sha256": "noanswer"},
            }
            write_evaluation_outputs(
                run_dir=run_dir,
                run_id="output-test",
                answer_rows=answer_rows,
                abstention_rows=abstention_rows,
                api_records=api_records,
                mapping_rows=mapping_rows,
                dataset=[question],
                mapping_summary={"status": "passed"},
                preflight=preflight,
            )
            for filename in OUTPUT_CONTRACT_FILES:
                self.assertTrue((run_dir / filename).is_file(), filename)
            retrieval_metrics = json.loads(
                (run_dir / "retrieval_metrics.json").read_text(encoding="utf-8")
            )
            self.assertIn("overall", retrieval_metrics)
            self.assertIn("hit_at_4", retrieval_metrics["overall"])
            answer_metrics = json.loads(
                (run_dir / "answer_metrics.json").read_text(encoding="utf-8")
            )
            self.assertIn("overall", answer_metrics)
            cost_metrics = json.loads(
                (run_dir / "cost_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cost_metrics["llm"]["actual_cost"], "unknown")
            status = json.loads(
                (run_dir / "RUN_STATUS.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["run_status"], "initialized")


if __name__ == "__main__":
    unittest.main()
