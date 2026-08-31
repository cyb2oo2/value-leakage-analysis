import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from experiments.E02_trajectory_localization.holdout_annotation_helper import (
    TARGET_TYPES,
    locate_line,
    main,
    next_incomplete,
    show_reasoning,
)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_wrapper(
    directory: Path,
    blind_id: str,
    reasoning: str,
    visible_final: str = "SECRET VISIBLE FINAL",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = (
        f"BLINDED HOLDOUT ROLLOUT {blind_id}\n"
        "Public wrapper metadata.\n"
        "===== FULL REASONING (VERBATIM) =====\n\n"
        f"{reasoning}\n\n"
        "===== VISIBLE FINAL ANSWER (VERBATIM) =====\n\n"
        f"{visible_final}\n"
    )
    path = directory / f"{blind_id}.txt"
    path.write_bytes(payload.encode("utf-8"))
    return path


class HoldoutAnnotationHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / "public_bundle"
        self.rollouts = self.bundle / "holdout_rollouts"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _tables(
        self,
        operation_statuses: list[str],
        target_statuses: list[list[str]],
    ) -> tuple[Path, Path]:
        ids = [f"V{index:03d}" for index in range(1, len(operation_statuses) + 1)]
        operation_rows = [
            {"blind_id": blind_id, "annotation_status": status, "notes": ""}
            for blind_id, status in zip(ids, operation_statuses, strict=True)
        ]
        target_rows = []
        for blind_id, statuses in zip(ids, target_statuses, strict=True):
            for target_type, status in zip(TARGET_TYPES, statuses, strict=True):
                target_rows.append(
                    {
                        "blind_id": blind_id,
                        "target_type": target_type,
                        "adjudication_status": status,
                        "notes": "",
                    }
                )
        return (
            _write_csv(
                self.root / "operation.csv",
                ["blind_id", "annotation_status", "notes"],
                operation_rows,
            ),
            _write_csv(
                self.root / "targets.csv",
                ["blind_id", "target_type", "adjudication_status", "notes"],
                target_rows,
            ),
        )

    def test_next_returns_first_operation_then_fixed_target_stage(self) -> None:
        operation, targets = self._tables(
            ["", ""],
            [["", "", ""], ["", "", ""]],
        )
        self.assertEqual(
            next_incomplete(operation, targets),
            {"blind_id": "V001", "stage": "operation", "target_type": None},
        )

        operation, targets = self._tables(
            ["complete", ""],
            [["complete", "", ""], ["", "", ""]],
        )
        self.assertEqual(
            next_incomplete(operation, targets),
            {
                "blind_id": "V001",
                "stage": "target_adjudication",
                "target_type": "numerical_pivot",
            },
        )

    def test_next_reports_complete_and_rejects_stage_skips(self) -> None:
        operation, targets = self._tables(
            ["complete"],
            [["complete", "complete", "complete"]],
        )
        self.assertEqual(
            next_incomplete(operation, targets),
            {"blind_id": None, "stage": "complete", "target_type": None},
        )

        operation, targets = self._tables(
            ["complete"],
            [["", "complete", ""]],
        )
        with self.assertRaisesRegex(ValueError, "fixed order"):
            next_incomplete(operation, targets)

    def test_show_reasoning_reads_one_exact_wrapper_and_omits_visible_final(self) -> None:
        reasoning = "Alpha line.\nUnicode giraffe: 长颈鹿。"
        _write_wrapper(self.rollouts, "V001", reasoning)
        observed = show_reasoning(bundle=self.bundle, blind_id="V001")
        self.assertEqual(observed, reasoning)
        self.assertNotIn("SECRET VISIBLE FINAL", observed)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "show-reasoning",
                    "--bundle",
                    str(self.bundle),
                    "--blind-id",
                    "V001",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), reasoning + "\n")

    def test_show_reasoning_rejects_cr_invalid_utf8_and_delimiter_errors(self) -> None:
        path = _write_wrapper(self.rollouts, "V001", "Alpha")
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(ValueError, "CR bytes"):
            show_reasoning(rollouts_dir=self.rollouts, blind_id="V001")

        path.write_bytes(
            b"BLINDED HOLDOUT ROLLOUT V001\n"
            b"===== FULL REASONING (VERBATIM) =====\n\n"
            b"bad-utf8-\xff\n\n"
            b"===== VISIBLE FINAL ANSWER (VERBATIM) =====\n\nanswer"
        )
        with self.assertRaisesRegex(ValueError, "strict UTF-8"):
            show_reasoning(rollouts_dir=self.rollouts, blind_id="V001")

        path.write_text(
            "BLINDED HOLDOUT ROLLOUT V001\nno exact delimiters\n",
            encoding="utf-8",
            newline="",
        )
        with self.assertRaisesRegex(ValueError, "reasoning delimiter"):
            show_reasoning(rollouts_dir=self.rollouts, blind_id="V001")

    def test_locate_line_returns_all_unicode_spans_and_exact_hashes(self) -> None:
        reasoning = "前缀\nExact target Ω\nmiddle\nExact target Ω\nend"
        _write_wrapper(self.rollouts, "V007", reasoning, visible_final="do not expose")
        result = locate_line(
            rollouts_dir=self.rollouts,
            blind_id="V007",
            text="Exact target Ω",
        )
        expected_starts = [reasoning.index("Exact target Ω"), reasoning.rindex("Exact target Ω")]
        self.assertEqual(result["match_count"], 2)
        self.assertEqual(
            [match["start_char"] for match in result["matches"]],
            expected_starts,
        )
        target_hash = hashlib.sha256("Exact target Ω".encode("utf-8")).hexdigest()
        for match in result["matches"]:
            start, end = match["span"]
            self.assertEqual(reasoning[start:end], "Exact target Ω")
            self.assertEqual(match["target_sha256"], target_hash)
            self.assertEqual(
                match["prefix_before_sha256"],
                hashlib.sha256(reasoning[:start].encode("utf-8")).hexdigest(),
            )
            self.assertEqual(
                match["prefix_through_sha256"],
                hashlib.sha256(reasoning[:end].encode("utf-8")).hexdigest(),
            )

    def test_locate_line_rejects_substrings_multiline_and_missing_lines(self) -> None:
        _write_wrapper(self.rollouts, "V001", "Complete physical line\nAnother line")
        for invalid in ("physical line", "Complete physical line\nAnother line", "missing"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "physical line|line terminator"):
                    locate_line(
                        rollouts_dir=self.rollouts,
                        blind_id="V001",
                        text=invalid,
                    )

    def test_rejects_multi_id_wildcards_forbidden_paths_and_key_argument(self) -> None:
        _write_wrapper(self.rollouts, "V001", "Safe line")
        for invalid_id in ("V001,V002", "V*", "V01", "D001"):
            with self.subTest(invalid_id=invalid_id):
                with self.assertRaisesRegex(ValueError, "exactly one ID"):
                    show_reasoning(rollouts_dir=self.rollouts, blind_id=invalid_id)

        for forbidden_name in ("sealed", "runs"):
            forbidden = self.root / forbidden_name / "holdout_rollouts"
            _write_wrapper(forbidden, "V001", "Forbidden line")
            with self.subTest(forbidden_name=forbidden_name):
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    show_reasoning(rollouts_dir=forbidden, blind_id="V001")

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "show-reasoning",
                        "--bundle",
                        str(self.bundle),
                        "--blind-id",
                        "V001",
                        "--key",
                        "secret.json",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)

    def test_cli_next_and_locate_line_emit_json(self) -> None:
        operation, targets = self._tables(
            ["complete"],
            [["complete", "", ""]],
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(
                main(
                    [
                        "next",
                        "--operation-csv",
                        str(operation),
                        "--target-csv",
                        str(targets),
                    ]
                ),
                0,
            )
        self.assertEqual(json.loads(stdout.getvalue())["target_type"], "numerical_pivot")

        _write_wrapper(self.rollouts, "V001", "one\ntarget\nthree")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(
                main(
                    [
                        "locate-line",
                        "--rollouts-dir",
                        str(self.rollouts),
                        "--blind-id",
                        "V001",
                        "--text",
                        "target",
                    ]
                ),
                0,
            )
        self.assertEqual(json.loads(stdout.getvalue())["match_count"], 1)


if __name__ == "__main__":
    unittest.main()
