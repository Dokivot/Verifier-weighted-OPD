from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

from pydantic import ValidationError

from opd.checkpoints.merge import merge_adapter
from opd.checkpoints.promotion import evaluate_promotion
from opd.config import load_config
from opd.data.contamination import audit_contamination
from opd.data.eval_import import fetch_evaluation_data, import_evaluation_data
from opd.data.prepare import prepare_dataset
from opd.doctor import run_doctor
from opd.evaluation.compare import compare_runs
from opd.evaluation.lighteval import run_lighteval
from opd.evaluation.runner import evaluate
from opd.exceptions import OPDError
from opd.monitoring.budget import check_budget
from opd.reporting.analysis import build_cost_pareto, build_failure_analysis
from opd.reporting.build import build_report
from opd.rollout.pipeline import generate_rollouts
from opd.teacher.pipeline import annotate_rollouts
from opd.training.audit import audit_sparse_kl
from opd.training.pipeline import train
from opd.training.views import build_training_view
from opd.verifier.pipeline import verify_rollouts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opd", description="OPD-Lab pipeline CLI")
    parser.add_argument("--config", default="configs/smoke.yaml", help="YAML config path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor")
    subparsers.add_parser("budget")

    audit = subparsers.add_parser("audit")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    sparse_kl_audit = audit_sub.add_parser("sparse-kl")
    sparse_kl_audit.add_argument("--output", required=True)

    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint_sub = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    checkpoint_merge = checkpoint_sub.add_parser("merge")
    checkpoint_merge.add_argument("--adapter", required=True)
    checkpoint_merge.add_argument("--output", required=True)
    checkpoint_promote = checkpoint_sub.add_parser("promote")
    checkpoint_promote.add_argument("--baseline", required=True)
    checkpoint_promote.add_argument("--candidate", required=True)
    checkpoint_promote.add_argument("--checkpoint", required=True)
    checkpoint_promote.add_argument("--output", required=True)

    data = subparsers.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_sub.add_parser("prepare")
    data_sub.add_parser("audit-contamination")
    import_eval = data_sub.add_parser("import-eval")
    import_eval.add_argument("--name", required=True)
    import_eval.add_argument("--input", required=True, dest="input_path")
    fetch_eval = data_sub.add_parser("fetch-eval")
    fetch_eval.add_argument("--name", required=True)
    build_view = data_sub.add_parser("build-view")
    build_view.add_argument("--round", type=int, default=0, dest="round_id")
    build_view.add_argument(
        "--method",
        choices=[
            "vanilla-opd",
            "vanilla_opd",
            "verifier-opd",
            "verifier_opd",
            "confidence-opd",
            "confidence_opd",
            "weighted-opd",
            "weighted_opd",
        ],
        required=True,
    )

    rollout = subparsers.add_parser("rollout")
    rollout_sub = rollout.add_subparsers(dest="rollout_command", required=True)
    rollout_generate = rollout_sub.add_parser("generate")
    rollout_generate.add_argument("--round", type=int, default=0, dest="round_id")

    verify = subparsers.add_parser("verify")
    verify_sub = verify.add_subparsers(dest="verify_command", required=True)
    verify_math = verify_sub.add_parser("math")
    verify_math.add_argument("--round", type=int, default=0, dest="round_id")

    teacher = subparsers.add_parser("teacher")
    teacher_sub = teacher.add_subparsers(dest="teacher_command", required=True)
    teacher_annotate = teacher_sub.add_parser("annotate")
    teacher_annotate.add_argument("--round", type=int, default=0, dest="round_id")

    subparsers.add_parser("train")

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--suite", default="smoke")
    evaluation.add_argument("--checkpoint")

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--baseline", required=True)
    comparison.add_argument("--candidate", required=True)
    comparison.add_argument("--output", required=True)

    benchmark = subparsers.add_parser("benchmark")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    benchmark_run = benchmark_sub.add_parser("run")
    benchmark_run.add_argument("--checkpoint", required=True)
    benchmark_run.add_argument("--output-dir", required=True)
    benchmark_run.add_argument(
        "--tasks", help="Comma-separated registry names; defaults to benchmark.tasks"
    )

    report = subparsers.add_parser("report")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    report_build = report_sub.add_parser("build")
    report_build.add_argument("--experiment", required=True)
    report_failures = report_sub.add_parser("failures")
    report_failures.add_argument("--predictions", required=True)
    report_failures.add_argument("--output", required=True)
    report_cost = report_sub.add_parser("cost")
    report_cost.add_argument("--output-dir", required=True)
    return parser


def _print_result(value: object) -> None:
    if isinstance(value, Path):
        print(value)
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--config" in arguments:
        index = arguments.index("--config")
        if index + 1 >= len(arguments):
            parser.error("--config requires a path")
        config_pair = arguments[index : index + 2]
        del arguments[index : index + 2]
        arguments = [*config_pair, *arguments]
    args = parser.parse_args(arguments)
    try:
        config = load_config(args.config)
        result: object
        if args.command == "doctor":
            result = run_doctor(config)
        elif args.command == "budget":
            result = check_budget(config)
        elif args.command == "audit" and args.audit_command == "sparse-kl":
            result = audit_sparse_kl(config, args.output)
        elif args.command == "checkpoint" and args.checkpoint_command == "merge":
            result = merge_adapter(config, adapter_path=args.adapter, output_path=args.output)
        elif args.command == "checkpoint" and args.checkpoint_command == "promote":
            result = evaluate_promotion(
                config,
                baseline_summary_path=args.baseline,
                candidate_summary_path=args.candidate,
                checkpoint_path=args.checkpoint,
                output_path=args.output,
            )
        elif args.command == "data" and args.data_command == "prepare":
            result = prepare_dataset(config)
        elif args.command == "data" and args.data_command == "audit-contamination":
            result = audit_contamination(config)
        elif args.command == "data" and args.data_command == "import-eval":
            result = import_evaluation_data(config, name=args.name, input_path=args.input_path)
        elif args.command == "data" and args.data_command == "fetch-eval":
            result = fetch_evaluation_data(config, name=args.name)
        elif args.command == "data" and args.data_command == "build-view":
            result = build_training_view(
                config,
                round_id=args.round_id,
                method=args.method.replace("-", "_"),
            )
        elif args.command == "rollout" and args.rollout_command == "generate":
            result = generate_rollouts(config, round_id=args.round_id)
        elif args.command == "verify" and args.verify_command == "math":
            result = verify_rollouts(config, round_id=args.round_id)
        elif args.command == "teacher" and args.teacher_command == "annotate":
            result = annotate_rollouts(config, round_id=args.round_id)
        elif args.command == "train":
            result = train(config)
        elif args.command == "evaluate":
            evaluation_config = deepcopy(config)
            if args.checkpoint:
                evaluation_config["evaluation"]["model_name"] = args.checkpoint
                evaluation_config["evaluation"]["model_revision"] = "local-checkpoint"
                evaluation_config["evaluation"]["tokenizer_revision"] = "local-checkpoint"
            result = evaluate(evaluation_config, suite=args.suite)
        elif args.command == "compare":
            result = compare_runs(config, args.baseline, args.candidate, args.output)
        elif args.command == "benchmark" and args.benchmark_command == "run":
            task_names = args.tasks.split(",") if args.tasks else None
            result = run_lighteval(
                config,
                checkpoint=args.checkpoint,
                output_dir=args.output_dir,
                task_names=task_names,
            )
        elif args.command == "report" and args.report_command == "build":
            result = build_report(config, args.experiment)
        elif args.command == "report" and args.report_command == "failures":
            result = build_failure_analysis(config, args.predictions, args.output)
        elif args.command == "report" and args.report_command == "cost":
            result = build_cost_pareto(config, args.output_dir)
        else:
            parser.error("Unsupported command")
            return
        _print_result(result)
    except (OPDError, ValidationError, ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
