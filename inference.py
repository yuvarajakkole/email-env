"""
inference.py — Unified Inference + Training Entry Point
---------------------------------------------------------
Two modes:

  python inference.py --mode inference   # single-pass evaluation (default)
  python inference.py --mode train       # full RL training loop with learning

Inference mode:
  Runs all three tasks once using the TriagePolicy (LLM + memory + strategy).
  Uses any existing agent_memory.json to benefit from prior runs.

Train mode:
  Runs N episodes per task in curriculum order.
  Memory accumulates mistake patterns across episodes.
  Strategy layer recalibrates based on reward history.
  Produces training_report.json with full learning curve.

Reads from .env:
  OPENAI_API_KEY, API_BASE_URL, MODEL_NAME
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List

from env.core import EmailTriageEnv
from agent.memory import EpisodeMemory
from agent.strategy import StrategyLayer
from agent.policy import TriagePolicy
from agent.trainer import PolicyTrainer, TrainingConfig
from config import settings


# ─────────────────────────────────────────────
# Inference runner (single-pass)
# ─────────────────────────────────────────────

def run_inference(
    task_id:  str,
    seed:     int,
    policy:   TriagePolicy,
    verbose:  bool = True,
) -> Dict[str, Any]:
    env  = EmailTriageEnv(task_id=task_id, seed=seed)
    obs  = env.reset()
    done = False
    step_log = []
    sn = 0

    policy.begin_episode(task_id)

    print(f"\n{'='*70}")
    print(f"  INFERENCE | TASK: {task_id.upper():<8} | SEED: {seed}")
    print(f"{'='*70}")

    while not done and obs is not None:
        sn += 1
        if verbose:
            ep  = obs.episode_state
            flag = "🔥 FOLLOWUP " if obs.is_followup else ""
            print(f"\n[{sn:02d}] {flag}{obs.email_id} | {obs.subject[:55]}")
            print(f"     health={ep.inbox_health:.2f} sat={ep.user_satisfaction:.2f} "
                  f"errors={ep.consecutive_errors} unresolved={len(ep.unresolved_threads)}")

        action, uncertainty = policy.act(obs, task_id)

        if verbose:
            unc_flag = " 🤔" if uncertainty.ambiguity_score > 0.5 else ""
            print(f"  ▶ {action.label.value:<12} {action.priority.value:<7} "
                  f"{action.action_type.value:<25} "
                  f"conf={action.confidence:.2f} "
                  f"adj={uncertainty.adjusted_confidence:.2f}"
                  f"  amb={uncertainty.ambiguity_score:.2f}{unc_flag}")
            if action.response_text:
                print(f"  ▶ {action.response_text[:90].replace(chr(10),' ')}…")
            if uncertainty.reasoning != "Strategy: no override — confidence sufficient":
                print(f"  🧠 {uncertainty.reasoning}")

        obs, reward, done, info = env.step(action)

        if verbose:
            gt = info.get("ground_truth", {})
            print(f"  ◆ r={reward.total_reward:+.3f}  "
                  f"pen={reward.penalty:.3f}  cost={reward.action_cost:.3f}  "
                  f"[GT:{gt.get('label','?')}/{gt.get('priority','?')}/{gt.get('action_type','?')}]")
            print(f"    {reward.feedback}")

        # Record into memory
        policy.observe(
            obs or _stub_obs(),
            action, reward,
            info.get("ground_truth", {}),
            info,
        )

        step_log.append({
            "step":          sn,
            "email_id":      info.get("email_id"),
            "ground_truth":  info.get("ground_truth"),
            "action":        action.dict(),
            "uncertainty":   uncertainty.__dict__,
            "reward":        reward.dict(),
            "episode_state": info.get("episode_state", {}),
        })
        time.sleep(0.05)

    result = env.episode_result()
    policy.end_episode(result.final_score, env._ep_state)

    bd = result.grader_breakdown
    print(f"\n{'─'*70}")
    print(f"  TASK {task_id.upper()} COMPLETE")
    print(f"  Final Score: {result.final_score:.4f}  ({result.final_score*100:.1f}%)")
    print(f"  Cls:{bd.get('avg_classification',0):.3f}  "
          f"Pri:{bd.get('avg_priority',0):.3f}  "
          f"Act:{bd.get('avg_action',0):.3f}  "
          f"Rsp:{bd.get('avg_response',0):.3f}")
    print(f"  Health:{bd.get('final_inbox_health',1):.3f}  "
          f"UserSat:{bd.get('final_user_sat',1):.3f}  "
          f"Cost:{bd.get('total_action_cost',0):.3f}")
    print(f"{'─'*70}")

    return {
        "task_id":     task_id,
        "final_score": result.final_score,
        "breakdown":   bd,
        "step_log":    step_log,
    }


class _stub_obs:
    email_id = ""; subject = ""; sender = ""
    body = ""; thread_history = []; is_followup = False
    episode_state = None


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Email Triage RL — Inference & Training")
    parser.add_argument("--mode",     default="inference",
                        choices=["inference", "train"],
                        help="inference=single-pass eval | train=multi-episode RL loop")
    parser.add_argument("--task",     default="all",
                        choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--output",   default="inference_results.json")
    parser.add_argument("--memory",   default="agent_memory.json",
                        help="Path to persistent agent memory file")
    parser.add_argument("--strategy", default="balanced",
                        choices=["aggressive", "balanced", "conservative"])
    parser.add_argument("--n-easy",   type=int, default=3)
    parser.add_argument("--n-medium", type=int, default=4)
    parser.add_argument("--n-hard",   type=int, default=5)
    parser.add_argument("--no-curriculum", action="store_true", default=False)
    parser.add_argument("--verbose",  action="store_true", default=True)
    args = parser.parse_args()

    settings.validate()

    print(f"\n{'#'*70}")
    print(f"  Email Triage RL v3 — {'Training' if args.mode=='train' else 'Inference'}")
    print(f"  Config: {settings.safe_summary()}")
    print(f"{'#'*70}")

    # Build agent components
    memory   = EpisodeMemory(memory_path=args.memory)
    strategy = StrategyLayer(risk_mode=args.strategy)
    policy   = TriagePolicy(memory=memory, strategy=strategy, verbose=args.verbose)

    if memory.n_episodes() > 0:
        print(f"\n  📚 Loaded memory: {memory.n_episodes()} prior episodes | "
              f"ECE: {memory.calibrator.calibration_error():.3f} | "
              f"Weakest labels: {memory.weakest_labels(3)}")

    # ── Training mode ──────────────────────────────────────────────────
    if args.mode == "train":
        config = TrainingConfig(
            n_episodes_easy   = args.n_easy,
            n_episodes_medium = args.n_medium,
            n_episodes_hard   = args.n_hard,
            strategy_mode     = args.strategy,
            base_seed         = args.seed,
            report_path       = args.output.replace(".json", "_training.json"),
            memory_path       = args.memory,
            curriculum        = not args.no_curriculum,
            verbose           = args.verbose,
        )
        trainer = PolicyTrainer(config=config, policy=policy, memory=memory)
        report  = trainer.train()

        with open(config.report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nTraining report saved → {config.report_path}")
        return

    # ── Inference mode ─────────────────────────────────────────────────
    tasks   = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    results: Dict[str, Any] = {}

    for task_id in tasks:
        results[task_id] = run_inference(
            task_id = task_id,
            seed    = args.seed,
            policy  = policy,
            verbose = args.verbose,
        )

    # Summary
    print(f"\n{'#'*70}")
    print(f"  FINAL SCORES SUMMARY")
    print(f"{'#'*70}")
    total = 0.0
    for task_id, r in results.items():
        s = r["final_score"]; total += s
        print(f"  {task_id.upper():<8} {s:.4f}  {'█' * int(s * 25)}")
    avg = total / len(results) if results else 0.0
    print(f"  {'─'*50}")
    print(f"  AVERAGE  {avg:.4f}  ({avg*100:.1f}%)")

    # Agent stats
    stats = policy.stats()
    print(f"\n  Agent stats:")
    print(f"    LLM calls:      {stats['total_calls']}")
    print(f"    Fallback rate:  {stats['fallback_rate']:.1%}")
    print(f"    ECE:            {stats['ece']:.3f}")
    print(f"    Override rate:  {stats['strategy']['override_rate']:.1%}")
    print(f"    Weakest labels: {stats['weakest_labels']}")
    print(f"{'#'*70}\n")

    # Save
    output = {
        "model":      settings.model_name,
        "base_url":   settings.api_base_url,
        "seed":       args.seed,
        "results":    results,
        "agent":      stats,
        "summary": {
            "avg_score":   round(avg, 4),
            "task_scores": {t: r["final_score"] for t, r in results.items()},
        },
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
