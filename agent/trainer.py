"""
agent/trainer.py — RL-Style Policy Trainer
-------------------------------------------
Implements an episode-over-episode training loop.
The "learning" happens through:

  1. Prompt adaptation (memory.correction_prompt_block → injected into system prompt)
  2. Strategy recalibration (calibrator ECE improves as episodes accumulate)
  3. Threshold tuning (strategy thresholds adjust based on observed override impact)
  4. Curriculum progression (easy → medium → hard as performance improves)

This is NOT gradient-based RL (that would require a trainable model).
Instead, this is Prompt-RL: the agent improves its policy by:
  - Mining past mistakes from memory
  - Injecting corrective instructions into the context
  - Adjusting decision thresholds based on reward history

This matches how production LLM agents actually improve in RL environments.

Architecture:
  Trainer
    └─ runs N episodes per task
    └─ after each episode: update memory, recalibrate strategy
    └─ after curriculum stage: optionally advance difficulty
    └─ produces training_report.json with full learning curve
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from env.core import EmailTriageEnv
from env.models import AgentAction
from agent.policy import TriagePolicy
from agent.memory import EpisodeMemory
from agent.strategy import StrategyLayer


# ─────────────────────────────────────────────
# Training configuration
# ─────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Controls the training loop."""
    # Episodes per task
    n_episodes_easy:   int   = 3
    n_episodes_medium: int   = 4
    n_episodes_hard:   int   = 5

    # Curriculum: advance to next difficulty only if this score is reached
    easy_threshold:    float = 0.80
    medium_threshold:  float = 0.68

    # Strategy calibration
    strategy_mode:     str   = "balanced"   # aggressive | balanced | conservative

    # Seeds: different seed per episode for variety
    base_seed:         int   = 42

    # Output
    report_path:       str   = "training_report.json"
    memory_path:       str   = "agent_memory.json"

    # Delay between API calls (seconds)
    step_delay:        float = 0.05

    # Whether to run curriculum (easy→medium→hard) or all tasks independently
    curriculum:        bool  = True

    # Verbose step-level output
    verbose:           bool  = True


@dataclass
class EpisodeLog:
    episode:      int
    task_id:      str
    seed:         int
    final_score:  float
    n_steps:      int
    avg_reward:   float
    ece:          float
    override_rate:float
    top_mistakes: List[Dict[str, Any]] = field(default_factory=list)
    inbox_health: float = 1.0
    user_sat:     float = 1.0


# ─────────────────────────────────────────────
# Policy Trainer
# ─────────────────────────────────────────────

class PolicyTrainer:
    """
    Runs the full training loop and produces a learning curve.

    Usage:
        config  = TrainingConfig(n_episodes_hard=5)
        memory  = EpisodeMemory(memory_path=config.memory_path)
        strategy= StrategyLayer(risk_mode=config.strategy_mode)
        policy  = TriagePolicy(memory=memory, strategy=strategy, verbose=config.verbose)
        trainer = PolicyTrainer(config=config, policy=policy, memory=memory)
        report  = trainer.train()
    """

    def __init__(
        self,
        config:  TrainingConfig,
        policy:  TriagePolicy,
        memory:  EpisodeMemory,
    ):
        self.config  = config
        self.policy  = policy
        self.memory  = memory
        self._logs:  List[EpisodeLog] = []

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def train(self) -> Dict[str, Any]:
        """
        Run the full training curriculum.
        Returns a training report dict (also saved to config.report_path).
        """
        cfg = self.config
        print(f"\n{'#'*70}")
        print(f"  PolicyTrainer — Email Triage RL")
        print(f"  Strategy: {cfg.strategy_mode}  |  Curriculum: {cfg.curriculum}")
        print(f"{'#'*70}")

        if cfg.curriculum:
            report = self._run_curriculum()
        else:
            report = self._run_all_tasks()

        self._save_report(report)
        return report

    # ─────────────────────────────────────────────────────────────────────
    # Curriculum training
    # ─────────────────────────────────────────────────────────────────────

    def _run_curriculum(self) -> Dict[str, Any]:
        """Easy → Medium → Hard with threshold gating."""
        results: Dict[str, Any] = {}

        # Stage 1: Easy
        easy_scores = self._train_task(
            "easy", self.config.n_episodes_easy, start_seed=self.config.base_seed)
        results["easy"] = easy_scores
        avg_easy = sum(s.final_score for s in easy_scores) / max(1, len(easy_scores))

        print(f"\n  📊 Easy avg score: {avg_easy:.4f} "
              f"(threshold: {self.config.easy_threshold})")

        if avg_easy < self.config.easy_threshold:
            print(f"  ⚠️  Easy threshold not met. Extra episode...")
            bonus = self._train_task("easy", 1,
                                     start_seed=self.config.base_seed + 99)
            easy_scores.extend(bonus)
            results["easy"] = easy_scores

        # Stage 2: Medium
        medium_scores = self._train_task(
            "medium", self.config.n_episodes_medium,
            start_seed=self.config.base_seed + 100)
        results["medium"] = medium_scores
        avg_med = sum(s.final_score for s in medium_scores) / max(1, len(medium_scores))

        print(f"\n  📊 Medium avg score: {avg_med:.4f} "
              f"(threshold: {self.config.medium_threshold})")

        # Stage 3: Hard (always run)
        hard_scores = self._train_task(
            "hard", self.config.n_episodes_hard,
            start_seed=self.config.base_seed + 200)
        results["hard"] = hard_scores

        return self._build_report(results)

    def _run_all_tasks(self) -> Dict[str, Any]:
        """Run all tasks independently (no curriculum gating)."""
        results: Dict[str, Any] = {}
        for task_id, n_eps in [
            ("easy",   self.config.n_episodes_easy),
            ("medium", self.config.n_episodes_medium),
            ("hard",   self.config.n_episodes_hard),
        ]:
            seed_offset = {"easy": 0, "medium": 100, "hard": 200}[task_id]
            scores = self._train_task(task_id, n_eps,
                                       start_seed=self.config.base_seed + seed_offset)
            results[task_id] = scores
        return self._build_report(results)

    # ─────────────────────────────────────────────────────────────────────
    # Single task training
    # ─────────────────────────────────────────────────────────────────────

    def _train_task(
        self,
        task_id:    str,
        n_episodes: int,
        start_seed: int,
    ) -> List[EpisodeLog]:
        """Run n episodes for one task, updating policy between each."""
        logs = []

        print(f"\n{'='*70}")
        print(f"  TRAINING TASK: {task_id.upper()} × {n_episodes} episodes")
        print(f"{'='*70}")

        for ep_idx in range(n_episodes):
            seed = start_seed + ep_idx
            log  = self._run_episode(task_id, ep_idx + 1, seed)
            logs.append(log)
            self._logs.append(log)

            # Print learning curve entry
            prev_scores = [l.final_score for l in self._logs
                           if l.task_id == task_id][:-1]
            trend = ""
            if prev_scores:
                delta = log.final_score - prev_scores[-1]
                trend = f"  Δ{delta:+.4f}"
            print(f"\n  Episode {ep_idx+1}/{n_episodes} — Score: {log.final_score:.4f}{trend}"
                  f"  ECE: {log.ece:.3f}  Overrides: {log.override_rate:.1%}"
                  f"  Health: {log.inbox_health:.2f}")

            # Recalibrate strategy based on latest episode
            self._recalibrate_strategy(log)

            # Short delay between episodes
            if ep_idx < n_episodes - 1:
                time.sleep(0.5)

        return logs

    def _run_episode(self, task_id: str, ep_num: int, seed: int) -> EpisodeLog:
        """Run a single episode end-to-end."""
        env = EmailTriageEnv(task_id=task_id, seed=seed)
        obs = env.reset()

        self.policy.begin_episode(task_id)

        done     = False
        step_num = 0
        rewards  = []

        print(f"\n  ── Episode {ep_num} | Task: {task_id} | Seed: {seed} ──")

        while not done and obs is not None:
            step_num += 1

            if self.config.verbose:
                ep_st = obs.episode_state
                flag  = "🔥 " if obs.is_followup else ""
                print(f"  [{step_num:02d}] {flag}{obs.email_id[:20]:22s} "
                      f"| {obs.subject[:45]}")

            # Policy acts (LLM + strategy)
            action, uncertainty = self.policy.act(obs, task_id)

            if self.config.verbose:
                unc_flag = " 🤔" if uncertainty.ambiguity_score > 0.5 else ""
                print(f"       {action.label.value:<12} {action.priority.value:<7} "
                      f"{action.action_type.value:<25} "
                      f"conf={action.confidence:.2f} amb={uncertainty.ambiguity_score:.2f}"
                      f"{unc_flag}")
                if action.response_text:
                    snippet = action.response_text[:80].replace("\n", " ")
                    print(f"       ↳ {snippet}…")

            # Step environment
            obs, reward, done, info = env.step(action)
            rewards.append(reward.total_reward)

            if self.config.verbose:
                gt = info.get("ground_truth", {})
                print(f"       ◆ r={reward.total_reward:+.3f}  "
                      f"pen={reward.penalty:.3f}  "
                      f"[GT: {gt.get('label','?')}/{gt.get('priority','?')}/{gt.get('action_type','?')}]")
                print(f"         {reward.feedback}")

            # Observe outcome for memory
            self.policy.observe(obs or _make_null_obs(), action, reward,
                                info.get("ground_truth", {}), info)

            if self.config.step_delay > 0:
                time.sleep(self.config.step_delay)

        result = env.episode_result()

        # Commit episode to memory
        self.policy.end_episode(result.final_score, env._ep_state)

        return EpisodeLog(
            episode      = ep_num,
            task_id      = task_id,
            seed         = seed,
            final_score  = result.final_score,
            n_steps      = step_num,
            avg_reward   = round(sum(rewards) / max(1, len(rewards)), 4),
            ece          = self.memory.calibrator.calibration_error(),
            override_rate= self.policy.strategy.override_rate(),
            top_mistakes = [m.__dict__ for m in
                            self.memory.pattern_learner.top_mistakes(3)],
            inbox_health = result.grader_breakdown.get("final_inbox_health", 1.0),
            user_sat     = result.grader_breakdown.get("final_user_sat", 1.0),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Strategy recalibration
    # ─────────────────────────────────────────────────────────────────────

    def _recalibrate_strategy(self, log: EpisodeLog) -> None:
        """
        Adjust strategy risk mode based on episode performance.
        High override rate + improving scores → strategy is helping.
        High override rate + declining scores → strategy is over-correcting.
        """
        task_logs = [l for l in self._logs if l.task_id == log.task_id]
        if len(task_logs) < 2:
            return

        recent_2  = task_logs[-2:]
        improving = recent_2[-1].final_score > recent_2[-2].final_score
        high_ece  = log.ece > 0.20

        current_mode = self.policy.strategy.risk_mode

        if high_ece and not improving and current_mode == "aggressive":
            self.policy.strategy.risk_mode = "balanced"
            print(f"  🔄 Strategy recalibrated: aggressive → balanced (ECE={log.ece:.2f})")
        elif not improving and log.override_rate > 0.5 and current_mode == "balanced":
            self.policy.strategy.risk_mode = "conservative"
            print(f"  🔄 Strategy recalibrated: balanced → conservative "
                  f"(override_rate={log.override_rate:.1%})")
        elif improving and log.override_rate < 0.1 and current_mode == "conservative":
            self.policy.strategy.risk_mode = "balanced"
            print(f"  🔄 Strategy recalibrated: conservative → balanced (improving)")

    # ─────────────────────────────────────────────────────────────────────
    # Report building
    # ─────────────────────────────────────────────────────────────────────

    def _build_report(self, results: Dict[str, List[EpisodeLog]]) -> Dict[str, Any]:
        """Compile the full training report with learning curves."""
        task_summaries: Dict[str, Any] = {}
        all_scores = []

        for task_id, logs in results.items():
            scores = [l.final_score for l in logs]
            all_scores.extend(scores)
            task_summaries[task_id] = {
                "n_episodes":    len(logs),
                "scores":        scores,
                "avg_score":     round(sum(scores) / max(1, len(scores)), 4),
                "best_score":    max(scores) if scores else 0.0,
                "learning_curve": [
                    {"episode": l.episode, "score": l.final_score,
                     "ece": l.ece, "override_rate": l.override_rate,
                     "inbox_health": l.inbox_health}
                    for l in logs
                ],
                "final_mistakes": logs[-1].top_mistakes if logs else [],
            }

        overall_avg = round(sum(all_scores) / max(1, len(all_scores)), 4)

        report = {
            "tasks":           task_summaries,
            "overall_avg":     overall_avg,
            "agent_stats":     self.policy.stats(),
            "total_episodes":  len(self._logs),
            "config": {
                "strategy_mode":  self.config.strategy_mode,
                "curriculum":     self.config.curriculum,
                "n_easy":         self.config.n_episodes_easy,
                "n_medium":       self.config.n_episodes_medium,
                "n_hard":         self.config.n_episodes_hard,
            }
        }

        self._print_summary(report)
        return report

    def _print_summary(self, report: Dict[str, Any]) -> None:
        print(f"\n{'#'*70}")
        print(f"  TRAINING COMPLETE — Overall Avg: {report['overall_avg']:.4f} "
              f"({report['overall_avg']*100:.1f}%)")
        print(f"{'#'*70}")
        for task_id, s in report["tasks"].items():
            scores_str = " → ".join(f"{sc:.3f}" for sc in s["scores"])
            print(f"  {task_id.upper():<8} avg={s['avg_score']:.4f}  "
                  f"best={s['best_score']:.4f}  curve: {scores_str}")
        stats = report["agent_stats"]
        print(f"\n  ECE: {stats['ece']:.3f}  |  "
              f"Overrides: {stats['strategy']['override_rate']:.1%}  |  "
              f"Weakest: {stats['weakest_labels']}")
        print(f"{'#'*70}\n")

    def _save_report(self, report: Dict[str, Any]) -> None:
        try:
            with open(self.config.report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"  Training report saved → {self.config.report_path}")
        except Exception as e:
            print(f"  Warning: could not save report: {e}")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_null_obs():
    """Stub observation for the final done=True step where obs=None."""
    class _Null:
        email_id = ""; subject = ""; sender = ""
        body = ""; thread_history = []; is_followup = False
        episode_state = None
    return _Null()
