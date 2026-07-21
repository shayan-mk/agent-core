# Agent-Core Self-Evolving Skill Bank

> Design based on `upstream/develop`, ReSkill v2
> (2026-06-08), SkillOS v1 (2026-05-07), and SkillRL v1 (2026-02-09).

## Executive Decision

Extend agent-core's existing skill lifecycle with immutable whole-bank versions,
controlled candidate creation, and a reward-based adoption gate. SkillBank uses
the same local packages, `SkillUseRail`, `SkillTool`, and agent assembly as
ordinary agents. It is not a second skill format, registry, or runtime loader.

Use agent-core's batch/offline RL path for the first ReSkill integration. Despite
its name, that path generates fresh same-task rollout groups before each GRPO
update. It therefore owns the version assignment, rollout evidence, and training
boundary needed for old/new bank evaluation. VERL continues to own PPO/GRPO
optimization; SkillBank does not alter its advantage or actor-update logic.

Local, workspace, team, imported, community, sharing-hub, and
JiuwenSwarm-produced packages can seed the initial bank after they are
materialized locally. SkillBank does not synchronize remote sources or write
back to them.

ReSkill supplies the co-evolution and adoption protocol. SkillRL informs the
general/task retrieval hierarchy, and SkillOS contributes BM25 retrieval only.
SkillOS curator training and SkillRL cold-start SFT are not implemented.

```text
existing skill sources -> effective local view -> immutable bank versions
  -> ReSkill creator -> grouped policy/skill co-evolution and adoption
  -> existing SkillUseRail and SkillTool
```

## 1. Agent-Core Foundation

### 1.1 Skill Runtime and Sources

An agent-core skill is a directory containing `SKILL.md` and optional scripts,
references, assets, and evolution files. `SkillUseRail` discovers packages from
one or more local roots. In `all` mode it puts enabled skill descriptions in the
prompt; in `auto_list` mode it exposes `list_skill` for progressive discovery.
`SkillTool` reads the selected package through the agent's `SysOperation`.

Skills currently arrive through several paths:

- harness workspace and team-workspace roots;
- caller-provided local roots through the single-agent skill API;
- one-shot GitHub downloads through `RemoteSkillUtil`;
- locally cached AutoHarness community repositories;
- sharing-hub package downloads; and
- task packages and `SkillDelta` output materialized in JiuwenSwarm workspaces.

There is no online authoritative skill repository. All runtime consumption is
local. Remote repositories and hubs are import sources, not live subscriptions
or the system of record.

For multiple roots, `SkillUseRail` keeps the first package with a duplicate
directory name. The harness also unions disabled entries from each root's
`skills_state.json`. A materialized bank must preserve both behaviors. Core's
`SkillManager`/`SkillUtil` is an in-memory live-agent registry; it does not
persist versions, proposals, or promotion evidence.

### 1.2 Existing Evolution

`SkillEvolutionRail` collects trajectory evidence and uses
`SkillExperienceOptimizer` to propose `EvolutionRecord` updates.
`ExperienceManager` stages, scores, approves, and applies those records through
`EvolutionStore`. Approved records update the managed experience projection in
`SKILL.md` and `evolution/` details.

This machinery is useful creator infrastructure, but it is mutable and
per-skill. `EvolutionArchiveService` stores per-skill rollback pairs, and the
sharing format omits evolution/archive state. Neither provides an immutable
whole-bank version that old and new rollouts can mount concurrently.

The ReSkill loop does not invoke this mutable evolution path. Its creator emits
package-level ADD/MODIFY/DELETE operations against an isolated bank copy.

### 1.3 RL Stack

Agent-core's batch/offline path is:

```text
MainTrainer
  -> TrainingCoordinator creates rollout.n attempts per source task
  -> RuntimeExecutor runs one DeepAgent per attempt
  -> reward is attached to RolloutMessage
  -> BaseVerlTrainingExecutor performs the PPO/GRPO update
```

`rollout.n` defaults to 8. `RLTask` represents one trajectory attempt;
`RolloutMessage` is its task-level result; `rollout_info` contains one `Rollout`
per LLM call. The encoder groups training data by `origin_task_id`.

With `whole_trajectory=False`, each LLM turn becomes a separately weighted
training sample. The ReSkill integration requires `whole_trajectory=True` so
one bank observation corresponds to one GRPO sample and environment/tool tokens
are masked.

Agent-core imports VERL in its dataset, worker, converter, and training-executor
layers. The integration keeps that boundary unchanged. The production/online
path asynchronously trains from delayed samples and defaults to one rollout per
item, so it does not construct comparable old/new groups and is not the initial
ReSkill controller.

## 2. Paper-Derived Direction

| Method | Mechanism | Reported setup | Use in agent-core |
|---|---|---|---|
| [ReSkill](https://arxiv.org/abs/2606.01619) | An assertion-driven creator proposes ADD/MODIFY/DELETE. Old and new banks share GRPO rollout groups; discounted Thompson Sampling allocates traffic and a reward gate accepts or rejects each candidate. | Qwen3-4B/8B policies, Claude 4.5 Sonnet creator, one or two 8xA100 nodes depending on benchmark. | Primary co-evolution and adoption protocol. |
| [SkillRL](https://arxiv.org/abs/2602.08234) | A teacher distills successes and counterfactual lessons from failures. General skills are always supplied and up to six task skills are retrieved. Cold-start SFT teaches skill use; low-performing categories trigger bank updates during GRPO. | Qwen2.5-7B executor, OpenAI o3 teacher, 8 H100 GPUs. | Retrieval hierarchy only. Cold-start SFT and unconditional bank union are excluded. |
| [SkillOS](https://arxiv.org/abs/2605.06614) | A trainable curator edits a Markdown SkillRepo for grouped task streams. BM25 retrieves skills and curator GRPO combines task, call-validity, content-quality, and compression rewards. | Qwen3-8B curator/executor, Qwen3-32B judge, 16 H100 GPUs. | BM25 retrieval only; no separate curator is implemented. |

ReSkill's unchanged rollout count does not mean zero additional compute: creator
calls, diagnosis, trigger validation, larger prompts, and bank I/O still add
cost. SkillOS trains the curator with RL; ReSkill and SkillRL use prompted
creator/teacher models while RL updates the executor policy.

## 3. Design Requirements

1. **Reuse the existing lifecycle.** Keep agent-core packages, source adapters,
   `SkillUseRail`, `SkillTool`, and agent assembly, and preserve evolution state
   in complete package snapshots.
2. **Version the complete effective bank.** Include support files, evolution
   state, and `skills_state.json`; preserve source order and disabled state;
   exclude transient `.git/` and `__pycache__/` data.
3. **Isolate writes and reads.** Candidate creation occurs in a writable copy.
   Published versions are read-only. Each rollout gets one explicit version,
   one fresh workspace, one rail, and one scoped `SysOperation`.
4. **Use controlled outcome evidence.** Allocate old and new banks within the
   same rollout workload. Environment reward is the adoption signal;
   infrastructure or reward-pipeline failures are not task evidence.
5. **Audit every decision.** Record parent and candidate versions, operations,
   source trajectories, creator identity, provenance, evidence, and decision.
6. **Assign before agent construction.** Store the version on `RLTask`, carry it
   through both `RolloutMessage` outcomes and persistence, and observe completed
   messages before training-batch filters.
7. **Keep policy and bank responsibilities separate.** VERL owns PPO/GRPO.
   Agent-core owns source materialization, version assignment, evidence,
   candidate creation, and promotion.
8. **Control active-bank changes.** Imported and shared packages are bootstrap
   inputs. After initialization, only an accepted ReSkill proposal changes the
   active bank.

## 4. Target Architecture

```text
ordered local sources
  -> initialize_skill_bank (materialize, snapshot, activate)
  -> writable candidate copy
  -> SkillBankCandidateBuilder snapshot (candidate)
  -> grouped Thompson allocation and reward evidence
  -> promote/reject + proposal evidence
  -> existing SkillUseRail -> existing SkillTool -> DeepAgent
```

### 4.1 Versions and Sources

`SkillBankStore` provides immutable snapshots, explicit/active resolution,
atomic promotion and rollback, proposal audit, and conservative garbage
collection. A content manifest identifies every snapshot. `ACTIVE` selects the
ordinary inference default; A/B rollouts resolve explicit versions and never
switch the global pointer.

```text
<bank_root>/
  ACTIVE
  VERSION_COUNTER
  .skill_bank.lock
  versions/bank_000001/{manifest.json, skills/...}
  proposals/<proposal-id>.json
```

`materialize_effective_skill_bank` creates one canonical first-root-wins view
and records per-package provenance. The store contains no Git client, community
discovery, or remote synchronization. Versions are complete bank copies so they
can be mounted independently and audited directly.

The RL factory mounts a resolved version's read-only `skills/` root directly and
creates a separate writable rollout workspace. Both roots are allowed by the
rollout-owned `SysOperation`. `RuntimeExecutor` unregisters rails, tears down
agent-owned tools, removes the system operation, and deletes the workspace after
every rollout outcome.

### 4.2 Candidate Creation

The candidate creation path is:

```text
SkillBankCandidateBuilder.build(proposal)   -> immutable candidate
SkillBankCreatorPipeline(parent, reservoir) -> package candidate
```

`SkillBankCandidateBuilder` copies the parent, applies the creator's declared
package operations, validates the final package set, records provenance, and
snapshots the result. DELETE affects only the candidate copy.

The first-party ReSkill creator groups retained episodes by stable task key and
prioritizes groups containing contrasting outcomes and versions. It performs
contrastive diagnosis, grades deterministic trajectory assertions, considers
proposal history, and emits package ADD/MODIFY/DELETE operations. Diagnosis and
authoring use the same configured creator model sequentially, avoiding another
resident LLM.

Authored packages declare `scope`, `when_to_use`, `trigger_type`, and an optional
`trigger_pattern`. Action-pattern triggers must fire on at least half of retained
episodes before building a candidate; failed validation is returned to the
creator for a bounded retry. This checks that a trigger is usable, not that the
candidate is beneficial. Reward evidence remains the promotion gate.

### 4.3 Evaluation and Adoption

During GRPO, a centralized `SkillBankAdoptionCycle` assigns baseline/candidate
versions before construction and consumes complete `RolloutMessage` evidence.
A configured threshold converts float rewards to Bernoulli success.
Retained trajectories feed the creator; exact per-cycle counts feed the bandit.

Allocation uses ReSkill's Beta(1,1) priors per cycle, a 0.15 exploration floor
per arm, and discounted updates:

```text
alpha <- w * alpha + successes
beta  <- w * beta  + observations - successes
w      = (1 + observations / M)^-1
```

The controller accepts when `E[p_candidate] > E[p_baseline]`, keeps a bounded
trajectory reservoir (default 200), and can estimate memory `M` from completed
adoption windows by predictive likelihood. It checkpoints the trained policy
before a bank decision, updates the active pointer and audit record, then
schedules the next candidate.

### 4.4 Retrieval and Triggers

Retrieval and triggering remain filters on the existing rail. Optional BM25
retrieval produces an `enabled_skills` allow-list; packages explicitly marked
`scope: general` are always included. An optional per-model-call selector then
applies deterministic triggers:

- `general`: visible on every model call;
- `beginning`: visible on the first model call; and
- `action_pattern`: visible after a matching prior tool action.

Packages without trigger metadata retain general behavior. `when_to_use` is
included in the ordinary skill description. The selected view controls prompt
and `SkillTool` exposure; no alternate rail, tool, or skill model is introduced.

## 5. Agent-Core Integration

| Area | Integration |
|---|---|
| Harness assembly | Shared source resolution and optional explicit skill roots feed the existing `SkillUseRail`. Normal agents and RL agents use the same `resolve_deep_agent_parts` path. |
| RL task/runtime | `RLTask` and `RolloutMessage` carry version and stable task keys. The built-in factory resolves one version before constructing the agent. |
| Evidence | `TrainingCoordinator` observes messages before classifier/validator filtering. `FileRolloutStore` persists version labels. |
| Training boundary | `MainTrainer` commits evidence only after a successful policy step and invokes the adoption cycle without changing GRPO internals. |
| Creation | `OfflineRLOptimizer.set_skill_creator()` configures the ReSkill creator used by the adoption cycle. |
| Bootstrap | Existing source adapters materialize packages locally before `initialize_skill_bank()` creates and activates the initial version. |

The version label is evaluation metadata, not a GRPO feature, so
`RLBatchBuilder` and VERL data structures do not need bank-specific logic.

## 6. Phased Plan

The phases are architectural dependency boundaries, not implementation-status
reports. Later phases retain the guarantees established earlier.

### P0: SkillBank Substrate

- Immutable whole-bank snapshots, active/explicit resolution, promotion,
  rollback, proposal audit, and garbage collection.
- Read-only version roots, isolated writable candidates and rollout workspaces,
  and teardown of rollout-owned global resources.

Exit gate: snapshots are reproducible; old and candidate versions run
concurrently without state leakage; promotion and rollback are atomic.

### P1: Source Materialization and Candidates

- Effective-root materialization with ordinary harness precedence and state.
- Package candidates built from an isolated parent copy through the ReSkill
  creator and shared candidate builder.
- Complete provenance and proposal audit before a candidate enters RL rollout
  allocation.

Exit gate: static-bank behavior matches ordinary loading, and candidate
construction cannot mutate the active version.

### P2: ReSkill in Batch/Offline GRPO

- Version identity from task creation through runtime, persistence, and audit.
- Thompson assignment for each rollout attempt and one bank observation per
  whole trajectory.
- Central reservoir, discounted Thompson allocation, decision hook,
  and scheduled ReSkill creator around the existing rollout/train loop.

Exit gate: version exposure is recorded, both arms meet the configured minimum
observations, and policy training retains its rollout budget.

### P3: Creation and Retrieval

- Contrastive success/failure diagnosis, deterministic assertion grading,
  proposal history, and trigger validation.
- ReSkill package ADD/MODIFY/DELETE through the built-in creator.
- BM25 allow-list retrieval and deterministic per-call triggers through the
  existing rail.

Exit gate: creation or retrieval improves held-out accuracy/cost without
bypassing the version gate.

## 7. Ownership and Non-Goals

- Agent-core owns local source materialization, bank versions, candidates,
  version assignment, evidence, and promotion.
- VERL owns PPO/GRPO optimization; this design does not modify its algorithms.
- Existing source adapters, `SkillManager`, `SkillUseRail`, and `SkillTool`
  remain the intake and runtime foundation.
- No remote repository is the active system of record, and no automatic process
  writes evolution back to a public source.
- A dedicated SkillOS curator, cold-start SFT, arbitrary proposal plugins, and
  automatic remote-source refresh are not implemented.
