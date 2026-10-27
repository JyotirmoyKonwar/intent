# SkillAlignBench: Annotation Guideline (draft v1)

Status: **first draft, 2026-10-28.** Written from [PLAN.md](PLAN.md) §4 and §5.6. It will be revised after the pilot on 20 skills (Days 4–5). Items marked **open** are not decided yet.

This guideline tells an annotator (a person, or the pre-annotation prompt in `bench/preannotate.py`) how to turn one skill folder into intents, behaviors and findings. It uses the schema in PLAN.md §5.5.

---

## 1. Definitions

**Skill.** A folder `S = (D, E, M)`. `D` is the declarative content: `SKILL.md` frontmatter and body, README, other docs, and comments. `E` is the executable content: source files, scripts, config, and fenced code that the agent is told to run. `M` is metadata: platform, repo, commit, license.

**Intent unit.** One atomic functional claim that the skill makes about itself. It is merged across every declarative source that states it. Each intent has an `action`, an `object`, `constraints[]`, a `polarity` (positive or negative), and at least one source span with a verbatim quote.
- A **negative claim** ("never uploads data", "read-only", "works offline") is an intent with `polarity: negative`. It is the strongest signal of divergence.
- A claim is an intent only if it is about what the skill does. "Fast and reliable" is not an intent. "Saves results only to `./out`" is.

**Behavior unit.** One observable effect of the skill, at the granularity of (capability, target, call site). Each behavior has a `capability` from the taxonomy in PLAN.md §4.3, an `origin` (`code`, `agent-instruction`, `config`, or `dependency`), a `reachability` (`entry`, `reachable`, or `dormant`), evidence spans, and a call chain where one exists.
- An **agent-instruction** behavior is text in `SKILL.md` or another doc that tells the agent to do something (read a file, call a host, run a command, hide an action from the user). It counts as a behavior even though no code runs.
- A **composite** behavior is a source → transform → sink path across files, for example `ENV_READ → ENCODE_OBFUSCATE → NET_OUTBOUND`. A composite is a separate unit that points to its parts. A finding that involves a composite refers to the composite.

**Fulfillment.** A set of behaviors `B'` fulfills intent `i` when the combined effect achieves `i.action` on `i.object` and respects every constraint in `i.constraints`. Fulfillment is about effect, not wording. Similar words do not count as fulfillment, and different words do not count as divergence.

**Misalignment types.**

| Type | Meaning | Mark it when |
|---|---|---|
| **UI** Unimplemented Intent | No behavior set fulfills the intent | Nothing in the skill achieves `action` on `object` |
| **UI-partial** | Some constraints or variants are met, not all | The code covers one of several promised variants (e.g. S3 but not GCS) |
| **UB** Undeclared Behavior | A reachable behavior that no intent covers and that the implied table (§3) does not list | The skill does something its description never mentions and its category would not imply |
| **SD** Semantically Divergent Behavior | A behavior linked to an intent that differs in object, scope, destination, condition or magnitude, or that contradicts a negative claim | The intent says "saves locally" and the code POSTs remotely |

Record the **divergence axis** for every SD: `object`, `scope`, `destination`, `condition`, `magnitude`, or `negative-claim`.

**Dormant.** A behavior whose code cannot run from any entry point, or a fenced example that the surrounding text does not ask the agent to run. Dormant behaviors are annotated and flagged `dormant`. They do not count as UB and are reported separately.

**Severity.** `info`, `low`, `medium`, `high`, or `critical`. It is computed by the rules in PLAN.md §6 (D4) from the capability, the destination, whether the behavior is composite, and signs of obfuscation. Annotators do not set severity by hand. They record `risk` only where the rules cannot be applied.

**Intent suspected.** For every finding, one of `drift`, `deliberate`, or `unclear`. This separates benign drift from disguise. It is an annotator judgment and must come with a rationale.

---

## 2. Decision procedure

Apply it per skill, in this order. Record a rationale for every finding.

```
For each intent i:
   any behavior set that achieves i.action on i.object?           no  → UI
   all constraints and variants met?                              no  → UI-partial
   (negative intent) does any reachable behavior contradict it?   yes → SD (negative-claim)

For each reachable behavior b:
   linked to some intent i (same purpose)?
       yes → same object, scope, destination, condition, magnitude as i says?   no  → SD (record axis)
             contradicts any negative claim?                                     yes → SD (negative-claim)
       no  → implied by the skill's category (§3)?                               yes → declared, no finding
                                                                                 no  → UB

Dormant behaviors → annotate, flag `dormant`, no UB, reported separately.
```

**Linking a behavior to an intent.** A behavior is linked to an intent when it serves the same purpose, even if the wording differs. "Fetches the page" and "downloads a document from a URL" are linked. "Fetches the page" and "formats Markdown" are not.

**Which behaviors are reachable.** A behavior is `reachable` when an entry point (the agent running the skill, a script the skill names, a hook, or a command in `SKILL.md` the agent is told to run) can reach it. It is `entry` when it is the entry point. The rule for fenced code in `SKILL.md` is **open** (see §5).

---

## 3. Implied-behavior table (v1)

A behavior on this list does not need to be declared for a skill in that category. It is still UB or SD if it appears in the "Still counts" column.

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

The table is v1. The pilot may add rows or move entries. Every change is logged with the skill that caused it.

---

## 4. Edge cases

Decided in advance:

- **Third-party libraries.** Behavior is analyzed at the API boundary. `requests.post` is NET_OUTBOUND. Library internals are not analyzed. A dependency that is suspicious in itself (a typosquat, a post-install hook) becomes a `config` behavior.
- **Vendored code.** Minified or vendored third-party files (for example `*.umd.js`) are treated like third-party libraries. Annotators do not read them line by line. A suspicious sign in them is a `config` or `dependency` behavior.
- **Conditional and time-triggered behavior.** A reachable behavior counts even when it is behind a condition. The condition is recorded in `conditions[]`.
- **Obfuscated code we cannot resolve.** Annotated as `ENCODE_OBFUSCATE`. This is UB unless the skill declares it.
- **Inline code the agent is told to run.** Executable content, even when it is in `SKILL.md`.
- **Inline code shown only as an example.** Not an instruction to run. Annotated as dormant unless the surrounding text tells the agent to run it. The official pdf skill has 12 Python examples of this kind.
- **Untagged fences.** A fence with no language tag is still text. If it holds shell commands the agent is told to run, it is executable content. The normalizer treats an untagged fence as `shell` when its first command looks like a shell command (`git`, `npm`, `./x.sh`, and similar). Other untagged fences stay unknown.
- **Other fence tags** (`markdown`, `diff`, `text`) are documentation, not code. They are not annotated as behavior unless the surrounding text tells the agent to act on them.
- **Granularity.** One behavior per (capability, target, call site). Composites point to their parts. Findings that involve a composite refer to the composite.
- **Several intents, one behavior.** One behavior can fulfill several intents. Link it to each one. It is one behavior, not several.
- **One intent, several behaviors.** The intent is fulfilled when the behaviors together achieve the action. Record each behavior in `alignment` for that intent.
- **Negative claims with no code.** A negative claim with no behavior that contradicts it is not a finding. It stays an intent with status `fulfilled` if the skill has no contrary behavior.
- **Vague documentation.** A skill whose docs say very little has few intents. Its behaviors then mostly fall into UB or are declared by the implied table. That is a real result, not a reason to add intents the docs do not state. (Covered by `VAGUE_DOC` in the injected split.)
- **Canary hosts.** Any host in a benchmark example or injected skill is `canary.local` or another reserved name. No real host is contacted.

---

## 5. Open items

These are not decided. Each one has a day on which it will be decided.

1. **Fenced code in SKILL.md: example or instruction? (Day 4, M2.)** Proposal: a virtual file is `reachable` only when the text around it tells the agent to run it. Otherwise it is `dormant`. Without this, UB precision suffers on skills with many examples.
2. **Untagged fences hold shell commands. (Fixed 2026-10-28.)** The normalizer now treats an untagged fence whose first command is a known shell word as `shell`. On the `branch-review` skill this adds the six `git grep` and `git ls-files` fences in `references/scoring-evidence-guide.md`. Unrecognized untagged fences are still skipped.
3. **Unmapped fence tags.** Five fences in the `impeccable` skill are tagged `text`, `markdown` or `diff` and are dropped silently. Probably correct, but the decision should be written down.
4. **Category assignment.** The category list is in PLAN.md §5.6 and §5.5. The classifier (Day 3) assigns it. The annotator can override it, and the override is recorded.
5. **Severity for UB without a risk signal.** Default to `low` until the pilot shows whether that is right.

---

## 6. Revision log

- **v1 (2026-10-28):** first draft from PLAN.md §4 and §5.6, plus the normalizer check on 7 real skills.
