# Vibe Coding — Tips for Modern Developers

> Đọc 10–15 phút.
>
> **Phần 1** là general primer — vibe-coding workflow nói chung.
> **Phần 2** là deep-dive vào **Claude Code**, AI assistant Anthropic chạy trong terminal — recommended cho lab này và mọi software work khác.

---

# Phần 1: Vibe coding nói chung

## Vibe coding là gì?

**Vibe coding** (Andrej Karpathy, 02/2025) — bạn để LLM viết phần lớn code, còn bạn đảm nhận vai *architect* và *reviewer*: mô tả intent → review diff → accept hoặc reject. Bạn không gõ từng dòng `for` loop; bạn ép spec rõ ràng và đảm bảo không có bug ngầm trong diff trả về.

Vibe coding ≠ "copy-paste từ ChatGPT". Vibe coding là một *workflow*:

```
   intent (spec)
      ↓
   prompt LLM
      ↓
   review diff (không skip!)
      ↓
   run + verify
      ↓
   commit hoặc rollback
```

Bỏ qua bất kỳ bước nào → vibe coding biến thành "gambling with code".

---

## 2 phong cách kỷ luật: SDD và TDD

### Spec-Driven Development (SDD)

> Viết **spec** trước, code sau. Spec là contract giữa bạn và LLM.

**Spec đầy đủ** thường gồm:
- *Inputs:* tên + kiểu + ràng buộc của mỗi tham số
- *Output:* shape + kiểu + invariants
- *Behavior:* edge case, lỗi, side-effects
- *Constraints:* latency budget, memory cap, dependency cấm dùng

LLM viết code khớp spec. Bạn review diff để verify từng dòng implement đúng spec. Spec mơ hồ → code mơ hồ → debug 1 giờ.

### Test-Driven Development (TDD) cho LLM era

> Viết **test** trước, code sau. Test là spec dạng máy chấm.

```
Vòng 1: "Write a pytest test for <function> that asserts <invariants>.
         Don't write the implementation yet — only the test."
```

Test pass-by-construction (bạn run, test fail vì chưa có code). Sau đó:

```
Vòng 2: "Now implement <function> such that the test passes."
```

Vibe code phần implement, nhưng **test là không đổi**. Nếu test sai (ví dụ assert sai logic), bạn phát hiện ngay từ vòng 1, không phải sau khi deploy.

TDD đặc biệt mạnh với vibe coding vì LLM hay hallucinate edge cases — tests làm bộ chống hallucination.

---

## Khi nào vibe code, khi nào tự nghĩ?

| Vibe code thoải mái | Tự nghĩ kỹ trước khi prompt |
|---|---|
| API route boilerplate (FastAPI, Express, …) | Lựa chọn algorithm / data structure cốt lõi |
| Pydantic / Zod / TypeScript schemas | Concurrency model (lock vs lock-free vs CAS) |
| Test scaffolding (pytest fixtures, mocks) | Failure semantics (retry, idempotency) |
| Config files (YAML, JSON, env) | Schema migration / backward compat |
| README skeleton, docstrings | Security boundary (auth, sandboxing) |
| Synthetic data generators / fixtures | Performance budget tradeoffs |
| Error handling cho I/O (try/except boilerplate) | Cache invalidation strategy |
| Refactor "đổi tên field X → Y" trên cả repo | Architecture (vector vs graph, monolith vs micro) |

**Quy tắc đơn giản:** nếu bug sẽ là *silent regression* (hệ thống chạy nhưng kém hơn, không lỗi rõ) thay vì *loud failure* (exception, test fail), đó là **think-hard zone**. Đừng để LLM tự quyết.

---

## 5 prompt patterns universal

### 1. Specs in, code out

> Càng narrow → cleaner diff, ít iterate hơn.

```
[VAGUE — DON'T]
"Write a function to validate emails"

[NARROW — DO]
"Function: validateEmail(addr: str) -> bool
Inputs: addr — non-empty string up to 254 chars
Output: bool — True if matches RFC 5322 simplified regex
Examples: 'user@example.com' → True; 'invalid' → False; 'a@.com' → False
Constraints: pure function, no I/O, no external libs"
```

### 2. Validate trước khi generate

> Với công thức / thuật toán: hỏi AI giải thích, cross-check, mới nhờ implement.

```
Step 1: "Explain Reciprocal Rank Fusion. Formula? Rank 0-based or 1-based? k=?"
Step 2: Bạn cross-check answer với reference (paper / docs / textbook).
Step 3: "Implement search_hybrid(...) per the formula above. rank is 1-based, k=60."
```

Nhiều AI hallucinate viết công thức sai mà code vẫn run — silent regression khó debug.

### 3. Tests trước, code sau (TDD)

> Test là spec dạng máy chấm. Viết test trước → code phải pass test.

```
"Write a pytest test that asserts X. Don't write implementation yet."
```

Sau khi test viết đúng (run pass-by-construction = fail), prompt implement.

### 4. Minimal repro → expand

> Đừng yêu cầu LLM viết toàn bộ feature trong 1 prompt. Build incrementally.

```
Step A: "Write minimal X with 1 feature."
Step B: "Run + verify."
Step C: "Now extend X to handle case Y."
Step D: "Now wrap in benchmark/test loop."
```

LLM ít hallucinate khi context đã có working baseline.

### 5. Plan → code → review loop

> 3 vòng: AI propose 3 approaches → bạn pick → AI implement → bạn review.

```
Vòng 1: "Propose 3 approaches to do X. Compare on (cost, complexity, scalability)."
Vòng 2: Bạn pick 1: "Use approach #2 because Z."
Vòng 3: "Implement approach #2 + write test."
Vòng 4: Bạn review diff line-by-line.
```

Đừng skip vòng 1 — bạn sẽ stuck trong local optimum mà LLM nghĩ ra đầu tiên.

---

## 3 anti-patterns phổ biến

### 1. Hỏi AI quyết định kiến trúc

❌ "Which embedding model should I use?"
→ AI pick default trong training data, không biết corpus của bạn.

✅ "I have a 1M-doc Vietnamese corpus, GPU=A10, latency budget=20ms. List 3 candidate embedding models with (MTEB-vi score, dim, RAM/1M vecs, cost). Recommend top 1, explain why."

### 2. Generate-and-trust không test

❌ Accept AI-written code → commit → push → discover bug in prod.

✅ AI generate → bạn run test → test pass → review diff → commit. Nếu chưa có test, viết test trước (pattern #3 ở trên).

### 3. "Make it faster" không có số

❌ "Make this latency faster"
→ AI optimize ngẫu nhiên, có thể slower.

✅ "P99 hiện tại = 87ms (measured by `<command>`). Target < 50ms. Profile shows 60% time in `<function>`. Suggest 3 optimizations with expected speedup."

### 4. Bonus — prompt thiếu context

❌ "Fix this bug"

✅ Paste exact error message + expected output + minimal repro + relevant file paths + last commit that worked. Mơ hồ in → mơ hồ out.

---

## Workflow điển hình cho 1 task

```
1. Đọc / viết spec (5 phút)         → bạn nghĩ
2. Plan: think-hard zone? (1 phút)  → bạn nghĩ
3. Prompt với spec rõ                → AI sinh
4. Review diff line-by-line          → bạn xác minh
5. Run test / benchmark              → máy verify
6. Commit hoặc rollback              → bạn quyết định
```

Không skip step 4 và 5. Đó là chỗ vibe coding fail-soft thay vì fail-loud.

---

# Phần 2: Claude Code

[Claude Code](https://code.claude.com) là agentic coding assistant của Anthropic chạy trong terminal. Nó là *vibe-coding tool* tốt nhất 2026 cho nghiêm túc software work — multi-file plans, careful edits, in-terminal plan mode + tasklist + review loop. Lab này khuyến khích dùng Claude Code (hoặc Codex CLI / OpenCode tương tự); README sẽ assume bạn có 1 trong 3.

> Tham khảo gốc: [code.claude.com/docs/en/how-claude-code-works](https://code.claude.com/docs/en/how-claude-code-works) · [memory](https://code.claude.com/docs/en/memory) · [permission-modes](https://code.claude.com/docs/en/permission-modes) · [common-workflows](https://code.claude.com/docs/en/common-workflows) · [best-practices](https://code.claude.com/docs/en/best-practices) · [claude-directory](https://code.claude.com/docs/en/claude-directory)

---

## Core concepts

### How Claude Code works — the agentic loop

Claude Code không phải chatbot trả lời rồi đợi. Nó **tự loop** qua 3 phase: **gather context → take action → verify results**, lặp lại đến khi task xong:

```
Your prompt → Claude gathers context (read files, grep)
            → Claude takes action (edit, run command)
            → Claude verifies (run tests, check output)
            → repeat until task complete
            (you can interrupt at any point)
```

Loop adapts theo task. Một câu hỏi codebase chỉ cần context. Một bug fix loop nhiều lần. Một refactor cần extensive verification. Claude tự quyết mỗi step cần gì dựa trên kết quả step trước.

Bạn cũng là phần của loop — bấm `Esc` để interrupt, type correction, Claude điều chỉnh ngay không restart.

**Components:**
- **Models**: Sonnet (default cho coding), Opus (reasoning phức tạp). Switch với `/model`.
- **Tools**: 5 categories — File ops · Search · Execution · Web · Code intelligence. Mỗi tool call trả info → feeds back loop.

### Extend Claude Code

5 mechanisms để thêm capability vào Claude Code, từ light → heavy:

| Mechanism | What it adds | Khi nào dùng |
|---|---|---|
| **CLAUDE.md** | Static instructions Claude đọc mỗi session | Project conventions, common commands, "always do X" rules |
| **Skills** (`.claude/skills/`) | Domain knowledge + repeatable workflows, load on demand | Templates, scripts, multi-step tasks invoked với `/<skill-name>` |
| **Hooks** (`.claude/settings.json`) | Shell commands chạy tự động ở lifecycle events | Format on save, lint after edit, block edits to specific paths |
| **MCP servers** | Connection tới external services | Database queries, GitHub API, Notion, Figma, Sentry |
| **Subagents** (`.claude/agents/`) | Specialized helpers với own context window | Code review, security audit, parallel research |
| **Plugins** | Bundle of skills + hooks + MCP + subagents | Install community packages — `/plugin` to browse |

**Match feature to goal:**
- "Always run X" → **hook** (deterministic)
- "Sometimes need Y" → **skill** (load on demand)
- "Connect to Z service" → **MCP**
- "Review/refactor in parallel" → **subagent** (own context)

### Explore the `.claude/` directory

Cấu trúc thư mục Claude Code đọc, ở 2 cấp:

**Project-level** (`./.claude/` — checked into git, share với team):
```
your-project/
├── CLAUDE.md                    # Project instructions, loaded every session
├── .claude/
│   ├── CLAUDE.md                # Alt location (same as ./CLAUDE.md)
│   ├── settings.json            # Permissions, hooks, MCP servers
│   ├── settings.local.json      # Personal overrides (in .gitignore)
│   ├── rules/                   # Path-scoped instructions (load when matching files open)
│   │   ├── api-design.md
│   │   └── testing.md
│   ├── agents/                  # Custom subagents
│   ├── skills/                  # Reusable skills + workflows
│   ├── commands/                # Legacy slash commands (use skills instead)
│   └── hooks/                   # Hook scripts
└── CLAUDE.local.md              # Personal project notes (in .gitignore)
```

**User-level** (`~/.claude/` — applies tới mọi project):
```
~/.claude/
├── CLAUDE.md                    # Personal preferences across projects
├── settings.json                # Personal global settings
├── skills/                      # Personal skills
├── agents/                      # Personal subagents
└── projects/<project>/memory/   # Auto memory (per-repo)
    ├── MEMORY.md                # Concise index, loaded every session
    └── <topic>.md               # Detail files, loaded on demand
```

**Loading order:** managed policy → user global → ancestor dirs → project root → working dir → CLAUDE.local.md last (closest to current dir wins).

### Explore the context window

Context window = workspace memory. Mỗi token Claude đọc consumes context. Khi đầy, Claude compacts (summarize older context). Quan trọng:

**Loaded at session start:**
- System prompt
- CLAUDE.md (full content, all ancestor levels)
- `MEMORY.md` first 200 lines / 25 KB (auto memory)
- Skill descriptions (full content loaded only when invoked)
- Tool definitions

**Grows as you work:**
- Conversation history
- File contents (mỗi `Read` tool call adds the file)
- Command outputs
- Subagent return summaries

**Strategies to manage:**
- Run `/context` — see what's using space
- `/clear` — wipe between unrelated tasks
- Subagents — exploration runs in *separate* context, you only get the summary
- Skills with `disable-model-invocation: true` — descriptions don't load until used
- `@file.py` references load file once into the conversation
- Place persistent rules in CLAUDE.md (compact-resistant), not in chat

**When context fills:** auto-compaction triggers. Add a "Compact Instructions" section to CLAUDE.md to control what's preserved. `/compact focus on the API changes` for targeted compaction.

---

## Use Claude Code

### Store instructions and memories

Hai mechanism song song:

#### CLAUDE.md (you write)
- Plain markdown file (no schema)
- Loaded at start of every session
- Run `/init` to auto-generate a starter from your codebase
- Keep under 200 lines (longer files get partially ignored)
- **Include**: build commands, code style different from defaults, testing instructions, repo etiquette, project quirks
- **Exclude**: anything Claude can read from the code, standard language conventions, file-by-file descriptions
- Test by removing a line — if Claude makes the mistake, keep it; if not, prune it

**Locations** (more specific wins):
- `~/.claude/CLAUDE.md` — global personal preferences
- `./CLAUDE.md` or `./.claude/CLAUDE.md` — project shared (commit this)
- `./CLAUDE.local.md` — personal project (gitignore this)
- Subdirectory `CLAUDE.md` — loaded on-demand when Claude reads that subdir

#### Auto memory (Claude writes)
- Lives at `~/.claude/projects/<repo>/memory/`
- Claude saves notes when you correct it or it discovers a project pattern
- Toggle on/off in `/memory`
- First 200 lines of `MEMORY.md` load each session; topic files load on-demand
- Plain markdown — edit/delete freely

#### `/memory` command
- Lists CLAUDE.md + auto memory files loaded
- Toggles auto memory
- Opens auto memory folder

#### `#` shortcut
- Type `#` at start of any message — Claude saves it as a memory entry

#### Imports with `@`
- `@README.md` inside CLAUDE.md inlines the referenced file at load time
- Use to pull in shared instructions: `@~/.claude/my-prefs.md`

### Permission modes

Press `Shift+Tab` to cycle. 6 modes total:

| Mode | What runs without asking | Best for |
|---|---|---|
| `default` | Reads only | Sensitive work, getting started |
| `acceptEdits` | Reads, file edits, common fs commands (`mkdir`, `mv`, `cp`) | Reviewing changes via `git diff` after the fact |
| `plan` | Reads only — Claude proposes a plan, no edits until you approve | Exploring before changing, multi-file features |
| `auto` | Everything (with classifier safety check) | Long uninterrupted tasks; requires Max/Team/Enterprise |
| `dontAsk` | Only pre-approved tools (allowlist) | CI pipelines, locked-down environments |
| `bypassPermissions` | Everything, no checks | **DANGEROUS** — use only in isolated containers/VMs |

**Workflow tip — `plan` mode:**
1. `Shift+Tab` twice → `plan` (read-only)
2. Ask Claude to read code + propose plan
3. Review the plan, refine via conversation
4. Approve and exit `plan` mode → Claude implements
5. Press `Esc Esc` to rewind if anything goes wrong

**Configure persistent default** in `.claude/settings.json`:
```json
{ "permissions": { "defaultMode": "acceptEdits" } }
```

**Specific allowlist** (skip prompts for trusted commands):
```json
{ "permissions": { "allow": ["Bash(npm test:*)", "Bash(git status)"] } }
```

### Common workflows

#### Explore → Plan → Code → Test
The flagship workflow:
```
[plan mode]   read /src/auth and explain how sessions work
[plan mode]   create a plan for adding Google OAuth
[default]     implement the plan, write tests, run them
[default]     commit with descriptive message and open a PR
```

#### Fix bugs efficiently
```
I'm seeing this error when I run npm test: <paste>
suggest 3 ways to fix the @ts-ignore in user.ts
update user.ts to add the null check you suggested
```

#### Refactor code
```
find deprecated API usage in our codebase
suggest how to refactor utils.js to use modern JavaScript features
refactor utils.js to use ES2024 features while maintaining same behavior
run tests for the refactored code
```

#### Work with images
- Drag-and-drop image into the CLI
- Or paste with `Ctrl+V` (NOT `Cmd+V`)
- Or pass a path: `Analyze this image: /path/to/image.png`
- Useful: paste error screenshots, UI mockups, architecture diagrams

#### Reference files with `@`
- `@src/utils/auth.js` — inlines file content
- `@src/components/` — directory listing
- `@github:repos/owner/repo/issues` — MCP resource fetch

#### Resume / fork a session
- `claude --continue` — resume most recent
- `claude --resume` — pick from list
- `/branch` or `--fork-session` — copy history into new session
- Useful when a task spans multiple sittings

#### Run parallel sessions (worktrees)
- `claude --worktree feature-auth` in terminal A
- `claude --worktree bugfix-login` in terminal B
- Each isolated checkout, no edit collisions
- Browse with `/resume` keyboard shortcuts

#### Pipe Claude into scripts
```bash
claude -p "summarize these commits" < git_log.txt
git log --oneline -20 | claude -p "what changed?"
```
Use in CI, pre-commit hooks, batch processing.

#### Slash commands
- `/init` — generate starter CLAUDE.md
- `/clear` — reset context
- `/compact <focus>` — compact with focus
- `/context` — show what's using context space
- `/permissions` — manage allow/ask/deny rules
- `/agents` — configure subagents
- `/doctor` — diagnose installation
- `/memory` — view + edit memories
- `/rewind` (or `Esc Esc`) — undo to a previous state

### Best practices

The single highest-leverage tip: **give Claude a way to verify its work.** Tests, expected outputs, screenshots — anything Claude can run to check itself.

| Before | After |
|---|---|
| "implement validateEmail" | "implement validateEmail. Test cases: 'user@example.com' → True, 'invalid' → False, 'a@.com' → False. Run the tests after." |
| "make the dashboard look better" | "[paste screenshot] implement this design. Take a screenshot of the result and compare. List differences and fix them." |
| "the build is failing" | "build fails with: <paste error>. Fix and verify it succeeds. Address root cause, don't suppress." |

**Other principles:**

1. **Explore before implementing.** Use `plan` mode for anything spanning multiple files. Skip planning only for one-line fixes.
2. **Be specific upfront.** Reference files (`@src/auth.ts`), mention constraints, point to existing patterns ("follow HotDogWidget.php's approach").
3. **Delegate, don't dictate.** Give context + intent, let Claude figure out which files to read.
4. **Course-correct early.** `Esc` to interrupt the moment Claude goes wrong. Don't wait for it to finish a wrong path.
5. **`/clear` between unrelated tasks.** Stale context = degraded performance.
6. **Use subagents for investigation.** Reading 50 files with a subagent only adds the summary to your main context.
7. **CLAUDE.md hygiene.** Keep under 200 lines. Prune ruthlessly. If Claude already does X correctly, delete the X rule.
8. **Two failed corrections rule.** If you've corrected the same issue twice, `/clear` and write a sharper initial prompt instead of correcting a third time.
9. **Trust but verify.** Plausible-looking code can hide edge cases. Always run tests / lints / type checks before merging.
10. **Use `gh` (or other CLI tools).** Tell Claude to use `gh` for GitHub, `aws` for AWS, etc. Token-efficient and auth-handled.

---

## Recommended starter setup for this lab

For Lab 22 specifically (or any Day 19/20/22 lab):

1. **Install Claude Code** — `npm install -g @anthropic-ai/claude-code` then `claude` in repo root
2. **Run `/init`** — generates a starter `CLAUDE.md` from this repo
3. **Add 3 lines** to the generated CLAUDE.md:
   ```
   - When editing notebook .py files (jupytext py:percent), preserve the `# %%` cell markers
   - Run `make verify` before suggesting submission readiness
   - VRAM math, hyperparameter values, and dataset names should match the deck (`day07-...tex`); flag deviations
   ```
4. **Use `plan` mode** (`Shift+Tab` twice) before any multi-file refactor
5. **Use `/clear`** between switching from "data prep" to "training" to "deploy"
6. **Use a subagent** for "investigate why my reward gap is negative" — keeps your main context clean

---

## CLI tool alternatives

Lab này dùng được với mọi vibe-coding CLI:

| Tool | Best at | Project file |
|---|---|---|
| **Claude Code** (Anthropic) | Multi-file plans, careful edits, longer reasoning, in-terminal plan mode | `CLAUDE.md` |
| **Codex CLI** (OpenAI) | Fast iteration, GPT/o1 family, agent mode chạy command thực | `AGENTS.md` |
| **OpenCode** (open-source) | Multi-provider (Anthropic/OpenAI/local Ollama), no vendor lock-in | `AGENTS.md` |
| **Cursor** (IDE) | GUI workflow, inline diff, multi-cursor edits | `.cursorrules` |

Đa số CLI tool đọc fallback tới `AGENTS.md` nếu không có file riêng, nên 1 file `AGENTS.md` thường đủ cho cả 3 CLI tool. Claude Code cụ thể đọc `CLAUDE.md` hoặc import `AGENTS.md` qua `@AGENTS.md`.

---

## Đọc thêm

- Andrej Karpathy — "Vibe coding" tweet (02/2025)
- Simon Willison — "Vibe coding is here, and it's pretty cool" (02/2025)
- [Claude Code docs](https://code.claude.com/docs) — official, always current
- Anthropic — "Effective coding with Claude" engineering blog
- Geoffrey Litt — "Malleable software" essay (2024) — context cho tại sao vibe coding work

