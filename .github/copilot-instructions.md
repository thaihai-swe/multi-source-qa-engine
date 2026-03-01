# Copilot Agent Operational Policy
## Documentation Protection (STRICT)

### 🚫 Restricted Actions (Default Behavior)

GitHub Copilot Agent MUST NOT create, modify, rewrite, refactor, or delete any documentation content unless the user provides explicit confirmation.

This restriction applies to ALL documentation formats, including but not limited to:

- Markdown files (`*.md`)
- Text files (`*.txt`)
- Documentation directories (`/docs`, `/documentation`)
- README files
- CHANGELOG files
- Architecture or design documents
- Code comments intended as documentation
- Inline docstrings or XML documentation comments
  - C#: `///`
  - Python: `""" docstring """`
  - Java/Kotlin: `/** */`
  - JSDoc / TSDoc comments

---

### ❗ Definition of Documentation Changes

The following actions are considered documentation modifications and are STRICTLY PROHIBITED:

- Generating new documentation files
- Editing existing documentation
- Reformatting documentation
- Auto-fixing grammar or wording
- Updating examples or explanations
- Adding/removing inline comments or docstrings
- Updating README sections automatically
- Syncing documentation with code changes

---

### ✅ Allowed Only With Explicit User Confirmation

The agent may proceed ONLY IF the user explicitly confirms using clear approval language such as:

- "Update the documentation"
- "You may modify docs"
- "Proceed with documentation changes"
- "Generate documentation now"

Implicit intent is NOT sufficient.

---

### 🔒 Required Agent Behavior

When a task would modify documentation, the agent MUST:

1. STOP execution.
2. Explain which documentation would be affected.
3. Ask for explicit confirmation.
4. Wait for user approval before proceeding.

Example response:

> This action would modify documentation files or docstrings.
> Documentation changes are restricted by policy.
> Please confirm explicitly if you want me to proceed.

---

### ⚠️ Priority Level

This rule has **HIGH PRIORITY** and overrides:
- automatic refactoring
- cleanup operations
- formatting fixes
- code generation side effects

---

### 🧠 Safe Alternatives

If documentation updates are needed but not approved, the agent should:

- Continue modifying code only
- Leave TODO comments instead of changing docs
- Provide suggested documentation updates in chat output only

---

### 📌 Compliance Requirement

Failure to follow this policy is considered a violation of agent operating rules.

The agent must always default to **NO DOCUMENTATION MODIFICATION** unless explicitly approved.
