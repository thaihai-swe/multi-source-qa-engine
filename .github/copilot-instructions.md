
## GitHub Copilot Custom Instructions

### **Documentation Protocol**

You are strictly prohibited from generating, modifying, or deleting documentation files (e.g., `.md`, `.txt`, or inline docstrings) without explicit user confirmation.

**Follow these steps for every documentation-related task:**

1. **Identify the Need:** If a code change requires a documentation update, or if the user asks for a new document.
2. **Propose, Don't Execute:** Describe what documentation you intend to create or change. Provide a brief outline or summary of the content.
3. **The Confirmation Trigger:** You must end your response with a clear question:
> *"Would you like me to proceed with creating/updating this documentation? (Yes/No)"*


4. **Wait for Input:** Do not generate the actual file content or write to the filesystem until the user responds with "Yes" or an equivalent affirmation.
5. **Respect User Decisions:** If the user responds with "No" or any negative response, do not proceed with documentation generation. Instead, ask if they would like to modify the proposed outline or if they have specific requirements.
