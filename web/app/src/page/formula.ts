// The formula field (LandscapeSection.swift's `expressionField`): any
// expression the language parses, sampled when return is pressed. A refused
// expression is marked where the engine says it went wrong: the reply's `at`
// counts characters as the language does (Unicode scalar values), and a
// mirror of the field's text, laid over it in the same font, underlines
// those characters.

import type { ExpressionAt } from "../engine/protocol.ts";
import { h } from "./dom.ts";

export class Formula {
  readonly root: HTMLElement;
  readonly input: HTMLInputElement;
  private readonly mirror: HTMLElement;
  private readonly message: HTMLElement;
  private readonly draft: HTMLElement;
  /** True from the first keystroke until the expression is sampled or put
   * back (escape): the engine's expression does not overwrite what is
   * typed, focused or not, while the two differ. */
  editing = false;
  /** The engine's expression, as last shown. */
  private engines = "";

  constructor(onSubmit: (text: string) => void) {
    this.input = h("input", {
      type: "text",
      id: "formula",
      spellcheck: "false",
      autocomplete: "off",
      autocapitalize: "off",
      "aria-describedby": "formula-error",
      title: "Press return to sample it. Anything the language parses: gamma(z), 1/Γ(z), zeta(z), exp(1/z), z^3 - 1",
    });
    this.mirror = h("div", { class: "mirror", "aria-hidden": "true" });
    this.message = h("p", { class: "error", id: "formula-error", role: "status" });
    this.draft = h(
      "p",
      { class: "draft-note", id: "formula-draft", hidden: true },
      "Not sampled yet: return samples it, escape puts back the landscape's.",
    );
    this.root = h(
      "div",
      { class: "formula-field" },
      h("label", { class: "label", for: "formula" }, "f(z) ="),
      h("div", { class: "formula" }, this.input, this.mirror),
      this.message,
      this.draft,
    );
    this.input.addEventListener("input", () => {
      this.editing = true;
      this.clearError();
      this.markDraft();
    });
    this.input.addEventListener("scroll", () => {
      this.mirror.scrollLeft = this.input.scrollLeft;
    });
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        onSubmit(this.input.value);
      } else if (e.key === "Escape") {
        this.revert();
      }
    });
  }

  /** The engine's expression, unless something else is typed and not yet
   * sampled: that stays, said to be unsampled, until return samples it or
   * escape puts this back. Text replaced takes its marks with it: they
   * were of the text refused. */
  show(expression: string): void {
    this.engines = expression;
    if (this.editing && this.input.value !== expression) {
      this.markDraft();
      return;
    }
    this.editing = false;
    if (this.input.value !== expression) {
      this.input.value = expression;
      this.clearError();
    }
    this.markDraft();
  }

  /** Put back the engine's expression, dropping what was typed. */
  revert(): void {
    this.editing = false;
    this.input.value = this.engines;
    this.clearError();
    this.markDraft();
  }

  /** Say whether the field holds text the engine has not sampled. */
  private markDraft(): void {
    const drafted = this.editing && this.input.value !== this.engines;
    if (drafted) this.input.dataset.draft = "true";
    else delete this.input.dataset.draft;
    this.draft.hidden = !drafted;
  }

  /** Mark where `text` went wrong. */
  showError(text: string, at: ExpressionAt | null, message: string): void {
    this.message.textContent = message;
    this.input.setAttribute("aria-invalid", "true");
    this.mirror.replaceChildren();
    if (!at || at.position === null) return;
    const chars = Array.from(text);
    const start = Math.min(at.position, chars.length);
    const end = Math.min(start + Math.max(at.length, 1), chars.length);
    // A mistake past the end ("the expression stops in the middle") marks
    // the place after the last character.
    const fault = end > start ? chars.slice(start, end).join("") : " ";
    const mark = h("mark", { "data-position": at.position, "data-length": at.length }, fault);
    mark.id = "formula-marker";
    this.mirror.append(chars.slice(0, start).join(""), mark, chars.slice(end).join(""));
    this.mirror.scrollLeft = this.input.scrollLeft;
  }

  clearError(): void {
    this.message.textContent = "";
    this.input.removeAttribute("aria-invalid");
    this.mirror.replaceChildren();
  }
}
