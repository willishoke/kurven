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
  /** True from the first keystroke until the expression is sent or reset:
   * the engine's expression does not overwrite what is being typed. */
  editing = false;

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
    this.root = h(
      "div",
      { class: "formula-field" },
      h("label", { class: "label", for: "formula" }, "f(z) ="),
      h("div", { class: "formula" }, this.input, this.mirror),
      this.message,
    );
    this.input.addEventListener("input", () => {
      this.editing = true;
      this.clearError();
    });
    this.input.addEventListener("scroll", () => {
      this.mirror.scrollLeft = this.input.scrollLeft;
    });
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        onSubmit(this.input.value);
      } else if (e.key === "Escape") {
        this.editing = false;
        this.clearError();
        this.input.dispatchEvent(new CustomEvent("formula-reset", { bubbles: true }));
      }
    });
  }

  /** The engine's expression, unless one is being typed. */
  show(expression: string): void {
    if (this.editing && document.activeElement === this.input) return;
    if (this.input.value !== expression) this.input.value = expression;
    this.editing = false;
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
