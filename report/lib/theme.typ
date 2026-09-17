// Shared styling and math helpers for the triple-correlation report.
//
// One show-rule entry point, since this repo has a paper and no slide deck:
//
//   report-theme(title: ..., subtitle: ..., author: ..., date: ...)
//
// ASCII only, matching the repo-wide convention that scripts/check_ascii.py
// enforces for *.py and *.md. Write "->", "<=", "+/-" rather than the
// corresponding Unicode glyphs; tests/test_report_build.py extends the same
// rule to report/*.typ, which the pre-commit hook does not reach.

#let accent = rgb("#2a5d9f")
#let ink = rgb("#1c1c1c")
#let muted = rgb("#6a6a6a")
#let rule-color = rgb("#d0d0d0")

// Font stacks. Typst emits an "unknown font family" warning for every entry it
// cannot resolve, so list only what is actually installed on the platforms
// that build this: Segoe UI and Arial ship with Windows, and DejaVu Sans Mono
// arrives with matplotlib. On a Linux box without Segoe UI, Typst falls
// through to Arial and then to its own default -- adding speculative Linux
// families here would only add warnings on the machine doing the building.
#let body-font = ("Segoe UI", "Arial")
#let mono-font = ("Consolas", "DejaVu Sans Mono")

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

// The LaTeX original is a two-column OSA journal article. This is deliberately
// a single-column US-letter report instead: reproducing the journal class
// would mean carrying its float placement rules, and the seven `figure*`
// two-column floats in the source have no meaning once there is one column.
// Every figure is simply full-width here.
#let report-theme(
  title: "",
  subtitle: "",
  authors: (),
  affiliations: (),
  email: "",
  date: "",
  body,
) = {
  set document(title: title, author: authors.map(a => a.name))
  set page(
    paper: "us-letter",
    margin: (x: 2.4cm, y: 2.4cm),
    footer: context {
      set text(size: 9pt, fill: muted)
      align(center, str(counter(page).get().first()))
    },
  )
  set text(font: body-font, size: 10.5pt, fill: ink)
  set par(justify: true, leading: 0.62em, first-line-indent: 0pt, spacing: 0.9em)
  set heading(numbering: "1.1")
  set math.equation(numbering: "(1)")
  show link: set text(fill: accent)
  show raw: set text(font: mono-font, size: 0.9em)

  // Cross-references render as the bare number, matching LaTeX's \ref{}.
  // Typst otherwise prepends a supplement ("Section", "Equation", "Figure"),
  // which doubles the word at the many sites where the source already writes
  // "Section \ref{...}" / "Fig.~\ref{...}" / "(Eq. \ref{...})" -- and the
  // source writes one at most of them. Suppressing it globally keeps the
  // rendered wording the wording the paper was published with, at every site,
  // instead of leaving 57 call sites each to be judged individually.
  set ref(supplement: none)

  show heading.where(level: 1): it => block(
    above: 1.4em,
    below: 0.6em,
    text(size: 14pt, weight: "bold", it),
  )
  show heading.where(level: 2): it => block(
    above: 1.1em,
    below: 0.5em,
    text(size: 12pt, weight: "bold", fill: accent, it),
  )
  show heading.where(level: 3): it => block(
    above: 1.0em,
    below: 0.4em,
    text(size: 11pt, weight: "bold", style: "italic", it),
  )
  show figure.caption: set text(size: 9.5pt)

  // Title block. The author superscripts and the affiliation list are the
  // OSA \authormark / \address pair, rendered as plain superscript digits so
  // the mapping from author to institution survives the class change.
  align(
    center,
    {
      v(0.2cm)
      text(size: 17pt, weight: "bold", title)
      if subtitle != "" {
        linebreak()
        v(0.2em)
        text(size: 11pt, fill: muted, subtitle)
      }
      v(0.7em)
      text(
        size: 11pt,
        authors
          .enumerate()
          .map(((i, a)) => {
            let sep = if i == 0 { "" } else if i + 1 == authors.len() {
              ", and "
            } else { ", " }
            sep + a.name + super(a.marks)
          })
          .join(),
      )
      v(0.6em)
      set text(size: 8.5pt, fill: muted)
      set par(justify: false, leading: 0.5em)
      for (i, aff) in affiliations.enumerate() {
        super(str(i + 1)) + aff
        linebreak()
      }
      if email != "" {
        v(0.2em)
        super("*") + email
      }
      v(0.4cm)
      line(length: 100%, stroke: 0.8pt + rule-color)
    },
  )

  body
}

// Abstract: unnumbered, set narrower than the body so it reads as front
// matter rather than as section 0.
#let abstract(body) = {
  block(inset: (x: 0.6cm), {
    text(weight: "bold", "Abstract. ")
    body
  })
  v(0.3em)
  line(length: 100%, stroke: 0.8pt + rule-color)
}

// Back matter (Funding, Acknowledgments, ...). LaTeX's `backmatter` /
// `\bmsection` pair has no Typst equivalent, so these are rendered as
// unnumbered bold run-in headings: smaller than a section, in source order,
// and outside the `1.1` numbering so they do not become sections 6, 7, 8.
#let bmsection(title, body) = block(
  above: 0.9em,
  below: 0.3em,
  { text(size: 10.5pt, weight: "bold", title + ". ") + body },
)

// ---------------------------------------------------------------------------
// Figures
// ---------------------------------------------------------------------------

// The leading "/" makes these paths root-relative rather than relative to
// this file. Typst resolves a relative path against the file the call
// textually lives in, which is this one, in lib/ -- so a bare "assets/..."
// would look for lib/assets/. `typst compile --root .` is run from report/,
// so "/assets/..." is report/assets/... regardless of which document imports
// this.
#let assetpath(name) = "/assets/" + name + ".png"
#let figpath(name) = "/build/figures/" + name + ".pdf"

// A committed static PNG from assets/: the published render. Seven of the
// paper's eleven figures are still these, each for a reason recorded in
// TASKS.md -- a missing generator, an unpackageable dependency, a composite
// assembled outside the repo, or a solver that does not reproduce its own
// published claim. The remaining four go through `fig` below.
#let asset(name, caption: none, width: 100%) = figure(
  image(assetpath(name), width: width),
  caption: caption,
)

// A generated figure, named exactly as the Snakefile's FIGURES list names it.
// The name must appear in both DOCUMENT_FIGURES and a `report/figures/
// plot_<name>.py`; tests/test_report_build.py checks all three sites agree.
#let fig(name, caption: none, width: 100%) = figure(
  image(figpath(name), width: width),
  caption: caption,
)

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------
//
// These mirror the \newcommand definitions at the top of the LaTeX source, so
// the math bodies translate mechanically instead of being hand-expanded at
// every one of the several hundred call sites.
//
// Four of the eight LaTeX macros are defined there but never used in the
// body -- \braket, \sech, \rd and \Lagr all appear exactly once each, on
// their own \newcommand line -- so they are deliberately not ported. Verified
// by grep over scratch/main.tex; \rd and \Lagr in particular look used from
// a naive count of the string.

// \vec{q} -> vv(q). NOT Typst's `vec()`, which builds a column vector and
// would silently render a matrix where the source meant an arrow accent.
// Wrapped in a helper with a distinct name precisely so that mistake cannot be
// made by muscle memory: `vv` has no other meaning.
#let vv(x) = math.arrow(x)

// \abs{x} -> abs(x). Typst has a built-in `abs`, but naming it here keeps the
// LaTeX-to-Typst mapping in one table and lets it auto-size like \left|.
#let abs(x) = math.abs(x)

// \bra{x} and \ket{x}. Typst 0.15 has no bra/ket builtin, and the angle
// brackets U+27E8/U+27E9 are `chevron.l`/`chevron.r` here, not `angle.l` --
// `angle` is the geometric angle and rejects an `.l` modifier. `lr()` gives
// them the \left\langle auto-sizing the LaTeX macros had.
#let bra(x) = $lr(chevron.l #x bar.v)$
#let ket(x) = $lr(bar.v #x chevron.r)$

// \iu -> the imaginary unit: an upright i with the thin space \mkern1mu adds.
// A bare `i` in Typst math is an italic identifier that can merge with an
// adjacent letter into a multi-letter symbol name, which is why this is a
// helper rather than a literal.
#let iu = $upright(i)$

// \bigstar, the cross-correlation operator. LaTeX gets it from amssymb; Typst
// has star.stroked, used here as a binary operator so the spacing matches.
#let xcorr = $class("binary", star.stroked)$
