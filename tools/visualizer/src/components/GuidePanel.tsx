import { useEffect, useRef } from 'react';
import { MARK_LEGEND, SHORTCUTS, TAB_NOTES, TERMS } from '../catalog/glossary.js';
import { actions, useStore } from '../store.js';

export function GuidePanel() {
  const open = useStore(s => s.guideOpen);
  const closeButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (open) closeButton.current?.focus();
  }, [open]);

  if (!open) return null;

  return (
    <div className="guide-backdrop" onClick={() => actions.setGuide(false)}>
      <div
        className="guide"
        role="dialog"
        aria-modal="true"
        aria-label="what am I looking at"
        onClick={event => event.stopPropagation()}
      >
        <header className="guide-head">
          <h2>What am I looking at?</h2>
          <button ref={closeButton} onClick={() => actions.setGuide(false)} aria-label="close the guide">✕</button>
        </header>

        <div className="guide-body">
          <section>
            <p className="guide-lede">
              A machine-learning compiler takes the model you wrote and turns it into one fast function.
              It does that in dozens of small steps called <strong>passes</strong>, each rewriting the
              program a little. This page runs the real compiler and stops after every single one, so you
              can watch the program change.
            </p>
            <p className="guide-lede">
              Write a model on the left, press <strong>Run</strong>, then walk down the middle column.
            </p>
          </section>

          <section>
            <h3>The marks in the pass list</h3>
            <dl className="guide-marks">
              {MARK_LEGEND.map(mark => (
                <div key={mark.glyph + mark.label}>
                  <dt className={`mark ${mark.tone}`}>{mark.glyph}</dt>
                  <dd>{mark.label}</dd>
                </div>
              ))}
            </dl>
            <p className="guide-note">
              The two numbers on each row are how many IR nodes went in and came out. The strip at the top
              of the list draws that count for every step, scaled inside each IR level so the early passes
              stay visible next to the big jump at lowering.
            </p>
          </section>

          <section>
            <h3>The five tabs</h3>
            <dl className="guide-terms">
              {TAB_NOTES.map(note => (
                <div key={note.id}>
                  <dt>{note.label}</dt>
                  <dd>{note.meaning}</dd>
                </div>
              ))}
            </dl>
          </section>

          <section>
            <h3>Words this page uses</h3>
            <dl className="guide-terms">
              {TERMS.map(term => (
                <div key={term.term}>
                  <dt>{term.term}</dt>
                  <dd>{term.meaning}</dd>
                </div>
              ))}
            </dl>
          </section>

          <section>
            <h3>Keys</h3>
            <dl className="guide-terms keys">
              {SHORTCUTS.map(shortcut => (
                <div key={shortcut.keys}>
                  <dt><kbd>{shortcut.keys}</kbd></dt>
                  <dd>{shortcut.action}</dd>
                </div>
              ))}
            </dl>
          </section>
        </div>
      </div>
    </div>
  );
}
