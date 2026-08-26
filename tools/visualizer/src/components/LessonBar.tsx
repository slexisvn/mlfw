import { actions, useStore } from '../store.js';
import { LESSONS, lessonById } from '../catalog/lessons.js';
import type { Question } from '../catalog/lessons.js';

export function LessonPicker() {
  const active = useStore(s => s.lesson);
  if (active) return null;

  return (
    <label className="control lessons">
      <span>Lesson</span>
      <select
        value=""
        onChange={e => { if (e.target.value) void actions.startLesson(e.target.value); }}
      >
        <option value="">— none, I’ll poke at it myself —</option>
        {LESSONS.map(lesson => <option key={lesson.id} value={lesson.id}>{lesson.title}</option>)}
      </select>
    </label>
  );
}

export function LessonBar() {
  const progress = useStore(s => s.lesson);
  const status = useStore(s => s.status);
  const lesson = progress ? lessonById(progress.id) : null;
  if (!progress || !lesson) return null;

  const beat = lesson.beats[progress.at];
  const last = progress.at === lesson.beats.length - 1;
  const held = beat.asks !== undefined && progress.picked === null;

  return (
    <section className="lesson" aria-label={`lesson: ${lesson.title}`}>
      <header>
        <span className="lesson-title">{lesson.title}</span>
        <span className="lesson-count">{progress.at + 1} of {lesson.beats.length}</span>
        <button className="lesson-quit" onClick={() => actions.endLesson()}>leave the lesson</button>
      </header>

      <h3>{beat.title}</h3>
      <p>{beat.says}</p>

      {beat.asks && <Ask question={beat.asks} picked={progress.picked} />}

      <nav className="lesson-nav">
        <button
          disabled={progress.at === 0 || status === 'compiling'}
          onClick={() => { void actions.gotoBeat(progress.at - 1); }}
        >
          back
        </button>
        <button
          className="lesson-next"
          disabled={last || held || status === 'compiling'}
          title={held ? 'answer first — the guess is the part that teaches' : undefined}
          onClick={() => { void actions.gotoBeat(progress.at + 1); }}
        >
          {status === 'compiling' ? 'compiling…' : last ? 'that was the last one' : 'next'}
        </button>
      </nav>
    </section>
  );
}

function Ask({ question, picked }: { question: Question; picked: number | null }) {
  const answered = picked !== null;

  return (
    <div className="lesson-ask">
      <p className="lesson-question">{question.asks}</p>
      <div className="lesson-choices">
        {question.choices.map((choice, i) => (
          <button
            key={choice}
            className={choiceClass(i, picked, question.answer)}
            disabled={answered}
            onClick={() => actions.pick(i)}
          >
            {choice}
          </button>
        ))}
      </div>
      {answered && (
        <p className={picked === question.answer ? 'lesson-verdict right' : 'lesson-verdict wrong'}>
          {picked === question.answer ? 'yes — ' : 'not quite — '}
          {question.because}
        </p>
      )}
    </div>
  );
}

function choiceClass(index: number, picked: number | null, answer: number): string {
  if (picked === null) return 'choice';
  if (index === answer) return 'choice right';
  return index === picked ? 'choice wrong' : 'choice';
}
