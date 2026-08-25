import { useCallback, useEffect, useRef, useState } from 'react';

export type Size = { width: number; height: number };

export type ElementSize<T extends Element> = {
  size: Size;
  ref: (node: T | null) => void;
};

export function useElementSize<T extends Element>(fallback: Size): ElementSize<T> {
  const [size, setSize] = useState<Size>(fallback);
  const node = useRef<T | null>(null);

  const apply = useCallback((width: number, height: number) => {
    if (width <= 0 || height <= 0) return;
    setSize(current => (current.width === width && current.height === height ? current : { width, height }));
  }, []);

  const ref = useCallback((next: T | null) => {
    node.current = next;
    if (!next) return;
    const box = next.getBoundingClientRect();
    apply(box.width, box.height);
  }, [apply]);

  useEffect(() => {
    const target = node.current;
    if (!target) return;

    const remeasure = () => {
      const box = target.getBoundingClientRect();
      apply(box.width, box.height);
    };

    const observer = new ResizeObserver(entries => {
      const box = entries[0].contentRect;
      apply(box.width, box.height);
    });
    observer.observe(target);
    window.addEventListener('resize', remeasure);
    remeasure();

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', remeasure);
    };
  }, [apply]);

  return { size, ref };
}
