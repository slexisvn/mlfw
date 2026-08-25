const APPLE = /mac|iphone|ipad|ipod/i;

export const IS_APPLE = typeof navigator !== 'undefined' && APPLE.test(navigator.userAgent);

export const RUN_SHORTCUT = IS_APPLE ? '⌘⏎' : 'Ctrl+⏎';

export const RUN_SHORTCUT_WORDS = IS_APPLE ? 'Cmd + Enter' : 'Ctrl + Enter';
