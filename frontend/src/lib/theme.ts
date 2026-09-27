import { useCallback, useEffect, useState } from "react";

// Light/dark theme, persisted in localStorage and applied as a data-theme attribute
// on <html>. The pre-hydration script in index.html sets the initial attribute so
// there is no flash of the wrong theme; this module keeps React in sync.

export type Theme = "light" | "dark";

const STORAGE_KEY = "drifttrace-theme";

/** Read the saved theme, defaulting to light on first visit. */
export function getInitialTheme(): Theme {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* localStorage unavailable */
  }
  return "light";
}

function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* ignore persistence errors */
  }
}

/** React hook exposing the current theme and a toggle. */
export function useTheme(): { theme: Theme; toggle: () => void } {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((t) => (t === "light" ? "dark" : "light"));
  }, []);

  return { theme, toggle };
}
