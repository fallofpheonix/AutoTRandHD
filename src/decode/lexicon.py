"""src.decode.lexicon — word lookup and scoring for the beam search decoder.

The lexicon is a set of known historical spellings plus optional frequency
weights.  When the beam search encounters a word boundary it can query the
lexicon to boost in-vocabulary hypotheses.

File format (one entry per line)::

    word [score]

where *score* is an optional positive float (e.g. log-frequency).  If omitted
it defaults to 1.0.

Typical usage::

    from src.decode.lexicon import Lexicon

    lexicon = Lexicon.from_file("configs/lexicon.txt", vocab="abcdefghijklmnopqrstuvwxyz ...")
    print(lexicon.contains("dios"))      # True
    print(lexicon.score("dios"))         # 1.0 (or loaded frequency)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)


class Lexicon:
    """Word lookup and scoring for constrained decoding.

    Parameters
    ----------
    entries:
        Mapping from word string to score (positive float).
    """

    def __init__(self, entries: Dict[str, float]) -> None:
        self._entries: Dict[str, float] = entries

    # ------------------------------------------------------------------
    # Class constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        default_score: float = 1.0,
    ) -> "Lexicon":
        """Load a lexicon from a plain-text file.

        Parameters
        ----------
        path:
            Path to the lexicon file.  Format: ``word [score]`` per line.
            Lines starting with ``#`` are treated as comments.
        default_score:
            Score assigned to words that have no explicit score column.

        Returns
        -------
        Lexicon

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Lexicon file not found: {path}")

        entries: Dict[str, float] = {}
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                word = parts[0]
                try:
                    score = float(parts[1]) if len(parts) > 1 else default_score
                except ValueError:
                    logger.warning(
                        "Could not parse score on line %d: %r — using default.",
                        lineno, line,
                    )
                    score = default_score
                entries[word] = score

        logger.info("Loaded lexicon: %d words from %s", len(entries), path)
        return cls(entries)

    @classmethod
    def from_iterable(
        cls,
        words: Iterable[str],
        default_score: float = 1.0,
    ) -> "Lexicon":
        """Build a lexicon from an iterable of word strings.

        Parameters
        ----------
        words:
            Iterable of word strings.
        default_score:
            Uniform score assigned to all words.
        """
        return cls({w: default_score for w in words})

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def contains(self, word: str) -> bool:
        """Return ``True`` if *word* is in the lexicon."""
        return word in self._entries

    def score(self, word: str) -> float:
        """Return the score for *word*, or ``0.0`` if not found."""
        return self._entries.get(word, 0.0)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Lexicon(size={len(self._entries)})"

    def save(self, path: str | Path) -> None:
        """Persist the lexicon to a plain-text file.

        Parameters
        ----------
        path:
            Destination file.  Parent directories are created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for word, sc in sorted(self._entries.items()):
                f.write(f"{word} {sc}\n")
        logger.info("Saved lexicon (%d words) → %s", len(self._entries), path)
