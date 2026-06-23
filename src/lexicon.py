import logging
from typing import Dict, Tuple, Set
import requests
import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)

# Fallback basic lexicons for safety if URL fetching is blocked/offline
DEFAULT_POS_LEXICON = {
    "bagus": 3, "senang": 4, "hebat": 4, "cinta": 5, "setuju": 3, "baik": 3,
    "lolos": 4, "sukses": 4, "semangat": 4, "terima kasih": 5, "membantu": 3,
    "bangga": 4, "alhamdulillah": 5, "untung": 3, "mudah": 3, "menang": 4,
}

DEFAULT_NEG_LEXICON = {
    "jelek": -3, "marah": -4, "benci": -5, "kecewa": -4, "gagal": -4,
    "susah": -3, "ribet": -3, "buruk": -3, "rugi": -3, "lemah": -2,
    "mahal": -2, "salah": -2, "mengecewakan": -5, "amburadul": -4,
}


class LexiconLabeler:
    """Performs lexicon-based sentiment analysis for Indonesian text.
    
    Downloads or uses fallback sentiment lexicons (InSet Fajri et al.) to compute
    sentiment score:
        score = sum(positive_weights) - sum(abs(negative_weights))
    And assigns a categorical label:
        score > 0  -> Positif
        score < 0  -> Negatif
        score == 0 -> Netral
    """
    
    def __init__(
        self,
        pos_url: str = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv",
        neg_url: str = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"
    ):
        """Initializes the lexicon sets and resolves conflicts."""
        logger.info("Initializing LexiconLabeler...")
        
        self.lexicon_pos: Dict[str, int] = self._load_tsv_lexicon(pos_url, is_positive=True)
        self.lexicon_neg: Dict[str, int] = self._load_tsv_lexicon(neg_url, is_positive=False)
        
        # Resolve conflicts: words that appear in both positive and negative lists
        self._resolve_conflicts()
        
    def _load_tsv_lexicon(self, url: str, is_positive: bool) -> Dict[str, int]:
        """Loads a TSV lexicon from a URL with custom formatting and fallback."""
        try:
            logger.info(f"Loading lexicon from {url}...")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            
            df = pd.read_csv(
                StringIO(r.text),
                sep="\t",
                header=None,
                names=["word", "weight"]
            )
            
            # Formatting and cleaning
            df["word"] = df["word"].astype(str).str.lower().str.strip()
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
            df = df.dropna(subset=["weight"])
            df["weight"] = df["weight"].astype(int)
            
            mapping = dict(zip(df["word"], df["weight"]))
            logger.info(f"Loaded {len(mapping)} words from {url}.")
            return mapping
            
        except Exception as e:
            logger.warning(f"Failed to load lexicon from URL ({e}). Using local fallback.")
            return DEFAULT_POS_LEXICON if is_positive else DEFAULT_NEG_LEXICON
            
    def _resolve_conflicts(self) -> None:
        """Removes words that appear in both positive and negative lexicons."""
        conflicts: Set[str] = set(self.lexicon_pos.keys()) & set(self.lexicon_neg.keys())
        if conflicts:
            logger.info(f"Resolving {len(conflicts)} conflicting words appearing in both positive and negative lexicons...")
            for word in conflicts:
                self.lexicon_pos.pop(word, None)
                self.lexicon_neg.pop(word, None)
            logger.info("Conflict resolution finished.")
            
    def label_sentiment(self, text: str) -> Tuple[int, str]:
        """Calculates the sentiment score and determines the label class.
        
        Args:
            text: A preprocessed string of text.
            
        Returns:
            Tuple[int, str]: (sentiment_score, sentiment_label)
        """
        if not isinstance(text, str) or not text.strip():
            return 0, "Netral"
            
        score = 0
        words = text.split()
        
        for token in words:
            score += self.lexicon_pos.get(token, 0)
            score -= abs(self.lexicon_neg.get(token, 0))
            
        if score > 0:
            return score, "Positif"
        elif score < 0:
            return score, "Negatif"
        else:
            return 0, "Netral"
