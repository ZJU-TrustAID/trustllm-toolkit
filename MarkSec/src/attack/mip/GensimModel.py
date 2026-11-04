import gensim.downloader
import logging

logger = logging.getLogger(__name__)

class GensimModel:
    """
    A robust and optimized wrapper for Gensim Word Vectors.
    Optimized for Watermark Removal attacks.
    """

    def __init__(self, model_name='glove-wiki-gigaword-100'):
        """
        Initialize the Gensim model.
        
        Args:
            model_name (str): Model to load. Defaults to 'glove-wiki-gigaword-100' 
        """
        self.model_name = model_name
        self.model = None
        
        # Cache dictionary: {word: [synonyms...]}
        self.simi_words_dict = {}

        self._load_model()

    def _load_model(self):
        """Internal method to load the model safely."""
        try:
            logger.info(f"[Gensim] Loading model '{self.model_name}'...")
            self.model = gensim.downloader.load(self.model_name)
            logger.info(f"[Gensim] Model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"[Gensim] Failed to load model: {e}")
            self.model = None

    def find_simi_words(self, target_word, simi_num=10):
        """
        Find similar words (synonyms) for a target word.
        
        Args:
            target_word (str): The word to replace.
            simi_num (int): Number of candidates to return.
            
        Returns:
            list: A list of synonym strings. Returns empty list if word not found.
        """
        if not self.model:
            return []

        # 1. Check cache (Cache Hit)
        # If this word was queried before, return directly without heavy vector computation
        if target_word in self.simi_words_dict:
            return self.simi_words_dict[target_word][:simi_num]

        found_words = []
        try:
            # 2. Query Gensim (Query)
            # Retrieve more candidates (topn * 2) for subsequent filtering
            candidates = self.model.most_similar(target_word, topn=simi_num * 2)
            
            for word, score in candidates:
                # 3. Simple filtering (Filtering)
                # Exclude variants of the word itself (e.g., "Apple" and "apple")
                if word.lower() == target_word.lower():
                    continue
                
                found_words.append(word)
                
                if len(found_words) >= simi_num:
                    break
            
            # 4. Cache Update (Cache Update)
            self.simi_words_dict[target_word] = found_words

        except KeyError:
            # Word not in vocabulary
            # Record an empty list in the cache to prevent errors on subsequent queries for this non-existent word
            self.simi_words_dict[target_word] = []
        except Exception as e:
            logger.warning(f"[Gensim] Unexpected error for '{target_word}': {e}")
            return []

        return found_words

if __name__=='__main__':
    import time
    
    print("Initializing...")
    start = time.time()
    gensim_model = GensimModel()
    print(f"Loaded in {time.time() - start:.2f}s")

    words = ['like', 'good', 'apple', 'computer', 'AI']
    for word in words:
        simi = gensim_model.find_simi_words(word, simi_num=5)
        print(f"{word} -> {simi}")