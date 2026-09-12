import re
import spacy
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load once, reuse everywhere (spacy takes ages to start up)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


class AdaptiveMTKProcessor:
    def __init__(self, min_code_frequency=3, min_keyword_frequency=3, 
                 code_frequency_ratio=0.001, enable_broad_patterns=False,
                 use_ml_classification=True):
        """How picky to be when learning patterns from the data.

        min_code_frequency: ignore a pattern seen fewer times than this
        min_keyword_frequency: same idea, for keywords
        code_frequency_ratio: same, but scales with dataset size
        enable_broad_patterns: also try the loose patterns (noisy, off by default)
        use_ml_classification: sort keywords with ML models (needs them installed)
        """
        self.hardware_keywords = set()
        self.technical_keywords = set()
        self.error_keywords = set()
        self.procedure_keywords = set()
        
        self.platform_regexes = []
        self.code_regexes = []

        self.min_code_frequency = min_code_frequency
        self.min_keyword_frequency = min_keyword_frequency
        self.code_frequency_ratio = code_frequency_ratio
        self.enable_broad_patterns = enable_broad_patterns
        self.use_ml_classification = use_ml_classification
        
        self._classifier = None
        self._sentence_model = None
        self._initialize_ml_models()

        self._text_cache = {}
        
    def _initialize_ml_models(self):
        """Load the ML models. Whatever isn't installed just stays off."""
        if not self.use_ml_classification:
            return
            
        try:
            from transformers import pipeline
            logger.info("Loading transformers classification model...")
            self._classifier = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli",
                device=0 if self._has_cuda() else -1  # gpu if there is one, cpu otherwise
            )
            logger.info("Transformers model loaded successfully")
        except ImportError:
            logger.warning("Transformers not available")
        except Exception as e:
            logger.warning(f"Failed to load transformers: {e}")
            
        try:
            # plan B in case transformers isn't around
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence transformer model...")
            self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except ImportError:
            logger.warning("Sentence transformers not available")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
    
    def _has_cuda(self):
        """Got a GPU?"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def discover_patterns(self, df: pd.DataFrame):
        """Read through the tickets and learn which patterns and keywords matter."""
        logger.info("Discovering patterns from data...")
        
        # throw all the text into one pile
        text_columns = ['title', 'description', 'repeat_steps']
        available_columns = [col for col in text_columns if col in df.columns]
        
        all_text = [
            text for col in available_columns 
            for text in df[col].fillna('').astype(str).tolist()
        ]
        
        combined_text = ' '.join(all_text).lower()
        corpus_size = len(all_text)
        
        # bigger dataset, higher bar for keeping a pattern
        adaptive_code_freq = max(
            self.min_code_frequency, 
            int(corpus_size * self.code_frequency_ratio)
        )
        
        logger.info(f"Using adaptive code frequency threshold: {adaptive_code_freq}")
        
        self._discover_and_compile_platform_patterns(combined_text, adaptive_code_freq)
        self._discover_and_compile_code_patterns(combined_text, adaptive_code_freq)
        
        # pull out the important keywords
        self._discover_keywords(all_text, self.min_keyword_frequency)
        
        logger.info(f"Pattern discovery complete:")
        logger.info(f"  Hardware keywords: {len(self.hardware_keywords)}")
        logger.info(f"  Technical keywords: {len(self.technical_keywords)}")
        logger.info(f"  Error keywords: {len(self.error_keywords)}")
        logger.info(f"  Procedure keywords: {len(self.procedure_keywords)}")
        logger.info(f"  Platform patterns: {len(self.platform_regexes)}")
        logger.info(f"  Code patterns: {len(self.code_regexes)}")
    
    def _discover_and_compile_platform_patterns(self, text: str, min_frequency: int):
        """Find platform ids (ro.xxx, mtk_xxx, versions) worth keeping."""
        # strict ones first, these rarely misfire
        conservative_patterns = [
            r'\bro\.[a-z][a-z0-9_.]{3,}\b',     # Android properties (stricter)
            r'\bmtk[_-]?[a-z0-9]{2,}\b',        # MTK identifiers
            r'\bandroid[_.-]?[0-9]+\.[0-9]+\b', # Android versions
            r'\b[a-z]+_v?[0-9]+\.[0-9]+\b',     # Version patterns
            r'\b[A-Z]{2,}[0-9]{3,}\b',          # Chip/model codes (more digits)
            r'\bbuild[_.-][a-z0-9.]+\b',        # Build identifiers
            r'\balps\.[a-z0-9_.]+\b',           # ALPS tags like ALPS.K2.MP1
            r'\bmt[0-9]{4}[a-z]*\b',            # Chip ids like MT6896
            r'\bdimensity[ _-]?[0-9]+\b',       # Dimensity 9300
            r'\bk[a-z0-9]+_64\b',               # Board names like k689v1_64
        ]
        
        # loose ones, only if someone asks for them
        broad_patterns = [
            r'\b[a-z]{3,}[0-9]{4,}\b',          # Hardware model patterns
            r'\b[A-Z][a-z]+[0-9]{3,}\b',        # CamelCase + numbers
        ]
        
        patterns_to_test = conservative_patterns
        if self.enable_broad_patterns:
            patterns_to_test.extend(broad_patterns)
            logger.warning("Broad patterns enabled - may catch noise")
        
        for pattern in patterns_to_test:
            try:
                compiled_pattern = re.compile(pattern)
                matches = compiled_pattern.findall(text)
                unique_matches = set(matches)
                
                if len(unique_matches) >= min_frequency:
                    self.platform_regexes.append(compiled_pattern)
                    logger.debug(f"  Kept platform pattern: {pattern} ({len(unique_matches)} unique matches)")
            except re.error as e:
                logger.warning(f"Invalid regex pattern {pattern}: {e}")
    
    def _discover_and_compile_code_patterns(self, text: str, min_frequency: int):
        """Same idea but for error codes (0x..., err-12, ...)."""
        conservative_patterns = [
            r'\b0x[0-9a-f]{3,}\b',              # Hex codes (at least 3 digits)
            r'\b[a-z]+[-_][0-9]{2,}\b',         # Error codes like mtk-123, wifi_04
            r'\berr(?:or)?[-_]?[0-9]+\b',       # Error patterns
            r'\b[A-Z]{2,}[0-9]{3,}\b',          # Uppercase + numbers (stricter)
            r'\bcode[-_]?[0-9]+\b',             # Explicit code patterns
            r'\bs_ft_[a-z_]+\b',                # SP Flash Tool errors
            r'\bke_[a-z0-9_]+\b',               # Kernel exceptions
            r'\berr_[a-z]+_[0-9]+\b',           # ERR_DRM_3301 style codes
            r'\bavc\b',                         # SELinux denials
        ]
        
        # very loose, mostly garbage
        broad_patterns = [
            r'\b[0-9]{5,}\b',                   # Long numeric codes (very broad)
        ]
        
        patterns_to_test = conservative_patterns
        if self.enable_broad_patterns:
            patterns_to_test.extend(broad_patterns)
        
        for pattern in patterns_to_test:
            try:
                compiled_pattern = re.compile(pattern)
                matches = compiled_pattern.findall(text)
                unique_matches = set(matches)
                
                # demand more proof from the loose ones
                threshold = min_frequency * 2 if pattern in broad_patterns else min_frequency
                
                if len(unique_matches) >= threshold:
                    self.code_regexes.append(compiled_pattern)
                    logger.debug(f"  Kept code pattern: {pattern} ({len(unique_matches)} unique matches)")
            except re.error as e:
                logger.warning(f"Invalid regex pattern {pattern}: {e}")
    
    def _discover_keywords(self, texts: List[str], min_frequency: int):
        """Pull out the keywords that actually matter, using TF-IDF."""
        if not texts:
            logger.warning("No texts provided for keyword discovery")
            return
        
        # clean the text but keep dots and dashes, codes need them
        cleaned_texts = []
        for text in texts:
            if not text or not text.strip():
                continue
            # Preserve technical terms and hyphens, drop the rest
            cleaned = re.sub(r'[^\w\s.\-_]', ' ', text.lower())
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned:
                cleaned_texts.append(cleaned)
        
        if not cleaned_texts:
            logger.warning("No valid texts after cleaning")
            return
        
        # tuned for tech text: keeps dotted terms, drops boilerplate
        vectorizer = TfidfVectorizer(
            max_features=2000,
            min_df=max(2, min_frequency),  # ignore one-offs
            max_df=0.8,  # drop words that show up everywhere
            ngram_range=(1, 3),  # single words plus short phrases
            stop_words='english',
            token_pattern=r'\b[a-zA-Z0-9][a-zA-Z0-9._\-]*\b'  # keeps ro.xxx style terms together
        )
        
        try:
            logger.info("Running TF-IDF analysis...")
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # average score per term
            mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            
            # best first
            feature_scores = list(zip(feature_names, mean_scores))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"TF-IDF extracted {len(feature_scores)} features")
            
            # sort the top 300 into buckets
            self._categorize_keywords(feature_scores[:300])  # Top 300 features
            
        except Exception as e:
            logger.error(f"TF-IDF analysis failed: {e}")
            # TF-IDF blew up, just count words instead
            self._simple_frequency_analysis(texts, min_frequency)
    
    def _categorize_keywords(self, feature_scores: List[Tuple[str, float]]):
        """Put each keyword in a bucket: hardware, technical, error or procedure."""
        if not feature_scores:
            return
        
        # Easy ones go through the word lists first. Free and instant.
        leftovers = []
        for feature, score in feature_scores:
            if not self._rule_place(feature, score):
                leftovers.append((feature, score))

        if not leftovers:
            return

        # Whatever the lists couldn't place goes to BART.
        logger.info(f"{len(leftovers)} keywords left for the model...")
        if self._classifier is not None:
            self._categorize_with_transformers(leftovers)
        # ...or the lighter one if that's all there is.
        elif self._sentence_model is not None:
            self._categorize_with_similarity(leftovers)
        else:
            logger.info("No ML models available, leftovers stay unplaced")
    
    def _categorize_with_transformers(self, feature_scores: List[Tuple[str, float]]):
        """Sort keywords with the zero-shot model. Slow but accurate."""
        # short labels tested better than full sentences
        categories = ["hardware", "technology", "error", "procedure"]
        
        logger.info("Classifying keywords using transformers model...")
        
        # go in batches so it doesn't crawl
        batch_size = 50
        for i in range(0, len(feature_scores), batch_size):
            batch = feature_scores[i:i + batch_size]
            
            for feature, score in batch:
                if score < 0.01:  # not worth the model's time
                    continue

                # junk lengths
                if len(feature) < 2 or len(feature) > 50:
                    continue

                try:
                    result = self._classifier(feature, categories,
                                            hypothesis_template="This text is about {}.")

                    # take its best guess
                    top_label = result['labels'][0]
                    confidence = result['scores'][0]
                    
                    # only keep it if the model is sure
                    if confidence > 0.5:
                        self._assign_keyword_category(feature, top_label)
                        
                except Exception as e:
                    logger.debug(f"Failed to classify '{feature}': {e}")
                    continue
    
    def _categorize_with_similarity(self, feature_scores: List[Tuple[str, float]]):
        """Same sorting but with embedding similarity. Faster, dumber."""
        # one reference sentence per bucket
        category_prototypes = {
            'hardware': "hardware component device part sensor camera wifi bluetooth display audio speaker microphone battery chip processor memory storage",
            'technical': "technical process method algorithm procedure timeout bandwidth allocation buffer cache optimization configuration setting parameter",
            'error': "error failure crash fault exception hang freeze timeout malfunction bug issue problem failure broken",
            'procedure': "procedure action step instruction reboot restart reset flash update install remove enable disable configure setup"
        }
        
        # embed the references once
        prototype_embeddings = {}
        for category, text in category_prototypes.items():
            prototype_embeddings[category] = self._sentence_model.encode(text)
        
        logger.info("Classifying keywords using similarity matching...")
        
        features_to_classify = [f for f, s in feature_scores if s >= 0.01 and 2 <= len(f) <= 50]

        if features_to_classify:
            # embed everything in one go
            feature_embeddings = self._sentence_model.encode(features_to_classify)
            
            for feature, embedding in zip(features_to_classify, feature_embeddings):
                try:
                    # nearest bucket wins
                    similarities = {}
                    for category, proto_embedding in prototype_embeddings.items():
                        similarity = np.dot(embedding, proto_embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(proto_embedding)
                        )
                        similarities[category] = similarity
                    
                    best_category = max(similarities, key=similarities.get)
                    best_score = similarities[best_category]

                    # skip weak matches
                    if best_score > 0.25:
                        category_map = {
                            'hardware': "hardware component or device",
                            'technical': "technical process or method",
                            'error': "error failure or malfunction", 
                            'procedure': "procedure action or step"
                        }
                        self._assign_keyword_category(feature, category_map[best_category])
                        
                except Exception as e:
                    logger.debug(f"Failed to classify '{feature}': {e}")
                    continue
    
    def _compile_rule_res(self):
        """Build the word-list patterns once, reuse after that."""
        if self.__dict__.get("_rule_res") is None:
            hardware_patterns = [
                r'\b(camera|wifi|bluetooth|display|audio|speaker|mic|battery|sensor|chip|cpu|gpu|ram|storage|memory|npu|fingerprint|gps|gnss|drm|widevine|ufs|vibrator|haptic|gyro|barometer|sim|esim|dsp|emi|ddr|modem|touch|ois|nfc|charger|usb|rild|bootloader|antenna)\b',
                r'\b[a-z]*cam[a-z]*\b', r'\b[a-z]*wifi[a-z]*\b', r'\b[a-z]*audio[a-z]*\b'
            ]
            error_patterns = [
                r'\b(error|fail|crash|hang|freeze|timeout|fault|exception|bug|issue|problem|broken|panic|watchdog|denied|selinux|abort|downgrade|mismatch|violation|overheat|throttl|stuck)\b',
                r'\berr\b', r'\bfail\b'
            ]
            procedure_patterns = [
                r'\b(reboot|restart|reset|flash|update|install|remove|enable|disable|config|setup|start|stop|calibrate|reflash|sideload|reprovision|reenroll|relock|provision|audit|wipe|backup|restore|recalibrate)\b',
                r'\b(step|action|procedure|instruction|guide|method|process)\b'
            ]
            self.__dict__["_rule_res"] = (
                re.compile('|'.join(hardware_patterns)),
                re.compile('|'.join(error_patterns)),
                re.compile('|'.join(procedure_patterns)),
            )
        return self.__dict__["_rule_res"]

    def _rule_place(self, feature: str, score: float) -> bool:
        """Try the word lists. True if the keyword found a home."""
        if score < 0.02:
            return False
        hw_regex, err_regex, proc_regex = self._compile_rule_res()
        feature_lower = feature.lower()
        if hw_regex.search(feature_lower):
            self.hardware_keywords.add(feature)
        elif err_regex.search(feature_lower):
            self.error_keywords.add(feature)
        elif proc_regex.search(feature_lower):
            self.procedure_keywords.add(feature)
        else:
            # no list claims it, the model gets a shot at it
            return False
        return True

    def _categorize_with_rules(self, feature_scores: List[Tuple[str, float]]):
        """No models around? Just match against word lists."""
        logger.info("Using rule-based keyword classification")

        for feature, score in feature_scores:
            self._rule_place(feature, score)
    
    def _assign_keyword_category(self, feature: str, category_label: str):
        """Drop a keyword into the right bucket."""
        if "hardware" in category_label.lower():
            self.hardware_keywords.add(feature)
        elif "technology" in category_label.lower() or "technical" in category_label.lower():
            self.technical_keywords.add(feature)
        elif "error" in category_label.lower():
            self.error_keywords.add(feature)
        elif "procedure" in category_label.lower():
            self.procedure_keywords.add(feature)
    
    def _simple_frequency_analysis(self, texts: List[str], min_frequency: int):
        """Last resort: rank by raw word counts."""
        word_counts = Counter()
        
        for text in texts:
            if text and text.strip():
                words = re.findall(r'\b[a-zA-Z0-9][a-zA-Z0-9._\-]*\b', text.lower())
                word_counts.update(words)

        # frequent plus reasonable length usually means a real term
        common_words = [
            (word, count/100.0) for word, count in word_counts.items() 
            if count >= min_frequency and 2 < len(word) <= 30
        ]
        
        logger.info(f"Frequency analysis found {len(common_words)} candidate terms")
        
        # then sort them with the word lists
        self._categorize_with_rules(common_words[:200])
    
    # skips work we've already done
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Grab a saved result if we have one."""
        return self._text_cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, result: Dict):
        """Save a result. Kicks out the oldest ones when full."""
        if len(self._text_cache) > 1000:
            # make room
            oldest_keys = list(self._text_cache.keys())[:100]
            for key in oldest_keys:
                del self._text_cache[key]
        
        self._text_cache[cache_key] = result
    
    def extract_platform_info(self, text: str) -> List[str]:
        """Pull platform ids out of a piece of text."""
        if not text or not self.platform_regexes:
            return []
        
        platforms = set()
        text_lower = text.lower()
        
        for compiled_regex in self.platform_regexes:
            try:
                matches = compiled_regex.findall(text_lower)
                platforms.update(matches)
            except Exception as e:
                logger.debug(f"Regex matching failed: {e}")
                continue
        
        return list(platforms)
    
    def extract_codes(self, text: str) -> List[str]:
        """Pull error codes out of a piece of text."""
        if not text or not self.code_regexes:
            return []
        
        codes = set()
        text_lower = text.lower()
        
        for compiled_regex in self.code_regexes:
            try:
                matches = compiled_regex.findall(text_lower)
                codes.update(matches)
            except Exception as e:
                logger.debug(f"Regex matching failed: {e}")
                continue
        
        return list(codes)
    
    def categorize_hardware_module(self, text: str) -> str:
        """Guess which hardware the ticket is about."""
        if not text or not self.hardware_keywords:
            return "unknown"

        text_lower = text.lower()

        # longest match first, so "camera sensor" beats "cam"
        for keyword in sorted(self.hardware_keywords, key=len, reverse=True):
            if keyword in text_lower:
                return keyword
        
        return "unknown"
    
    def extract_technical_keywords(self, text: str) -> Set[str]:
        """Pick up any known tech or error terms in the text."""
        if not text:
            return set()

        text_lower = text.lower()
        keywords = set()

        for term in self.technical_keywords:
            if term in text_lower:
                keywords.add(f"tech_{term}")
        
        for error_type in self.error_keywords:
            if error_type in text_lower:
                keywords.add(f"error_{error_type}")
        
        return keywords
    
    def preprocess_title_enhanced(self, title: str) -> Dict[str, any]:
        """Clean up a title and pull out what's interesting."""
        if not title or not title.strip():
            return self._empty_title_result()

        # seen this one before?
        cache_key = f"title_{hash(title)}"
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        title_lower = title.lower()
        
        # pull out the structured bits
        platforms = self.extract_platform_info(title)
        codes = self.extract_codes(title)
        hw_module = self.categorize_hardware_module(title)
        tech_keywords = self.extract_technical_keywords(title)

        # scrub punctuation, but keep the characters codes need
        clean_title = re.sub(r'[^\w\s\.\-_]', ' ', title_lower)
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()

        tokens = [t for t in clean_title.split() if len(t) > 1 and t.isalnum()]

        # glue the findings back in as extra tokens
        feature_tokens = []
        feature_tokens.extend([f"platform_{p}" for p in platforms])
        feature_tokens.extend([f"code_{c}" for c in codes])
        
        if hw_module != "unknown":
            feature_tokens.append(f"module_{hw_module}")
        
        tokens.extend(feature_tokens)
        tokens.extend(list(tech_keywords))
        
        result = {
            'text': " ".join(sorted(set(tokens))),
            'platforms': platforms,
            'codes': codes,
            'hw_module': hw_module,
            'tech_keywords': list(tech_keywords)
        }
        
        # save for next time
        self._cache_result(cache_key, result)
        return result
    
    def _empty_title_result(self) -> Dict[str, any]:
        """Blank title placeholder."""
        return {
            'text': "",
            'platforms': [],
            'codes': [],
            'hw_module': "unknown",
            'tech_keywords': []
        }
    
    def preprocess_steps_enhanced(self, steps: str) -> Dict[str, any]:
        """Same deal for the repeat-steps field."""
        if not steps or not steps.strip():
            return self._empty_steps_result()
        
        # Check cache
        cache_key = f"steps_{hash(steps)}"
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        steps_lower = steps.lower().replace("→", " ").replace("->", " ")
        
        # spot known actions like reboot or flash
        procedures = []
        for keyword in self.procedure_keywords:
            if keyword in steps_lower:
                procedures.append(f"proc_{keyword}")
        
        # scrub and split
        clean_steps = re.sub(r'[^a-zA-Z0-9\s\-_.]', ' ', steps_lower)
        clean_steps = re.sub(r'\s+', ' ', clean_steps).strip()

        tokens = [t for t in clean_steps.split() if len(t) > 1 and t.replace('_', '').replace('-', '').isalnum()]
        tokens.extend(procedures)

        # how involved is this procedure?
        step_separators = len(re.findall(r'[→\->\n\.•]', steps))
        sequence_length = max(step_separators, len(procedures), 1)
        
        result = {
            'text': " ".join(sorted(set(tokens))),
            'procedures': procedures,
            'sequence_length': sequence_length
        }
        
        self._cache_result(cache_key, result)
        return result
    
    def _empty_steps_result(self) -> Dict[str, any]:
        """Blank steps placeholder."""
        return {
            'text': "",
            'procedures': [],
            'sequence_length': 1
        }
    
    def preprocess_description_enhanced(self, descriptions: List[str]) -> List[Dict[str, any]]:
        """Descriptions go through spacy (slow), so batch them in one pass."""
        if not descriptions:
            return []
        
        results = []
        uncached_descriptions = []
        uncached_indices = []
        
        # figure out which ones still need doing
        for i, desc in enumerate(descriptions):
            if not desc or not desc.strip():
                results.append(self._empty_description_result())
                continue
            
            cache_key = f"desc_{hash(desc)}"
            cached = self._get_cached_result(cache_key)
            
            if cached:
                results.append(cached)
            else:
                results.append(None)  # filled in below
                uncached_descriptions.append(desc)
                uncached_indices.append(i)
        
        # run the rest through spacy together
        if uncached_descriptions:
            logger.debug(f"Processing {len(uncached_descriptions)} uncached descriptions")
            
            for doc_idx, doc in enumerate(nlp.pipe(uncached_descriptions, batch_size=50)):
                try:
                    # keep real words, lemmatized
                    tokens = []
                    for token in doc:
                        if (not token.is_stop and 
                            len(token.text) > 2 and 
                            token.text.replace('_', '').replace('-', '').isalnum()):
                            tokens.append(token.lemma_.lower())
                    
                    # same structured bits as titles
                    desc_text = uncached_descriptions[doc_idx]
                    platforms = self.extract_platform_info(desc_text)
                    codes = self.extract_codes(desc_text)
                    hw_module = self.categorize_hardware_module(desc_text)
                    tech_keywords = self.extract_technical_keywords(desc_text)
                    
                    # Add feature tokens
                    feature_tokens = []
                    feature_tokens.extend([f"platform_{p}" for p in platforms])
                    feature_tokens.extend([f"code_{c}" for c in codes])
                    
                    if hw_module != "unknown":
                        feature_tokens.append(f"module_{hw_module}")
                    
                    tokens.extend(feature_tokens)
                    tokens.extend(list(tech_keywords))

                    result = {
                        'text': " ".join(sorted(set(tokens))),
                        'platforms': platforms,
                        'codes': codes,
                        'hw_module': hw_module,
                        'tech_keywords': list(tech_keywords),
                        'token_count': len(set(tokens))
                    }
                    
                    # save it and slot it into place
                    original_index = uncached_indices[doc_idx]
                    cache_key = f"desc_{hash(uncached_descriptions[doc_idx])}"
                    self._cache_result(cache_key, result)
                    results[original_index] = result
                    
                except Exception as e:
                    logger.warning(f"Failed to process description {doc_idx}: {e}")
                    results[uncached_indices[doc_idx]] = self._empty_description_result()
        
        return results
    
    def _empty_description_result(self) -> Dict[str, any]:
        """Blank description placeholder."""
        return {
            'text': "",
            'platforms': [],
            'codes': [],
            'hw_module': "unknown", 
            'tech_keywords': [],
            'token_count': 0
        }
    
    def get_stats(self) -> Dict[str, any]:
        """Quick summary of what got learned."""
        return {
            'hardware_keywords': len(self.hardware_keywords),
            'technical_keywords': len(self.technical_keywords),
            'error_keywords': len(self.error_keywords),
            'procedure_keywords': len(self.procedure_keywords),
            'platform_patterns': len(self.platform_regexes),
            'code_patterns': len(self.code_regexes),
            'cache_size': len(self._text_cache),
            'ml_models_available': {
                'transformers': self._classifier is not None,
                'sentence_transformer': self._sentence_model is not None
            }
        }
    
    def process_mediatek_tickets(self, csv_path: str) -> pd.DataFrame:
        """Run the whole thing: read the csv, clean everything, build search text."""
        logger.info(f"Loading data from {csv_path}…")
        df = pd.read_csv(csv_path)

        logger.info("Preprocessing titles…")
        df['title_processed'] = df['title'].fillna('').astype(str).apply(self.preprocess_title_enhanced)

        logger.info("Preprocessing steps…")
        df['steps_processed'] = df['repeat_steps'].fillna('').astype(str).apply(self.preprocess_steps_enhanced)

        logger.info("Preprocessing descriptions…")
        descriptions = df['description'].fillna('').astype(str).tolist()
        df['desc_processed'] = self.preprocess_description_enhanced(descriptions)

        logger.info("Creating enhanced_text field…")
        df['enhanced_text'] = df.apply(self.create_enhanced_features, axis=1)

        # handy columns for poking at the data
        df['hardware_module'] = df['title_processed'].map(lambda x: x['hw_module'])
        df['platform_count']   = df['title_processed'].map(lambda x: len(x['platforms']))
        df['procedure_count']  = df['steps_processed'].map(lambda x: len(x['procedures']))
        df['sequence_complexity'] = df['steps_processed'].map(lambda x: x['sequence_length'])

        logger.info(f"Finished processing {len(df)} tickets")
        return df

    def create_enhanced_features(self, row: Dict[str, any]) -> str:
        """One searchable blob per ticket. Same builder for old and new tickets."""
        title = row['title_processed']
        steps = row['steps_processed']
        desc = row['desc_processed']
        parts = [
            f"[TITLE] {title['text']}",
            f"[STEPS] {steps['text']}",
            f"[DESC] {desc['text']}"
        ]
        hw = title['hw_module']
        if hw and hw != "unknown":
            parts.append(f"[MODULE] {hw}")
        platforms = title['platforms']
        if platforms:
            parts.append(f"[PLATFORM] {' '.join(platforms)}")
        sequence_length = steps['sequence_length']
        if sequence_length > 1:
            parts.append(f"[SEQ_LEN] {sequence_length}")
        return " ".join(parts)

    def clear_cache(self):
        """Empty the cache."""
        self._text_cache.clear()
        logger.info("Processing cache cleared")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MediaTek support tickets")
    parser.add_argument("--input", default="mediatek_tickets.csv")
    parser.add_argument("--output", default="enhanced_mediatek_tickets.csv")
    parser.add_argument("--skip-pattern-discovery", action="store_true")
    args = parser.parse_args()

    processor = AdaptiveMTKProcessor()
    if not args.skip_pattern_discovery:
        processor.discover_patterns(pd.read_csv(args.input))
    df_processed = processor.process_mediatek_tickets(args.input)
    df_processed.to_csv(args.output, index=False)
    logger.info(f"Saved {args.output}")
