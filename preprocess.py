import re
import spacy
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load NLP model once at module level
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


class AdaptiveMTKProcessor:
    def __init__(self, min_code_frequency=3, min_keyword_frequency=3, 
                 code_frequency_ratio=0.001, enable_broad_patterns=False,
                 use_ml_classification=True):
        """
        Args:
            min_code_frequency: Minimum absolute frequency for code patterns
            min_keyword_frequency: Minimum frequency for keyword extraction
            code_frequency_ratio: Minimum frequency as ratio of corpus size
            enable_broad_patterns: Whether to enable potentially noisy broad patterns
            use_ml_classification: Whether to use ML models for classification
        """
        self.hardware_keywords = set()
        self.technical_keywords = set()
        self.error_keywords = set()
        self.procedure_keywords = set()
        
        # Pre-compiled regex patterns
        self.platform_regexes = []
        self.code_regexes = []
        
        # Configuration
        self.min_code_frequency = min_code_frequency
        self.min_keyword_frequency = min_keyword_frequency
        self.code_frequency_ratio = code_frequency_ratio
        self.enable_broad_patterns = enable_broad_patterns
        self.use_ml_classification = use_ml_classification
        
        # ML models - load once and reuse
        self._classifier = None
        self._sentence_model = None
        self._initialize_ml_models()
        
        # Cache for processed texts
        self._text_cache = {}
        
    def _initialize_ml_models(self):
        """Initialize ML models once at startup"""
        if not self.use_ml_classification:
            return
            
        try:
            # Initialize transformers classifier
            from transformers import pipeline
            logger.info("Loading transformers classification model...")
            self._classifier = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli",
                device=0 if self._has_cuda() else -1  # Use GPU if available
            )
            logger.info("Transformers model loaded successfully")
        except ImportError:
            logger.warning("Transformers not available")
        except Exception as e:
            logger.warning(f"Failed to load transformers: {e}")
            
        try:
            # Initialize sentence transformer as fallback
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence transformer model...")
            self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except ImportError:
            logger.warning("Sentence transformers not available")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
    
    def _has_cuda(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def discover_patterns(self, df: pd.DataFrame):
        """Automatically discover patterns and keywords from the data"""
        logger.info("Discovering patterns from data...")
        
        # Combine all text for pattern discovery - more efficient
        text_columns = ['title', 'description', 'repeat_steps']
        available_columns = [col for col in text_columns if col in df.columns]
        
        # Use list comprehension for better performance
        all_text = [
            text for col in available_columns 
            for text in df[col].fillna('').astype(str).tolist()
        ]
        
        combined_text = ' '.join(all_text).lower()
        corpus_size = len(all_text)
        
        # Calculate adaptive frequency thresholds
        adaptive_code_freq = max(
            self.min_code_frequency, 
            int(corpus_size * self.code_frequency_ratio)
        )
        
        logger.info(f"Using adaptive code frequency threshold: {adaptive_code_freq}")
        
        # Discover and compile patterns in parallel if possible
        self._discover_and_compile_platform_patterns(combined_text, adaptive_code_freq)
        self._discover_and_compile_code_patterns(combined_text, adaptive_code_freq)
        
        # Discover domain-specific keywords using TF-IDF
        self._discover_keywords(all_text, self.min_keyword_frequency)
        
        logger.info(f"Pattern discovery complete:")
        logger.info(f"  Hardware keywords: {len(self.hardware_keywords)}")
        logger.info(f"  Technical keywords: {len(self.technical_keywords)}")
        logger.info(f"  Error keywords: {len(self.error_keywords)}")
        logger.info(f"  Procedure keywords: {len(self.procedure_keywords)}")
        logger.info(f"  Platform patterns: {len(self.platform_regexes)}")
        logger.info(f"  Code patterns: {len(self.code_regexes)}")
    
    def _discover_and_compile_platform_patterns(self, text: str, min_frequency: int):
        """Discover platform/system identifiers and compile regex patterns"""
        # More specific and targeted patterns
        conservative_patterns = [
            r'\bro\.[a-z][a-z0-9_.]{3,}\b',     # Android properties (stricter)
            r'\bmtk[_-]?[a-z0-9]{2,}\b',        # MTK identifiers
            r'\bandroid[_.-]?[0-9]+\.[0-9]+\b', # Android versions
            r'\b[a-z]+_v?[0-9]+\.[0-9]+\b',     # Version patterns
            r'\b[A-Z]{2,}[0-9]{3,}\b',          # Chip/model codes (more digits)
            r'\bbuild[_.-][a-z0-9.]+\b',        # Build identifiers
        ]
        
        # Broader patterns (use with caution)
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
        """Discover error codes and identifiers and compile regex patterns"""
        conservative_patterns = [
            r'\b0x[0-9a-f]{3,}\b',              # Hex codes (at least 3 digits)
            r'\b[a-z]+[-_][0-9]{2,}\b',         # Error codes like mtk-123, wifi_04
            r'\berr(?:or)?[-_]?[0-9]+\b',       # Error patterns
            r'\b[A-Z]{2,}[0-9]{3,}\b',          # Uppercase + numbers (stricter)
            r'\bcode[-_]?[0-9]+\b',             # Explicit code patterns
        ]
        
        # Very broad patterns (high noise risk)
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
                
                # Apply stricter filtering for broad numeric patterns
                threshold = min_frequency * 2 if pattern in broad_patterns else min_frequency
                
                if len(unique_matches) >= threshold:
                    self.code_regexes.append(compiled_pattern)
                    logger.debug(f"  Kept code pattern: {pattern} ({len(unique_matches)} unique matches)")
            except re.error as e:
                logger.warning(f"Invalid regex pattern {pattern}: {e}")
    
    def _discover_keywords(self, texts: List[str], min_frequency: int):
        """Use TF-IDF to discover important keywords with better preprocessing"""
        if not texts:
            logger.warning("No texts provided for keyword discovery")
            return
        
        # More sophisticated text cleaning
        cleaned_texts = []
        for text in texts:
            if not text or not text.strip():
                continue
            # Preserve technical terms and hyphens, remove other punctuation
            cleaned = re.sub(r'[^\w\s.\-_]', ' ', text.lower())
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned:
                cleaned_texts.append(cleaned)
        
        if not cleaned_texts:
            logger.warning("No valid texts after cleaning")
            return
        
        # Enhanced TF-IDF with better parameters
        vectorizer = TfidfVectorizer(
            max_features=2000,  # Increased for better coverage
            min_df=max(2, min_frequency),  # At least 2 occurrences
            max_df=0.8,  # Remove very common terms
            ngram_range=(1, 3),  # Include trigrams for technical terms
            stop_words='english',
            token_pattern=r'\b[a-zA-Z0-9][a-zA-Z0-9._\-]*\b'  # Better technical term matching
        )
        
        try:
            logger.info("Running TF-IDF analysis...")
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF scores
            mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            
            # Sort features by importance
            feature_scores = list(zip(feature_names, mean_scores))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"TF-IDF extracted {len(feature_scores)} features")
            
            # Categorize keywords based on context and patterns
            self._categorize_keywords(feature_scores[:300])  # Top 300 features
            
        except Exception as e:
            logger.error(f"TF-IDF analysis failed: {e}")
            # Fallback to simple frequency analysis
            self._simple_frequency_analysis(texts, min_frequency)
    
    def _categorize_keywords(self, feature_scores: List[Tuple[str, float]]):
        """Categorize discovered keywords using the pre-loaded classifier"""
        if not feature_scores:
            return
        
        # Try transformers first (most accurate)
        if self._classifier is not None:
            self._categorize_with_transformers(feature_scores)
        # Fall back to sentence transformers
        elif self._sentence_model is not None:
            self._categorize_with_similarity(feature_scores)
        # Final fallback to simple methods
        else:
            logger.info("No ML models available, using rule-based classification")
            self._categorize_with_rules(feature_scores)
    
    def _categorize_with_transformers(self, feature_scores: List[Tuple[str, float]]):
        """Categorize using pre-loaded transformers classifier"""
        categories = [
            "hardware component or device",
            "technical process or method", 
            "error failure or malfunction",
            "procedure action or step"
        ]
        
        logger.info("Classifying keywords using transformers model...")
        
        # Process in batches for efficiency
        batch_size = 50
        for i in range(0, len(feature_scores), batch_size):
            batch = feature_scores[i:i + batch_size]
            
            for feature, score in batch:
                if score < 0.01:  # Skip very low-scoring features
                    continue
                
                # Skip very short or very long terms
                if len(feature) < 2 or len(feature) > 50:
                    continue
                    
                # Create context for better classification
                context = f"In mobile device troubleshooting context: {feature}"
                
                try:
                    # Classify the word/phrase
                    result = self._classifier(context, categories, 
                                            hypothesis_template="This term is related to {}")
                    
                    # Get the top prediction with confidence
                    top_label = result['labels'][0]
                    confidence = result['scores'][0]
                    
                    # Only keep high-confidence classifications
                    if confidence > 0.5:  # Reasonable threshold
                        self._assign_keyword_category(feature, top_label)
                        
                except Exception as e:
                    logger.debug(f"Failed to classify '{feature}': {e}")
                    continue
    
    def _categorize_with_similarity(self, feature_scores: List[Tuple[str, float]]):
        """Categorize using pre-loaded sentence transformer"""
        # Define better category prototypes
        category_prototypes = {
            'hardware': "hardware component device part sensor camera wifi bluetooth display audio speaker microphone battery chip processor memory storage",
            'technical': "technical process method algorithm procedure timeout bandwidth allocation buffer cache optimization configuration setting parameter",
            'error': "error failure crash fault exception hang freeze timeout malfunction bug issue problem failure broken",
            'procedure': "procedure action step instruction reboot restart reset flash update install remove enable disable configure setup"
        }
        
        # Get embeddings for prototypes once
        prototype_embeddings = {}
        for category, text in category_prototypes.items():
            prototype_embeddings[category] = self._sentence_model.encode(text)
        
        logger.info("Classifying keywords using similarity matching...")
        
        # Process in batches
        features_to_classify = [f for f, s in feature_scores if s >= 0.01 and 2 <= len(f) <= 50]
        
        if features_to_classify:
            # Batch encode all features at once
            feature_embeddings = self._sentence_model.encode(features_to_classify)
            
            for feature, embedding in zip(features_to_classify, feature_embeddings):
                try:
                    # Calculate similarity with each prototype
                    similarities = {}
                    for category, proto_embedding in prototype_embeddings.items():
                        similarity = np.dot(embedding, proto_embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(proto_embedding)
                        )
                        similarities[category] = similarity
                    
                    # Get best match
                    best_category = max(similarities, key=similarities.get)
                    best_score = similarities[best_category]
                    
                    # Only keep reasonably confident matches
                    if best_score > 0.25:  # Lower threshold for similarity
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
    
    def _categorize_with_rules(self, feature_scores: List[Tuple[str, float]]):
        """Rule-based categorization as final fallback"""
        # Define rule-based patterns
        hardware_patterns = [
            r'\b(camera|wifi|bluetooth|display|audio|speaker|mic|battery|sensor|chip|cpu|gpu|ram|storage|memory)\b',
            r'\b[a-z]*cam[a-z]*\b', r'\b[a-z]*wifi[a-z]*\b', r'\b[a-z]*audio[a-z]*\b'
        ]
        
        error_patterns = [
            r'\b(error|fail|crash|hang|freeze|timeout|fault|exception|bug|issue|problem|broken)\b',
            r'\berr\b', r'\bfail\b'
        ]
        
        procedure_patterns = [
            r'\b(reboot|restart|reset|flash|update|install|remove|enable|disable|config|setup|start|stop)\b',
            r'\b(step|action|procedure|instruction|guide|method|process)\b'
        ]
        
        # Compile patterns
        hw_regex = re.compile('|'.join(hardware_patterns))
        err_regex = re.compile('|'.join(error_patterns))
        proc_regex = re.compile('|'.join(procedure_patterns))
        
        logger.info("Using rule-based keyword classification")
        
        for feature, score in feature_scores:
            if score < 0.02:  # Higher threshold for rule-based
                continue
                
            feature_lower = feature.lower()
            
            if hw_regex.search(feature_lower):
                self.hardware_keywords.add(feature)
            elif err_regex.search(feature_lower):
                self.error_keywords.add(feature)
            elif proc_regex.search(feature_lower):
                self.procedure_keywords.add(feature)
            else:
                # Default to technical if it has reasonable score
                if score > 0.03:
                    self.technical_keywords.add(feature)
    
    def _assign_keyword_category(self, feature: str, category_label: str):
        """Helper method to assign features to appropriate keyword sets"""
        if "hardware" in category_label.lower():
            self.hardware_keywords.add(feature)
        elif "technical" in category_label.lower():
            self.technical_keywords.add(feature)
        elif "error" in category_label.lower():
            self.error_keywords.add(feature)
        elif "procedure" in category_label.lower():
            self.procedure_keywords.add(feature)
    
    def _simple_frequency_analysis(self, texts: List[str], min_frequency: int):
        """Simple frequency analysis fallback"""
        word_counts = Counter()
        
        for text in texts:
            if text and text.strip():
                # Better tokenization
                words = re.findall(r'\b[a-zA-Z0-9][a-zA-Z0-9._\-]*\b', text.lower())
                word_counts.update(words)
        
        # Get frequent words that are likely technical terms
        common_words = [
            (word, count/100.0) for word, count in word_counts.items() 
            if count >= min_frequency and 2 < len(word) <= 30
        ]
        
        logger.info(f"Frequency analysis found {len(common_words)} candidate terms")
        
        # Use rule-based classification
        self._categorize_with_rules(common_words[:200])
    
    # Caching methods for better performance
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached processing result"""
        return self._text_cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, result: Dict):
        """Cache processing result"""
        # Limit cache size
        if len(self._text_cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self._text_cache.keys())[:100]
            for key in oldest_keys:
                del self._text_cache[key]
        
        self._text_cache[cache_key] = result
    
    # Enhanced extraction methods with better performance
    def extract_platform_info(self, text: str) -> List[str]:
        """Extract platform identifiers using compiled regex patterns"""
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
        """Extract codes using compiled regex patterns"""
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
        """Identify primary hardware module from discovered keywords"""
        if not text or not self.hardware_keywords:
            return "unknown"
        
        text_lower = text.lower()
        
        # Look for hardware keywords, return the first match
        for keyword in sorted(self.hardware_keywords, key=len, reverse=True):  # Longer keywords first
            if keyword in text_lower:
                return keyword
        
        return "unknown"
    
    def extract_technical_keywords(self, text: str) -> Set[str]:
        """Extract technical terms using discovered keywords"""
        if not text:
            return set()
        
        text_lower = text.lower()
        keywords = set()
        
        # Check technical terms
        for term in self.technical_keywords:
            if term in text_lower:
                keywords.add(f"tech_{term}")
        
        # Check error types
        for error_type in self.error_keywords:
            if error_type in text_lower:
                keywords.add(f"error_{error_type}")
        
        return keywords
    
    def preprocess_title_enhanced(self, title: str) -> Dict[str, any]:
        """Enhanced title preprocessing with caching"""
        if not title or not title.strip():
            return self._empty_title_result()
        
        # Check cache first
        cache_key = f"title_{hash(title)}"
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        title_lower = title.lower()
        
        # Extract structured information using discovered patterns
        platforms = self.extract_platform_info(title)
        codes = self.extract_codes(title)
        hw_module = self.categorize_hardware_module(title)
        tech_keywords = self.extract_technical_keywords(title)

        # More aggressive cleaning while preserving important terms
        clean_title = re.sub(r'[^\w\s\.\-_]', ' ', title_lower)
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()
        
        # Better tokenization
        tokens = [t for t in clean_title.split() if len(t) > 1 and t.isalnum()]
        
        # Add extracted features as tokens
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
        
        # Cache the result
        self._cache_result(cache_key, result)
        return result
    
    def _empty_title_result(self) -> Dict[str, any]:
        """Return empty result structure"""
        return {
            'text': "",
            'platforms': [],
            'codes': [],
            'hw_module': "unknown",
            'tech_keywords': []
        }
    
    def preprocess_steps_enhanced(self, steps: str) -> Dict[str, any]:
        """Enhanced steps preprocessing with caching"""
        if not steps or not steps.strip():
            return self._empty_steps_result()
        
        # Check cache
        cache_key = f"steps_{hash(steps)}"
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached
        
        steps_lower = steps.lower().replace("→", " ").replace("->", " ")
        
        # Extract procedure sequences using discovered keywords
        procedures = []
        for keyword in self.procedure_keywords:
            if keyword in steps_lower:
                procedures.append(f"proc_{keyword}")
        
        # Better cleaning and tokenization
        clean_steps = re.sub(r'[^a-zA-Z0-9\s\-_.]', ' ', steps_lower)
        clean_steps = re.sub(r'\s+', ' ', clean_steps).strip()
        
        tokens = [t for t in clean_steps.split() if len(t) > 1 and t.replace('_', '').replace('-', '').isalnum()]
        
        # Add procedure features
        tokens.extend(procedures)
        
        # Better step complexity calculation
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
        """Return empty steps result structure"""
        return {
            'text': "",
            'procedures': [],
            'sequence_length': 1
        }
    
    def preprocess_description_enhanced(self, descriptions: List[str]) -> List[Dict[str, any]]:
        """Enhanced description preprocessing with batch processing and caching"""
        if not descriptions:
            return []
        
        results = []
        uncached_descriptions = []
        uncached_indices = []
        
        # Check cache for each description
        for i, desc in enumerate(descriptions):
            if not desc or not desc.strip():
                results.append(self._empty_description_result())
                continue
            
            cache_key = f"desc_{hash(desc)}"
            cached = self._get_cached_result(cache_key)
            
            if cached:
                results.append(cached)
            else:
                results.append(None)  # Placeholder
                uncached_descriptions.append(desc)
                uncached_indices.append(i)
        
        # Process uncached descriptions in batch
        if uncached_descriptions:
            logger.debug(f"Processing {len(uncached_descriptions)} uncached descriptions")
            
            for doc_idx, doc in enumerate(nlp.pipe(uncached_descriptions, batch_size=50)):
                try:
                    # Extract meaningful tokens more efficiently
                    tokens = []
                    for token in doc:
                        if (not token.is_stop and 
                            len(token.text) > 2 and 
                            token.text.replace('_', '').replace('-', '').isalnum()):
                            tokens.append(token.lemma_.lower())
                    
                    # Extract features using discovered patterns
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
                    
                    # Create result
                    result = {
                        'text': " ".join(sorted(set(tokens))),
                        'platforms': platforms,
                        'codes': codes,
                        'hw_module': hw_module,
                        'tech_keywords': list(tech_keywords),
                        'token_count': len(set(tokens))
                    }
                    
                    # Cache and store result
                    original_index = uncached_indices[doc_idx]
                    cache_key = f"desc_{hash(uncached_descriptions[doc_idx])}"
                    self._cache_result(cache_key, result)
                    results[original_index] = result
                    
                except Exception as e:
                    logger.warning(f"Failed to process description {doc_idx}: {e}")
                    results[uncached_indices[doc_idx]] = self._empty_description_result()
        
        return results
    
    def _empty_description_result(self) -> Dict[str, any]:
        """Return empty description result structure"""
        return {
            'text': "",
            'platforms': [],
            'codes': [],
            'hw_module': "unknown", 
            'tech_keywords': [],
            'token_count': 0
        }
    
    def get_stats(self) -> Dict[str, any]:
        """Get processor statistics"""
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
        """
        Main end-to-end pipeline:
        1) Read CSV
        2) Preprocess title / steps / description
        3) Create combined feature text
        4) Extract summary columns
        """
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

        # Summary columns for quick analysis
        df['hardware_module'] = df['title_processed'].map(lambda x: x['hw_module'])
        df['platform_count']   = df['title_processed'].map(lambda x: len(x['platforms']))
        df['procedure_count']  = df['steps_processed'].map(lambda x: len(x['procedures']))
        df['sequence_complexity'] = df['steps_processed'].map(lambda x: x['sequence_length'])

        logger.info(f"Finished processing {len(df)} tickets")
        return df

    def create_enhanced_features(self, row: Dict[str, any]) -> str:
        """Build the same searchable text for batch and newly added tickets."""
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
        """Clear the processing cache"""
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
