import spacy
import re


class AspectExtractionMixin:
    """Shared aspect extraction utilities for ABSA models"""

    def __init__(self):
        if not hasattr(self, 'nlp'):
            self.nlp = spacy.load("en_core_web_sm")

    def _get_text(self, aspect):
        return aspect.text.strip()

    def _get_root(self, aspect):
        return aspect.root if hasattr(aspect, "root") else aspect

    def _get_start_char(self, aspect):
        if hasattr(aspect, "start_char"):
            return aspect.start_char
        return aspect.idx

    def _get_end_char(self, aspect):
        if hasattr(aspect, "end_char"):
            return aspect.end_char
        t = aspect if not hasattr(aspect, "root") else aspect.root
        return t.idx + len(t.text)

    def _get_pos(self, aspect):
        return self._get_root(aspect).pos_

    def _is_valid_aspect(self, aspect) -> bool:
        text = self._get_text(aspect).lower().strip()
        text = re.sub(r'[^\w\s]+$', '', text)


        if len(text) <= 2:
            return False


        if re.fullmatch(r'[^a-z0-9\s]+', text):
            return False

        normalized = self._normalize_aspect(text).lower()

        # Skip very short normalized aspects
        if len(normalized) <= 2:
            return False

        if re.search(r'\w+-like\s+(aftertaste|taste|flavor)', normalized):
            return False

        # Reduced filler terms - only truly non-reviewable items
        filler_terms = {
            # Pure meta/pronouns/generic
            'thing', 'things', 'bit', 'lot', 'way', 'ways', 'part', 'sort',
            'while', 'time', 'times', 'day', 'days', 'night', 'week', 'weeks',
            'character', 'regards', 'eye', 'eyes', 'note', 'question', 'questions',
            'improvement', 'area', 'room',

            'kind', 'road', 'hiking', 'amazing', 'long lines', 'food', 'attraction',
            'real attraction', 'sugar', 'cream', 'country cream', 'selection',

            'piece', 'heaven', 'cow dung', 'parlors', 'neighboring ice cream parlors',
            'entire life', 'life', 'serve',

            'favorite places', 'tables', 'cookie', 'nice treat', 'far side',
            'favorite', 'shake', 'fall', 'special', 'sweetness',

            # People/relationships (not staff/service)
            'husband', 'wife', 'son', 'daughter', 'girl', 'girls', 'girlfriend', 'friends',
            'manager', 'owner', 'people', 'everyone', 'he', 'she',
            'buddies', 'customer', 'customers',
            'us', 'folks', 'brother', 'sister', 'law', 'sister-in-law',
            'lady', 'young lady', 'valentine', 'gentlemen', 'locals',

            # Geographic locations (not venue aspects)
            'urbana', 'monticello', 'westville', 'cu area',
            'town', 'barn', 'dairy', 'sages', 'sidney', 'champaign',
            'fields', 'corn fields',
            'phx', 'tempe', 'chandler', 'valley', 'az', 'walmart', 'google',
            'instagram', 'joe', 'st joe', 'bay', 'plaza', 'complex',
            'pokitrition', 'culvers', 'firehouse subs', 'dq', 'arizona',
            'chandler location', 'neighborhood', 'strip mall', 'japan',

            # Temporal/abstract
            'tolerance', 'season', 'years', 'minutes', 'mins',
            'sweetness tolerance', 'fall season special', 'same ownership',
            'ownership', 'those years', 'summer', 'afternoon', 'month',
            'business hours', 'hours', 'year round', 'ago', 'second time', 'first time',

            # Pure fragments (not food items)
            'ice', 'cream', 'cone', 'cones', 'swirl',

            'aftertaste', 'place',

            # Meta/digital
            'line', 'lines', 'entire line', 'product', 'menu item',
            'facebook', 'page', 'business', 'franchises', 'factory',
            'drive', 'trip', 'journey', 'excursions', 'trips', 'hiking trip',
            'road trip', 'stop', 'visit', 'visits',

            # Personal items
            'stomach', 'ache', 'home', 'experience', 'experiences', 'problem', 'life',
            'myself', 'jeans', 'car', 'review', 'reviews',
            'tongue', 'tooth', 'sweet tooth', 'mind', 'hair',
            'socks', 'date', 'cravings', 'priority',

            # Competitors/brands
            'baskin robbins', 'jarlings', 'custard cup',
            'rewind', 'blizzard', 'dripps',

            'bike club', 'year', 'round', 'evening ice cream cycling excursions', 'club',

            # Additional context words
            'places', 'wait', 'waiting', 'any time', 'tornado', 'tornadoes',
            'update', 'stars', 'star', 'ps', 'explanation', 'check', 'cash',
            'dusk', 'speed', 'light', 'minute', 'minutes',
            'joint', 'joints', 'routine',
            'detail', 'details', 'surprise', 'establishment', 'tip', 'pro tip',
            'highlights', 'lowlights',
            'neon sign', 'wall', 'grass wall',
            'effect', 'swirling effect', 'bowl', 'lid', 'slices',
            'volleyball', 'dinner', 'refreshment', 'cereal', 'cereals', 'real deal',
            'spot', 'spots', 'pace', 'handout', 'timing',
            'concept', 'concepts', 'kind', 'beauty', 'skin', 'sugar rush',
            'stuff', 'lightbulb',
            'money', 'opinions', 'opinion',
            'plethora', 'concoctions', 'combinations',
            'word', 'words', 'training', 'least', 'heat', 'covid', 'covid19', 'masks',
            'mask', 'top', 'balance', 'hint', 'hints', 'undertone', 'kick',
            'point', 'plus point', 'chewiness', 'richness', 'craving',
            'traffic', 'foot traffic', 'list', 'menu', 'menus',
            'tubs', 'refunds', 'refund policy', 'policy',
            'artisanal varieties', 'picky sticks', 'colors', 'color',
            'photos', 'pictures', 'picture',
            'default', 'rest', 'fan', 'fans', 'cartoons', 'saturday morning',
            'big bowl', 'news', 'article',
            'brownie bites', 'person', 'energy', 'setting', 'inside',
            'mix in', 'mix ins', 'pieces', 'base',
            'cooking', 'situation', 'disinfectants', 'environment',
            'substitutes', 'sushi burrito', 'almond slices', 'teddy grahams',
            'shop', 'shops', 'counter', 'window', 'picnic tables',
            'joy', 'pride', 'gems', 'bomb', 'miss', 'deal', 'complaint',
            'complaints', 'critique', 'sign', 'bite', 'bites', 'cup', 'cups',
            'quart', 'cashier', 'bobarista', 'worker', 'workers', 'club',
            'order', 'orders', 'selection', 'attention', 'sweets', 'items',
            'yum', 'overrun', 'sample', 'samples', 'hot day', 'humid night',
            'cold day', 'weekday', 'valentines day', 'weekend', 'firsts',
            'online order form', 'app', 'website', 'grocery pickup', 'news article',
            'bread', 'sliced bread', 'custard', 'world', 'establishment', 'establishments',
            'employees', 'employee', 'staff members'
        }

        if normalized in filler_terms:
            return False

        # Filter standalone ingredients (not flavor names)
        ingredient_terms = {
            'bananas', 'banana', 'graham', 'crackers', 'graham crackers',
            'pecans', 'peanuts', 'chocolate', 'cookie dough', 'vanilla',
            'milk', 'dough', 'splenda', 'aftertaste', 'gummy bears',
            'pocky sticks', 'shavings', 'chocolate shavings', 'puree',
            'blueberry puree', 'sherbet', 'creamsicle', 'frosted flakes',
            'reeses', 'reeses puff', 'peanut butter', 'ginger', 'lemon', 'lemons', 'honey',
            'caramel', 'peach', 'lychee', 'strawberry', 'strawberries',
            'matcha', 'coconut', 'aloe vera', 'jasmine', 'hojicha',
            'captain crunch berries', 'apple jacks', 'cinnamon toast',
            'fruity pebbles', 'pineapple', 'crush', 'crystal boba',
            'black boba', 'cookies', 'cream'
        }
        if normalized in ingredient_terms:
            return False

        # Filter phrases with personal indicators
        personal_patterns = ['my ', 'our ', 'your ', 'his ', 'her ', 'this ', 'these ', 'those ', "what's ", 'their ',
                             'every ', 'one of', "i'm "]
        if any(normalized.startswith(pattern) for pattern in personal_patterns):
            return False

        if normalized.startswith('the ') or normalized.startswith('such a '):
            return False

        # Filter if contains problematic terms
        problematic_terms = ['tolerance', 'husband', 'wife', 'girl', 'regards',
                             'myself', 'review', 'haha', 'google', 'star', 'defo', 'hmm',
                             'us', 'complaint', 'critique', 'hair', 'brother', 'sister',
                             'law', 's/o', 'pickup', 'article', 'girlfriend', 'valentine',
                             'socks', 'neighborhood']
        if any(term in normalized for term in problematic_terms):
            return False

        # Filter comparative references (ADD HERE)
        if any(word in normalized for word in ['far better', 'much better', 'neighboring']):
            return False

        # Filter temporal/seasonal/descriptive phrases - but allow "large size" etc
        if any(normalized.startswith(prefix) for prefix in [
            'fall ', 'season ', 'special ', 'rotating ', 'featured ', 'pro ',
            'same ', 'bit ', 'kind of', 'only ', 'lot of',
            'much ', 'more ', 'hard ', 'soft ', 'other ', 'both ',
            'hand made', 'second ', 'first '
        ]):
            return False


        # Filter "what" phrases
        if normalized.startswith('what'):
            return False

        # Filter numeric/time/price expressions
        if re.search(r'(\d+\s*(minute|hour|day|mile|star|dollar|\$|%|ft|weeks?)|1970s|50%|75%|25%|30mins)', normalized):
            return False

        # Filter standalone adjectives (but not compound terms)
        if self._get_pos(aspect) == 'ADJ' and len(normalized.split()) == 1:
            return False

        # Filter proper nouns that are locations or brand names
        if self._get_pos(aspect) == 'PROPN':
            filtered_names = {
                'urbana', 'monticello', 'westville', 'barn', 'dairy',
                'facebook', 'baskin', 'robbins', 'sages', 'sidney',
                'champaign', 'jarlings', 'walmart', 'rewind', 'instagram',
                'tempe', 'chandler', 'phx', 'google', 'pokitrition',
                'culvers', 'firehouse', 'dq', 'bay', 'az', 'covid',
                'arizona', 'dripps', 'japan', 'covid19',
                'champaign urbana', 'champaign-urbana'
            }
            if normalized in filtered_names:
                return False

        return True

    def _normalize_aspect(self, aspect_text: str) -> str:
        """Remove possessives/pronouns/leading modifiers and trailing punctuation."""
        text = aspect_text.strip()
        text = text.lower()

        # Remove "what a/what's/what is" constructions
        text = re.sub(r'^what(\s+a|\s+is|\'s)\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^fun\s+', '', text, flags=re.IGNORECASE)
        # Remove phrases like "all in all"
        text = re.sub(r'^all\s+in\s+all,?\s*', '', text, flags=re.IGNORECASE)

        # Remove leading descriptive adjectives - but preserve size/quality descriptors
        # that are integral to the aspect (e.g., "large size")
        while True:
            before = text
            text = re.sub(
                r'^(great|fun|nice|good|amazing|awesome|little|best|delicious|wonderful|superb|fattier|'
                r'cute|plain|simple|ample|basic|strange|near|constant|long|fresh|truly|made|'
                r'some|this|these|those|same|many|about|any|entire|every|home|typical|'
                r'traditional|american|private|outdoor|festive|picture|perfect|small|'
                r'trendy|layered|exceptionally|insanely|pretty|real|female|male|'
                r'biggest|massive|flat out|super|double|regular|original|'
                r'yummy|sounding|looking|delicate|lightly|cool|perfect|bit|various|'
                r'only|tiny|plus|surprising|surprisingly|popular|simple|easy|'
                r'limited|new|chewy|polite|expressive|subtle|distinct|'
                r'earthy|speedy|friendly|wide|affordable|fantastic|impressed|whole|'
                r'aromatic|buggy|slow|cute|little|tasty|fun|overall|decent|'
                r'quick|young|fabulous|hand|dense|creamy|sweet|quickly|'
                r'big|ol)\s+',
                '', text, flags=re.IGNORECASE
            )
            if text == before:
                break

        # Remove leading articles
        text = re.sub(r'^(a|an|the)\s+', '', text, flags=re.IGNORECASE)

        # Remove leading possessives/pronouns
        text = re.sub(r'^(my|their|our|your|his|her|its|one of)\s+', '', text, flags=re.IGNORECASE)

        # Remove leading intensifiers + adjectives (but not size/quality descriptors)
        text = re.sub(
            r'^(very|really|so|super|quite|extremely|too|way|pretty|a bit|a little|kind of|lot of)\s+(good|bad|tasty|nice|sweet|delicious|friendly|artificial|cute|clean|watery|rude|polite|fast)\s+',
            '', text, flags=re.IGNORECASE
        )

        # Remove remaining single intensifiers
        text = re.sub(
            r'^(very|really|so|super|quite|extremely|too|way|pretty|a bit|a little|kind of|even though|lot of|much|more|absolutely|both|always)\s+',
            '', text, flags=re.IGNORECASE
        )

        # Remove temporal/season descriptors
        text = re.sub(r'^(fall|season|special|featured|rotating|near|constant|late|night)\s+', '', text,
                      flags=re.IGNORECASE)


        # Remove "small town/ice cream/bubble tea" before nouns (but keep if it's the whole aspect)
        if not re.fullmatch(r'(small\s+town|ice\s+cream|bubble\s+tea|sweet\s+tea)', text, flags=re.IGNORECASE):
            text = re.sub(r'^(small\s+town|small|ice\s+cream|bubble\s+tea|sweet\s+tea|foot)\s+', '', text,
                          flags=re.IGNORECASE)

        # Remove business names
        text = re.sub(r'^(dairy\s+barn|sidney\s+dairy\s+barn|rewind|dripps)\s*', '', text, flags=re.IGNORECASE)

        # Remove corporate/local descriptors (but keep "local place")
        if not text.endswith('place'):
            text = re.sub(r'^(corporate|local|locally)\s+', '', text, flags=re.IGNORECASE)

        text = re.sub(r'^(vegan|non\s+vegan|fresh|artisanal|hard\s+scoop|hand\s+made)\s+', '',
                      text, flags=re.IGNORECASE)

        # Remove price/online indicators
        text = re.sub(r'^(1970s|cheap|expensive|pricier|online)\s+', '', text, flags=re.IGNORECASE)

        # Remove "with" constructions like "with mix-ins"
        text = re.sub(r'\s+with\s+.*$', '', text, flags=re.IGNORECASE)

        # Strip trailing punctuation
        text = re.sub(r'[^\w\s]+$', '', text)

        # Remove leading/trailing quotes
        text = re.sub(r'^["\']|["\']$', '', text)

        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _expand_aspect_with_modifiers(self, doc, chunk):
        """Expand aspect to include important preceding modifiers like size/quality descriptors"""
        if not hasattr(chunk, 'start'):
            return chunk

        start_idx = chunk.start

        # Look back up to 3 tokens for important modifiers
        important_modifiers = {'large', 'small', 'medium', 'big', 'huge', 'mini',
                               'hot', 'cold', 'frozen', 'fresh', 'iced',
                               'size', 'regular', 'grande', 'venti'}

        lookback_start = max(0, start_idx - 3)
        prefix_tokens = []

        for i in range(lookback_start, start_idx):
            token = doc[i]
            if token.text.lower() in important_modifiers or token.dep_ == 'amod':
                prefix_tokens.append(token)
            elif token.pos_ not in {'DET', 'ADP', 'PUNCT'}:
                # Reset if we hit something that's not a determiner/preposition
                prefix_tokens = []

        if prefix_tokens:
            # Create expanded span
            new_start = prefix_tokens[0].i
            return doc[new_start:chunk.end]

        return chunk

    def _merge_coordinated_aspects(self, aspects, doc, raw_text):
        """Merge adjacent aspects connected by 'and'/',' into single phrases."""
        if not aspects:
            return []

        items = []
        for a in aspects:
            items.append((self._get_start_char(a), self._get_end_char(a), a))

        items.sort(key=lambda x: x[0])

        merged = []
        skip_next = False

        for i in range(len(items)):
            if skip_next:
                skip_next = False
                continue

            start_i, end_i, a_i = items[i]
            merged_flag = False

            if i + 1 < len(items):
                start_j, end_j, a_j = items[i + 1]
                mid_sub = raw_text[end_i:start_j].lower()

                if re.search(r'\b(and|,|&)\b', mid_sub) and len(mid_sub) < 10:
                    candidate = raw_text[start_i:end_j].strip()
                    candidate_norm = self._normalize_aspect(candidate)

                    single_i = self._normalize_aspect(self._get_text(a_i)).lower()
                    single_j = self._normalize_aspect(self._get_text(a_j)).lower()

                    service_terms = {'service', 'staff', 'crew', 'team', 'employees'}
                    if single_i in service_terms or single_j in service_terms:
                        pass  # merged_flag stays False
                    elif (single_i in candidate_norm.lower() and
                          single_j in candidate_norm.lower() and
                          len(candidate_norm) > max(len(single_i), len(single_j))):

                        try:
                            token_start = None
                            token_end = None
                            for tok in doc:
                                if tok.idx >= start_i and token_start is None:
                                    token_start = tok.i
                                if tok.idx + len(tok.text) <= end_j:
                                    token_end = tok.i + 1

                            if token_start is not None and token_end is not None and token_end > token_start:
                                span = doc[token_start:token_end]
                                merged.append(span)
                                merged_flag = True
                                skip_next = True
                        except Exception:
                            merged.append(self.nlp(candidate_norm))
                            merged_flag = True
                            skip_next = True

            if not merged_flag:
                merged.append(a_i)

        final = []
        seen_exact = set()

        for a in merged:
            norm = self._normalize_aspect(self._get_text(a)).lower()
            if norm in seen_exact:
                continue
            seen_exact.add(norm)
            final.append(a)

        return final


    def _extract_aspects(self, doc):
        """Extract aspect candidates with validation and deduplication"""
        aspects = []
        seen = set()

        # Extract noun chunks first
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in {'NOUN', 'PROPN'} and self._is_valid_aspect(chunk):
                # Try to expand with important modifiers
                expanded_chunk = self._expand_aspect_with_modifiers(doc, chunk)

                norm = self._normalize_aspect(expanded_chunk.text).lower()

                # Additional validation after normalization
                if norm and norm not in seen and len(norm) >= 3:
                    # Less aggressive context filtering for generic terms
                    generic_terms = {'ice cream', 'soft serve', 'bubble tea', 'boba', 'chocolate ice cream',
                                     'vanilla ice cream'}

                    if norm in generic_terms:
                        # Only skip if clearly just mentioned in passing, not reviewed
                        should_skip = False
                        if expanded_chunk.start > 0:
                            prev_tokens = [doc[i].text.lower() for i in
                                           range(max(0, expanded_chunk.start - 2), expanded_chunk.start)]
                            # Reduced skip words - only skip if clearly not the subject
                            skip_words = {'of', 'about'}
                            if any(word in skip_words for word in prev_tokens):
                                should_skip = True
                        if should_skip:
                            continue

                    aspects.append(expanded_chunk)
                    seen.add(norm)

        # Extract coordinated nouns before "and"
        for token in doc:
            if token.pos_ in {'NOUN', 'PROPN'} and token.i + 1 < len(doc):
                next_tok = doc[token.i + 1]
                if next_tok.text.lower() in {'and', '&', ','}:
                    if self._is_valid_aspect(token):
                        norm = self._normalize_aspect(token.text).lower()
                        if norm and norm not in seen and len(norm) >= 3:
                            aspects.append(token)
                            seen.add(norm)

        # Extract standalone nouns not in chunks
        for token in doc:
            if token.pos_ in {'NOUN', 'PROPN'}:
                # Skip if inside an accepted chunk
                inside = False
                for ch in aspects:
                    if hasattr(ch, 'start') and token.i >= ch.start and token.i < ch.end:
                        inside = True
                        break
                if inside:
                    continue

                # Validate token
                if self._is_valid_aspect(token):
                    norm = self._normalize_aspect(token.text).lower()
                    if norm and norm not in seen and len(norm) >= 3:
                        aspects.append(token)
                        seen.add(norm)

        return aspects