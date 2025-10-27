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

        # Comprehensive filler terms - things that are NOT aspects
        filler_terms = {
            # Generic/vague
            'thing', 'things', 'bit', 'lot', 'way', 'ways', 'part', 'sort', 'yum', 'overrun',
            'while', 'time', 'times', 'day', 'days', 'night', 'week', 'weeks', 'sample', 'samples',
            'hot day', 'humid night', 'cold day', 'weekday', 'valentines day', 'weekend',
            'character', 'regards', 'eye', 'eyes', 'joy', 'pride', 'treat', 'treats', 'stop',
            'note', 'question', 'questions', 'improvement', 'area', 'room', 'firsts',

            # People/relationships
            'husband', 'wife', 'son', 'daughter', 'girl', 'girls', 'girlfriend', 'friends', 'gentlemen',
            'manager', 'owner', 'people', 'everyone', 'he', 'she', 'club', 'locals',
            'workers', 'worker', 'buddies', 'customer', 'customers', 'cashier',
            'bobarista', 'us', 'folks', 'brother', 'sister', 'law', 'sister-in-law',
            'employees', 'employee', 'staff members', 'lady', 'young lady', 'valentine',

            # Places/locations (not business aspects)
            'side', 'urbana', 'monticello', 'westville', 'cu area',
            'far side', 'town', 'barn', 'dairy', 'sages', 'sidney', 'champaign',
            'fields', 'corn fields', 'window', 'picnic tables',
            'phx', 'tempe', 'chandler', 'valley', 'az', 'walmart', 'google',
            'instagram', 'joe', 'st joe', 'bay', 'plaza', 'counter', 'complex',
            'pokitrition', 'culvers', 'firehouse subs', 'dq', 'arizona',
            'grocery pickup', 'news article', 'app', 'website', 'online order form',
            'chandler location', 'neighborhood', 'strip mall', 'japan',

            # Temporal/possession
            'tolerance', 'season', 'years', 'minutes', 'mins',
            'sweetness tolerance', 'fall season special', 'same ownership',
            'ownership', 'those years', 'summer', 'afternoon', 'month',
            'business hours', 'hours', 'year round', 'ago', 'second time', 'first time',

            # Ice cream fragments
            'ice', 'cream', 'cone', 'cones', 'swirl',

            # Abstract/meta
            'line', 'lines', 'entire line', 'product', 'menu item',
            'facebook', 'page', 'business', 'franchises', 'factory', 'gems',
            'drive', 'trip', 'journey', 'excursions', 'trips', 'hiking trip',
            'road trip', 'stop', 'visit', 'visits',

            # Comparatives/references
            'bread', 'sliced bread', 'custard', 'world', 'bomb', 'miss',
            'deal', 'complaint', 'complaints', 'critique', 'sign',

            # Health/body/personal
            'stomach', 'ache', 'home', 'experience', 'experiences', 'problem', 'life',
            'cup', 'cups', 'quart', 'myself', 'jeans', 'car', 'review', 'reviews',
            'tongue', 'tooth', 'sweet tooth', 'bite', 'bites', 'mind', 'hair',
            'socks', 'date', 'cravings', 'priority',

            # Competitors/brands
            'baskin robbins', 'jarlings', 'custard cup', 'establishments',
            'rewind', 'blizzard', 'dripps',

            # Additional context words
            'places', 'wait', 'waiting', 'any time', 'tornado', 'tornadoes',
            'update', 'stars', 'star', 'ps', 'explanation', 'check', 'cash',
            'dusk', 'speed', 'light', 'minute', 'minutes', 'order', 'orders',
            'joint', 'joints', 'selection', 'routine', 'attention',
            'detail', 'details', 'surprise', 'establishment', 'tip', 'pro tip',
            'highlights', 'lowlights', 'dessert', 'desserts', 'drink', 'drinks',
            'topping', 'toppings', 'neon sign', 'wall', 'grass wall',
            'effect', 'swirling effect', 'bowl', 'lid', 'slices',
            'volleyball', 'dinner', 'refreshment', 'cereal', 'cereals', 'real deal',
            'spot', 'spots', 'sweets', 'pace', 'handout', 'timing',
            'concept', 'concepts', 'kind', 'beauty', 'skin', 'sugar rush',
            'items', 'stuff', 'lightbulb', 'shake', 'shakes',
            'scoop', 'scoops', 'money', 'opinions', 'opinion', 'variety', 'varieties', 'choices', 'choice',
            'plethora', 'concoctions', 'combinations', 'creation', 'creations',
            'word', 'words', 'training', 'least', 'heat', 'covid', 'covid19', 'masks',
            'mask', 'top', 'balance', 'hint', 'hints', 'undertone', 'kick',
            'point', 'plus point', 'chewiness', 'richness', 'craving',
            'traffic', 'foot traffic', 'list', 'menu', 'menus', 'amount', 'amounts',
            'tubs', 'refunds', 'refund policy', 'policy',
            'artisanal varieties', 'picky sticks', 'colors', 'color',
            'flavors', 'flavor', 'flavoring', 'photos', 'pictures', 'picture',
            'default', 'rest', 'fan', 'fans', 'cartoons', 'saturday morning',
            'big bowl', 'news', 'article', 'milkshakes', 'milkshake',
            'brownie bites', 'person', 'energy', 'setting', 'inside',
            'options', 'option', 'mix in', 'mix ins', 'pieces', 'base',
            'cooking', 'situation', 'disinfectants', 'environment',
            'substitutes', 'sushi burrito', 'almond slices', 'teddy grahams',
            'shop', 'shops'
        }

        if normalized in filler_terms:
            return False

        # Filter ingredients when standalone (not part of flavor name)
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
        # Only filter if it's truly standalone
        if normalized in ingredient_terms:
            return False

        # Filter phrases with personal indicators
        personal_patterns = ['my ', 'our ', 'your ', 'his ', 'her ', 'this ', 'these ', 'those ', "what's ", 'their ',
                             'every ', 'one of', "i'm "]
        if any(normalized.startswith(pattern) for pattern in personal_patterns):
            return False

        # Filter if contains problematic terms
        problematic_terms = ['tolerance', 'husband', 'wife', 'girl', 'regards', 'club',
                             'myself', 'review', 'haha', 'google', 'star', 'defo', 'hmm',
                             'us', 'complaint', 'critique', 'hair', 'brother', 'sister',
                             'law', 's/o', 'pickup', 'article', 'girlfriend', 'valentine',
                             'socks', 'neighborhood']
        if any(term in normalized for term in problematic_terms):
            return False

        # Filter temporal/seasonal/descriptive phrases
        if any(normalized.startswith(prefix) for prefix in [
            'fall ', 'season ', 'special ', 'rotating ', 'featured ', 'pro ',
            'mix in', 'same ', 'good ', 'bit ', 'kind of', 'only ', 'lot of',
            'much ', 'more ', 'new ', 'hard ', 'soft ', 'other ', 'both ',
            'hand made', 'second ', 'first '
        ]):
            return False

        # Filter "what" phrases
        if normalized.startswith('what'):
            return False

        # Filter numeric/time/price expressions
        if re.search(r'(\d+\s*(minute|hour|day|mile|star|dollar|\$|%|ft|weeks?)|1970s|50%|75%|25%|30mins)', normalized):
            return False

        # Filter standalone adjectives
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
                'arizona', 'dripps', 'japan', 'covid19'
            }
            if normalized in filtered_names:
                return False

        return True

    def _normalize_aspect(self, aspect_text: str) -> str:
        """Remove possessives/pronouns/leading modifiers and trailing punctuation."""
        text = aspect_text.strip()

        # Normalize case first
        text = text.lower()

        # Remove "what a/what's/what is" constructions
        text = re.sub(r'^what(\s+a|\s+is|\'s)\s+', '', text, flags=re.IGNORECASE)

        # Remove leading descriptive adjectives (very comprehensive list)
        text = re.sub(
            r'^(great|fun|nice|good|amazing|awesome|little|local|best|delicious|wonderful|'
            r'cute|plain|simple|ample|basic|strange|near|constant|long|fresh|truly|made|'
            r'some|this|these|those|same|many|about|any|entire|every|home|typical|'
            r'traditional|american|private|outdoor|festive|picture|perfect|small|'
            r'trendy|layered|exceptionally|insanely|pretty|real|female|male|'
            r'biggest|massive|flat out|super|double|regular|original|'
            r'yummy|sounding|looking|delicate|lightly|cool|perfect|bit|various|'
            r'only|tiny|plus|surprising|surprisingly|popular|simple|easy|'
            r'limited|new|hard|soft|chewy|polite|expressive|subtle|distinct|'
            r'earthy|speedy|friendly|wide|affordable|fantastic|impressed|whole|'
            r'aromatic|buggy|slow|cute|little|tasty|fun|overall|decent|'
            r'quick|young|fabulous|hand|dense|creamy|sweet|quickly|'
            r'big|ol)\s+',
            '', text, flags=re.IGNORECASE
        )

        # Remove leading articles
        text = re.sub(r'^(a|an|the)\s+', '', text, flags=re.IGNORECASE)

        # Remove leading possessives/pronouns
        text = re.sub(r'^(my|their|our|your|his|her|its|one of)\s+', '', text, flags=re.IGNORECASE)

        # Remove leading intensifiers + adjectives
        text = re.sub(
            r'^(very|really|so|super|quite|extremely|too|way|pretty|a bit|a little|kind of|lot of)\s+(good|bad|tasty|nice|sweet|delicious|friendly|artificial|cute|clean|soft|watery|rude|polite|fast)\s+',
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

        # Remove "small town/ice cream/bubble tea" before nouns
        text = re.sub(r'^(small\s+town|small|ice\s+cream|bubble\s+tea|sweet\s+tea|foot)\s+', '', text,
                      flags=re.IGNORECASE)

        # Remove business names
        text = re.sub(r'^(dairy\s+barn|sidney\s+dairy\s+barn|rewind|dripps)\s*', '', text, flags=re.IGNORECASE)

        # Remove corporate/local/vegan/artisanal descriptors
        text = re.sub(r'^(corporate|local|locally|vegan|non\s+vegan|fresh|artisanal|hard\s+scoop|hand\s+made)\s+', '',
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

    def _extract_aspects(self, doc):
        """Extract aspect candidates with validation and deduplication"""
        aspects = []
        seen = set()

        # Extract noun chunks first
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in {'NOUN', 'PROPN'} and self._is_valid_aspect(chunk):
                norm = self._normalize_aspect(chunk.text).lower()

                # Additional validation after normalization
                if norm and norm not in seen and len(norm) >= 3:
                    # Context-aware filtering for generic terms
                    generic_terms = ['ice cream', 'soft serve', 'bubble tea', 'boba', 'drinks', 'shakes']
                    if norm in generic_terms:
                        # Check if it's being reviewed vs just mentioned
                        should_skip = False
                        if chunk.start > 0:
                            prev_tokens = [doc[i].text.lower() for i in range(max(0, chunk.start - 2), chunk.start)]
                            skip_words = {'for', 'of', 'at', 'about', 'with', 'from', 'in', 'get', 'have', 'try',
                                          'some', 'had', 'got'}
                            if any(word in skip_words for word in prev_tokens):
                                should_skip = True
                        if should_skip:
                            continue

                    aspects.append(chunk)
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