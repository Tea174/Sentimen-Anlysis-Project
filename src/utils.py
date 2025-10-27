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

        # Comprehensive filler terms - EXPANDED
        filler_terms = {
            # Generic/vague
            'thing', 'things', 'bit', 'lot', 'way', 'ways', 'part', 'parts', 'sort', 'yum', 'overrun',
            'while', 'time', 'times', 'day', 'days', 'night', 'nights', 'week', 'weeks', 'month', 'months',
            'hot day', 'humid night', 'cold day', 'weekday', 'weekdays', 'valentines day', 'weekend', 'weekends',
            'character', 'characters', 'regards', 'eye', 'eyes', 'joy', 'pride', 'stop', 'stops',
            'note', 'notes', 'question', 'questions', 'improvement', 'improvements', 'area', 'areas',
            'room', 'rooms', 'firsts', 'first', 'second', 'thirds', 'half', 'halves',
            'piece', 'pieces', 'side', 'sides', 'level', 'levels', 'type', 'types',
            'something', 'anything', 'everything', 'nothing', 'somewhere', 'anywhere',
            'stuff', 'item', 'items', 'thing', 'things', 'end', 'ends', 'start', 'starts',
            'moment', 'moments', 'instance', 'instances', 'case', 'cases', 'point', 'points',
            'fact', 'facts', 'reason', 'reasons', 'result', 'results', 'issue', 'issues',

            # People/relationships - EXPANDED
            'husband', 'wife', 'son', 'daughter', 'child', 'children', 'kid', 'kids',
            'girl', 'girls', 'boy', 'boys', 'girlfriend', 'boyfriend', 'partner', 'partners',
            'friends', 'friend', 'buddy', 'buddies', 'pal', 'pals', 'mate', 'mates',
            'gentlemen', 'gentleman', 'lady', 'ladies', 'woman', 'women', 'man', 'men',
            'manager', 'managers', 'owner', 'owners', 'boss', 'bosses',
            'people', 'person', 'everyone', 'someone', 'anyone', 'nobody',
            'he', 'she', 'they', 'them', 'him', 'her', 'club', 'clubs', 'locals', 'local',
            'workers', 'worker', 'customer', 'customers', 'client', 'clients',
            'cashier', 'cashiers', 'bobarista', 'barista', 'baristas',
            'us', 'we', 'folks', 'folk', 'brother', 'brothers', 'sister', 'sisters',
            'law', 'sister-in-law', 'brother-in-law', 'mother', 'mom', 'father', 'dad',
            'employees', 'employee', 'staff', 'staff members', 'crew', 'team',
            'young lady', 'old lady', 'valentine', 'valentines', 'date', 'dates',
            'server', 'servers', 'waiter', 'waiters', 'waitress', 'waitresses',
            'chef', 'chefs', 'cook', 'cooks', 'family', 'families', 'relative', 'relatives',
            'neighbor', 'neighbors', 'guest', 'guests', 'visitor', 'visitors',

            # Places/locations - EXPANDED
            'side', 'urbana', 'monticello', 'westville', 'cu area', 'cu', 'area',
            'far side', 'town', 'towns', 'city', 'cities', 'barn', 'dairy', 'sages', 'sidney',
            'champaign', 'fields', 'field', 'corn fields', 'window', 'windows',
            'picnic tables', 'picnic table', 'table', 'tables', 'desk', 'desks',
            'phx', 'tempe', 'chandler', 'valley', 'az', 'arizona', 'walmart', 'target',
            'google', 'instagram', 'facebook', 'twitter', 'tiktok', 'snapchat',
            'joe', 'st joe', 'bay', 'plaza', 'plazas', 'counter', 'counters', 'complex',
            'pokitrition', 'culvers', 'firehouse subs', 'dq', 'dairy queen',
            'grocery pickup', 'pickup', 'news article', 'article', 'articles',
            'app', 'apps', 'website', 'websites', 'online order form', 'form', 'forms',
            'chandler location', 'location', 'locations', 'neighborhood', 'neighborhoods',
            'strip mall', 'mall', 'malls', 'japan', 'china', 'korea', 'usa', 'america',
            'parking', 'parking lot', 'lot', 'street', 'streets', 'road', 'roads',
            'building', 'buildings', 'store', 'stores', 'outlet', 'outlets',
            'corner', 'corners', 'block', 'blocks', 'district', 'districts',
            'venue', 'venues', 'spot', 'spots', 'joint', 'joints',
            'establishment', 'establishments', 'place', 'places',

            # Temporal/possession - EXPANDED
            'tolerance', 'season', 'seasons', 'years', 'year', 'minutes', 'minute', 'mins', 'min',
            'seconds', 'second', 'hours', 'hour', 'hrs', 'hr',
            'sweetness tolerance', 'fall season special', 'same ownership', 'ownership',
            'those years', 'summer', 'spring', 'winter', 'fall', 'autumn',
            'afternoon', 'morning', 'evening', 'noon', 'midnight', 'dusk', 'dawn',
            'business hours', 'year round', 'ago', 'second time', 'first time',
            'times', 'once', 'twice', 'always', 'never', 'sometimes', 'often',
            'rarely', 'usually', 'occasionally', 'frequently', 'constantly',
            'today', 'tomorrow', 'yesterday', 'tonight', 'now', 'then', 'later',
            'earlier', 'soon', 'recently', 'lately', 'currently', 'presently',

            # Abstract terms that aren't aspects - EXPANDED
            'favorite', 'favorites', 'sweetness', 'bitterness', 'sourness',
            'special', 'specials', 'feature', 'features', 'highlight', 'highlights',
            'lowlight', 'lowlights', 'aspect', 'aspects', 'quality', 'qualities',
            'characteristic', 'characteristics', 'attribute', 'attributes',
            'property', 'properties', 'trait', 'traits', 'feature', 'features',

            # Ice cream fragments (standalone)
            'ice', 'cream', 'cone', 'cones', 'swirl', 'swirls', 'scoop', 'scoops',
            'cup', 'cups', 'bowl', 'bowls', 'dish', 'dishes', 'serving', 'servings',

            # Abstract/meta - EXPANDED
            'line', 'lines', 'entire line', 'product', 'products', 'menu item', 'menu items',
            'facebook', 'page', 'pages', 'post', 'posts', 'business', 'businesses',
            'franchises', 'franchise', 'factory', 'factories', 'gems', 'gem',
            'drive', 'drives', 'trip', 'trips', 'journey', 'journeys',
            'excursions', 'excursion', 'hiking trip', 'road trip', 'visit', 'visits',
            'outing', 'outings', 'adventure', 'adventures', 'expedition', 'expeditions',
            'stop', 'stops', 'destination', 'destinations', 'tour', 'tours',

            # Comparatives/references - EXPANDED
            'bread', 'sliced bread', 'custard', 'world', 'worlds', 'bomb', 'bombs',
            'miss', 'deal', 'deals', 'bargain', 'bargains', 'steal', 'steals',
            'complaint', 'complaints', 'critique', 'critiques', 'criticism', 'criticisms',
            'sign', 'signs', 'neon sign', 'banner', 'banners', 'poster', 'posters',
            'comparison', 'comparisons', 'reference', 'references', 'example', 'examples',

            # Health/body/personal - EXPANDED
            'stomach', 'stomachs', 'ache', 'aches', 'pain', 'pains', 'hurt', 'hurts',
            'home', 'homes', 'house', 'houses', 'apartment', 'apartments',
            'experience', 'experiences', 'problem', 'problems', 'life', 'lives',
            'lifestyle', 'lifestyles', 'routine', 'routines', 'habit', 'habits',
            'cup', 'cups', 'quart', 'quarts', 'pint', 'pints', 'gallon', 'gallons',
            'myself', 'yourself', 'himself', 'herself', 'ourselves', 'themselves',
            'jeans', 'pants', 'shirt', 'shirts', 'socks', 'sock',
            'car', 'cars', 'vehicle', 'vehicles', 'bike', 'bikes',
            'review', 'reviews', 'rating', 'ratings', 'feedback', 'comment', 'comments',
            'tongue', 'tongues', 'tooth', 'teeth', 'sweet tooth', 'mouth', 'mouths',
            'bite', 'bites', 'chew', 'chews', 'mind', 'minds', 'thought', 'thoughts',
            'hair', 'hairs', 'head', 'heads', 'body', 'bodies', 'skin', 'face', 'faces',
            'hand', 'hands', 'finger', 'fingers', 'arm', 'arms', 'leg', 'legs',
            'cravings', 'craving', 'desire', 'desires', 'want', 'wants', 'need', 'needs',
            'priority', 'priorities', 'preference', 'preferences', 'choice', 'choices',
            'decision', 'decisions', 'opinion', 'opinions', 'view', 'views',
            'perspective', 'perspectives', 'feeling', 'feelings', 'emotion', 'emotions',
            'mood', 'moods', 'vibe', 'vibes', 'energy', 'aura',

            # Competitors/brands - EXPANDED
            'baskin robbins', 'baskin', 'robbins', 'jarlings', 'custard cup',
            'rewind', 'blizzard', 'blizzards', 'dripps', 'dripp',
            'mcdonalds', 'starbucks', 'dunkin', 'krispy kreme',
            'ben jerry', 'haagen dazs', 'cold stone', 'marble slab',

            # Additional context words - EXPANDED
            'wait', 'waits', 'waiting', 'any time', 'anytime', 'sometime',
            'tornado', 'tornadoes', 'storm', 'storms', 'weather',
            'update', 'updates', 'news', 'information', 'info', 'data',
            'stars', 'star', 'rating', 'ratings', 'score', 'scores',
            'ps', 'fyi', 'btw', 'imo', 'imho', 'tbh', 'ngl',
            'explanation', 'explanations', 'description', 'descriptions',
            'check', 'checks', 'bill', 'bills', 'receipt', 'receipts',
            'cash', 'money', 'dollar', 'dollars', 'cent', 'cents', 'price', 'prices',
            'cost', 'costs', 'expense', 'expenses', 'fee', 'fees', 'charge', 'charges',
            'speed', 'light', 'lighting', 'brightness', 'shadow', 'shadows',
            'order', 'orders', 'ordering', 'purchase', 'purchases',
            'selection', 'selections', 'attention', 'attentions',
            'detail', 'details', 'surprise', 'surprises', 'shock', 'shocks',
            'tip', 'tips', 'pro tip', 'hint', 'hints', 'clue', 'clues',
            'dessert', 'desserts', 'drink', 'drinks', 'beverage', 'beverages',
            'topping', 'toppings', 'add-on', 'add-ons', 'extra', 'extras',
            'wall', 'walls', 'grass wall', 'ceiling', 'ceilings', 'floor', 'floors',
            'effect', 'effects', 'swirling effect', 'lid', 'lids', 'cap', 'caps',
            'slices', 'slice', 'chunk', 'chunks', 'portion', 'portions',
            'volleyball', 'sport', 'sports', 'game', 'games',
            'dinner', 'lunch', 'breakfast', 'brunch', 'meal', 'meals',
            'refreshment', 'refreshments', 'snack', 'snacks',
            'cereal', 'cereals', 'grain', 'grains', 'real deal', 'deal',
            'sweets', 'sweet', 'pace', 'pacing', 'handout', 'handouts',
            'timing', 'concept', 'concepts', 'idea', 'ideas', 'notion', 'notions',
            'kind', 'kinds', 'beauty', 'beauties', 'sugar rush', 'rush', 'rushes',
            'lightbulb', 'lightbulbs', 'bulb', 'bulbs', 'light bulb',
            'word', 'words', 'phrase', 'phrases', 'sentence', 'sentences',
            'training', 'trainings', 'lesson', 'lessons', 'course', 'courses',
            'least', 'most', 'heat', 'cold', 'temperature', 'temperatures',
            'covid', 'covid19', 'pandemic', 'virus', 'disease',
            'masks', 'mask', 'face mask', 'glove', 'gloves',
            'top', 'tops', 'bottom', 'bottoms', 'middle', 'center',
            'balance', 'balances', 'equilibrium', 'harmony',
            'undertone', 'undertones', 'overtone', 'overtones', 'note', 'notes',
            'kick', 'kicks', 'punch', 'punches', 'zing', 'zings',
            'plus point', 'minus point', 'pro', 'pros', 'con', 'cons',
            'chewiness', 'crunchiness', 'crispiness', 'softness', 'hardness',
            'richness', 'lightness', 'heaviness', 'thickness', 'thinness',
            'traffic', 'foot traffic', 'crowd', 'crowds', 'rush', 'line', 'lines',
            'list', 'lists', 'menu', 'menus', 'catalog', 'catalogs',
            'amount', 'amounts', 'quantity', 'quantities', 'volume', 'volumes',
            'tubs', 'tub', 'container', 'containers', 'package', 'packages',
            'refunds', 'refund', 'refund policy', 'policy', 'policies', 'rule', 'rules',
            'artisanal varieties', 'variety', 'varieties',
            'picky sticks', 'stick', 'sticks',
            'colors', 'color', 'hue', 'hues', 'shade', 'shades', 'tint', 'tints',
            'flavors', 'flavor', 'flavour', 'flavours', 'flavoring', 'taste', 'tastes',
            'photos', 'photo', 'pictures', 'picture', 'pic', 'pics', 'image', 'images',
            'default', 'defaults', 'standard', 'standards', 'norm', 'norms',
            'rest', 'remainder', 'leftovers', 'leftover',
            'fan', 'fans', 'fanatic', 'fanatics', 'enthusiast', 'enthusiasts',
            'cartoons', 'cartoon', 'animation', 'animations',
            'saturday morning', 'saturday', 'sunday', 'monday', 'tuesday',
            'wednesday', 'thursday', 'friday',
            'big bowl', 'small bowl', 'medium bowl',
            'news', 'article', 'articles', 'story', 'stories', 'report', 'reports',
            'milkshakes', 'milkshake', 'shake', 'shakes',
            'brownie bites', 'bite', 'bites', 'morsel', 'morsels',
            'setting', 'settings', 'environment', 'environments', 'atmosphere',
            'inside', 'outside', 'interior', 'exterior', 'indoor', 'outdoor',
            'options', 'option', 'alternative', 'alternatives',
            'mix in', 'mix ins', 'mix-in', 'mix-ins', 'add-in', 'add-ins',
            'base', 'bases', 'foundation', 'foundations',
            'cooking', 'baking', 'preparation', 'prep',
            'situation', 'situations', 'circumstance', 'circumstances',
            'disinfectants', 'disinfectant', 'cleaner', 'cleaners', 'sanitizer',
            'substitutes', 'substitute', 'replacement', 'replacements',
            'sushi burrito', 'burrito', 'burritos',
            'almond slices', 'teddy grahams', 'graham', 'grahams',
            'shop', 'shops', 'shopping', 'shopper', 'shoppers',
            'sample', 'samples', 'sampling', 'taster', 'tasters',
            'treat', 'treats', 'goodie', 'goodies', 'delight', 'delights',
            'batch', 'batches', 'lot', 'lots', 'bunch', 'bunches',
            'set', 'sets', 'collection', 'collections', 'assortment', 'assortments',
            'range', 'ranges', 'array', 'arrays', 'lineup', 'lineups',
            'combo', 'combos', 'combination', 'combinations', 'pairing', 'pairings',
            'version', 'versions', 'variant', 'variants', 'edition', 'editions',
            'style', 'styles', 'fashion', 'trend', 'trends',
            'method', 'methods', 'technique', 'techniques', 'approach', 'approaches',
            'way', 'ways', 'manner', 'manners', 'mode', 'modes',
            'form', 'forms', 'shape', 'shapes', 'format', 'formats',
            'design', 'designs', 'pattern', 'patterns', 'layout', 'layouts',
            'theme', 'themes', 'motif', 'motifs', 'decor', 'decoration',
            'look', 'looks', 'appearance', 'appearances', 'aesthetic', 'aesthetics',
            'ambiance', 'ambience', 'atmosphere', 'atmospheres',
            'change', 'changes', 'modification', 'modifications', 'adjustment', 'adjustments',
            'difference', 'differences', 'distinction', 'distinctions', 'contrast', 'contrasts',
            'similarity', 'similarities', 'resemblance', 'parallel', 'parallels',
            'level', 'levels', 'degree', 'degrees', 'extent', 'extents',
            'measure', 'measures', 'measurement', 'measurements',
            'rate', 'rates', 'ratio', 'ratios', 'proportion', 'proportions',
            'percentage', 'percentages', 'percent', 'fraction', 'fractions',
            'value', 'values', 'worth', 'merit', 'merits',
            'benefit', 'benefits', 'advantage', 'advantages', 'plus', 'pluses',
            'drawback', 'drawbacks', 'disadvantage', 'disadvantages', 'minus', 'minuses',
            'strength', 'strengths', 'weakness', 'weaknesses',
            'positive', 'positives', 'negative', 'negatives',
            'upside', 'upsides', 'downside', 'downsides',
        }

        if normalized in filler_terms:
            return False

        # Filter ingredients when standalone - EXPANDED
        ingredient_terms = {
            'bananas', 'banana', 'graham', 'crackers', 'graham crackers',
            'pecans', 'pecan', 'peanuts', 'peanut', 'nuts', 'nut',
            'chocolate', 'choco', 'cocoa', 'cacao',
            'cookie dough', 'dough', 'cookie', 'cookies',
            'vanilla', 'vanillas', 'milk', 'cream', 'creams',
            'splenda', 'sugar', 'sugars', 'sweetener', 'sweeteners',
            'aftertaste', 'gummy bears', 'gummy', 'gummies',
            'pocky sticks', 'pocky', 'shavings', 'chocolate shavings',
            'puree', 'purees', 'blueberry puree', 'blueberry',
            'sherbet', 'sherbets', 'sorbet', 'sorbets',
            'creamsicle', 'creamsicles', 'popsicle', 'popsicles',
            'frosted flakes', 'flakes', 'cereal',
            'reeses', 'reeses puff', 'reese', 'puff', 'puffs',
            'peanut butter', 'butter', 'butters',
            'ginger', 'lemon', 'lemons', 'lime', 'limes',
            'honey', 'honeys', 'syrup', 'syrups',
            'caramel', 'caramels', 'toffee', 'toffees',
            'peach', 'peaches', 'lychee', 'lychees',
            'strawberry', 'strawberries', 'berry', 'berries',
            'raspberry', 'raspberries', 'blackberry', 'blackberries',
            'blueberry', 'blueberries', 'cherry', 'cherries',
            'matcha', 'tea', 'teas', 'green tea',
            'coconut', 'coconuts', 'coco', 'aloe vera', 'aloe',
            'jasmine', 'hojicha', 'oolong', 'earl grey',
            'captain crunch berries', 'captain crunch', 'crunch',
            'apple jacks', 'apple', 'apples', 'jacks',
            'cinnamon toast', 'cinnamon', 'toast',
            'fruity pebbles', 'pebbles', 'fruity',
            'pineapple', 'pineapples', 'crush',
            'crystal boba', 'boba', 'bobas', 'tapioca',
            'black boba', 'pearls', 'pearl',
            'creme', 'crème', 'whipped cream', 'whip',
            'sprinkles', 'sprinkle', 'jimmies',
            'marshmallow', 'marshmallows', 'mallow', 'mallows',
            'fudge', 'hot fudge', 'brownie', 'brownies',
            'wafer', 'wafers', 'waffle', 'waffles',
            'pretzel', 'pretzels', 'chip', 'chips',
            'oreo', 'oreos', 'nutter butter', 'snickers',
            'kitkat', 'kit kat', 'twix', 'milky way',
            'mango', 'mangoes', 'papaya', 'papayas',
            'passion fruit', 'passion', 'guava', 'guavas',
            'kiwi', 'kiwis', 'orange', 'oranges',
            'grape', 'grapes', 'watermelon', 'melon', 'melons',
            'mint', 'mints', 'peppermint', 'spearmint',
            'lavender', 'rose', 'roses', 'hibiscus',
            'almond', 'almonds', 'walnut', 'walnuts',
            'hazelnut', 'hazelnuts', 'pistachio', 'pistachios',
            'cashew', 'cashews', 'macadamia', 'macadamias',
            'salt', 'salts', 'sea salt', 'pepper', 'peppers',
            'spice', 'spices', 'herb', 'herbs',
            'extract', 'extracts', 'essence', 'essences',
            'powder', 'powders', 'dust', 'dusts',
            'sauce', 'sauces', 'drizzle', 'drizzles',
            'topping', 'mix-in', 'ingredient', 'ingredients',
        }
        if normalized in ingredient_terms:
            return False

        # Filter phrases with personal indicators
        personal_patterns = ['my ', 'our ', 'your ', 'his ', 'her ', 'this ', 'these ',
                             'those ', "what's ", 'their ', 'every ', 'one of', "i'm ",
                             'its ', 'some ', 'any ', 'each ', 'all ', 'both ',
                             'either ', 'neither ', 'another ', 'other ']
        if any(normalized.startswith(pattern) for pattern in personal_patterns):
            return False

        # Filter if contains problematic terms
        problematic_terms = ['tolerance', 'husband', 'wife', 'girl', 'boy', 'regards', 'club',
                             'myself', 'yourself', 'himself', 'herself', 'ourselves', 'themselves',
                             'review', 'reviews', 'rating', 'ratings', 'haha', 'lol', 'lmao',
                             'google', 'facebook', 'instagram', 'twitter', 'star', 'stars',
                             'defo', 'definitely', 'hmm', 'umm', 'uhh',
                             'us', 'we', 'them', 'complaint', 'critique', 'criticism',
                             'hair', 'brother', 'sister', 'mother', 'father',
                             'law', 's/o', 'shoutout', 'pickup', 'article', 'girlfriend',
                             'boyfriend', 'valentine', 'socks', 'neighborhood', 'favorite',
                             'family', 'friend', 'buddy', 'pal', 'person', 'people']
        if any(term in normalized for term in problematic_terms):
            return False

        # Filter temporal/seasonal/descriptive phrases - EXPANDED
        if any(normalized.startswith(prefix) for prefix in [
            'fall ', 'winter ', 'spring ', 'summer ', 'autumn ',
            'season ', 'seasonal ', 'special ', 'rotating ', 'featured ',
            'pro ', 'mix in', 'same ', 'good ', 'great ', 'best ',
            'bit ', 'kind of', 'only ', 'lot of', 'lots of',
            'much ', 'many ', 'more ', 'most ', 'less ', 'least ',
            'new ', 'old ', 'hard ', 'soft ', 'other ', 'both ',
            'hand made', 'handmade', 'homemade', 'home made',
            'second ', 'first ', 'third ', 'last ', 'next ',
            'every ', 'each ', 'all ', 'some ', 'any ',
            'very ', 'really ', 'super ', 'ultra ', 'mega ',
            'quite ', 'pretty ', 'rather ', 'fairly ', 'too ',
            'so ', 'such ', 'that ', 'this ', 'these ', 'those ',
        ]):
            return False

        # Filter "what" phrases
        if normalized.startswith('what'):
            return False

        # Filter numeric/time/price expressions - EXPANDED
        if re.search(
                r'(\d+\s*(minute|min|hour|hr|day|week|month|year|mile|km|star|dollar|\$|%|ft|meter|m|cm|inch|in|lb|oz|kg|g)|1970s|1980s|1990s|2000s|50%|75%|25%|30mins|5stars)',
                normalized, re.IGNORECASE):
            return False

        # Filter standalone adjectives
        if self._get_pos(aspect) == 'ADJ' and len(normalized.split()) == 1:
            return False

        # Filter proper nouns that are locations or brand names - EXPANDED
        if self._get_pos(aspect) == 'PROPN':
            filtered_names = {
                'urbana', 'monticello', 'westville', 'champaign', 'sidney',
                'barn', 'dairy', 'sages', 'walmart', 'target', 'costco',
                'facebook', 'instagram', 'twitter', 'google', 'yelp', 'tripadvisor',
                'baskin', 'robbins', 'jarlings', 'rewind', 'dripps',
                'pokitrition', 'culvers', 'firehouse', 'dq', 'mcdonalds',
                'starbucks', 'dunkin', 'subway', 'wendys', 'arbys',
                'tempe', 'chandler', 'phx', 'phoenix', 'scottsdale',
                'mesa', 'glendale', 'peoria', 'gilbert',
                'bay', 'plaza', 'mall', 'az', 'arizona',
                'california', 'texas', 'florida', 'york', 'illinois',
                'covid', 'covid19', 'coronavirus',
                'japan', 'china', 'korea', 'thailand', 'vietnam',
                'mexico', 'canada', 'usa', 'america', 'europe',
                'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                'saturday', 'sunday', 'january', 'february', 'march',
                'april', 'may', 'june', 'july', 'august',
                'september', 'october', 'november', 'december',
            }
            if normalized in filtered_names:
                return False

            # Filter standalone size/quantity descriptors
        size_terms = {'small', 'medium', 'large', 'xl', 'xxl', 'big', 'huge', 'tiny', 'mini'}
        if normalized in size_terms:
            return False

        # Filter measurement units
        units = {'oz', 'ounce', 'ounces', 'lb', 'lbs', 'pound', 'pounds', 'kg', 'gram', 'grams',
                 'ml', 'liter', 'liters', 'gallon', 'gallons', 'cup', 'cups', 'pint', 'pints',
                 'quart', 'quarts', 'tbsp', 'tsp', 'tablespoon', 'teaspoon'}
        if normalized in units:
            return False

        return True

    def _normalize_aspect(self, aspect_text: str) -> str:
        """Remove possessives/pronouns/leading modifiers and trailing punctuation."""
        text = aspect_text.strip()

        # Normalize case first
        text = text.lower()

        # Remove "what a/what's/what is" constructions
        text = re.sub(r'^what(\s+a|\s+an|\s+is|\'s)\s+', '', text, flags=re.IGNORECASE)

        # Remove leading descriptive adjectives - EXPANDED
        text = re.sub(
            r'^(great|fun|nice|good|bad|amazing|awesome|excellent|fantastic|wonderful|'
            r'terrible|horrible|awful|poor|mediocre|decent|okay|ok|fine|'
            r'little|local|best|worst|delicious|tasty|yummy|gross|nasty|'
            r'cute|plain|simple|complex|complicated|basic|advanced|'
            r'ample|sufficient|inadequate|excessive|moderate|'
            r'strange|weird|odd|unusual|normal|typical|common|rare|'
            r'near|close|far|distant|nearby|adjacent|'
            r'constant|frequent|occasional|rare|continuous|'
            r'long|short|brief|extended|lengthy|'
            r'fresh|stale|old|new|recent|ancient|modern|'
            r'truly|really|very|super|ultra|mega|extremely|'
            r'made|handmade|homemade|hand-made|home-made|'
            r'some|this|that|these|those|such|'
            r'same|similar|different|identical|unique|'
            r'many|much|few|little|several|numerous|'
            r'about|around|approximately|roughly|nearly|'
            r'any|each|every|all|both|either|neither|'
            r'entire|whole|complete|full|partial|half|'
            r'home|outdoor|indoor|inside|outside|'
            r'typical|traditional|classic|modern|contemporary|'
            r'american|asian|european|mexican|italian|chinese|japanese|'
            r'private|public|personal|professional|'
            r'festive|casual|formal|fancy|plain|'
            r'picture|photo|instagram|'
            r'perfect|imperfect|flawless|flawed|'
            r'small|medium|large|huge|tiny|massive|enormous|gigantic|'
            r'trendy|stylish|fashionable|outdated|modern|'
            r'layered|stacked|piled|heaped|'
            r'exceptionally|incredibly|unbelievably|remarkably|'
            r'insanely|crazy|wildly|ridiculously|absurdly|'
            r'pretty|fairly|rather|quite|somewhat|slightly|'
            r'real|fake|authentic|genuine|artificial|synthetic|'
            r'female|male|unisex|gender|'
            r'biggest|smallest|largest|tiniest|hugest|'
            r'flat|round|square|circular|rectangular|'
            r'out|super|extra|double|triple|single|'
            r'regular|normal|standard|ordinary|usual|special|'
            r'original|copy|replica|duplicate|'
            r'yummy|delish|scrumptious|divine|heavenly|'
            r'sounding|looking|seeming|appearing|'
            r'delicate|robust|strong|weak|mild|intense|'
            r'lightly|heavily|moderately|slightly|'
            r'cool|warm|hot|cold|frozen|chilled|'
            r'perfect|ideal|optimal|suboptimal|'
            r'bit|various|assorted|mixed|varied|diverse|'
            r'only|sole|single|lone|solitary|'
            r'tiny|minuscule|microscopic|giant|colossal|'
            r'plus|minus|positive|negative|'
            r'surprising|expected|unexpected|predictable|'
            r'surprisingly|unexpectedly|predictably|'
            r'popular|unpopular|famous|unknown|obscure|'
            r'simple|easy|difficult|hard|challenging|'
            r'limited|unlimited|restricted|unrestricted|'
            r'new|old|ancient|modern|contemporary|vintage|'
            r'hard|soft|firm|tender|tough|gentle|'
            r'chewy|crunchy|crispy|smooth|creamy|'
            r'polite|rude|courteous|discourteous|respectful|'
            r'expressive|bland|boring|exciting|dull|'
            r'subtle|obvious|apparent|hidden|'
            r'distinct|indistinct|clear|vague|'
            r'earthy|airy|light|heavy|dense|'
            r'speedy|slow|fast|quick|rapid|sluggish|'
            r'friendly|unfriendly|warm|welcoming|hostile|'
            r'wide|narrow|broad|slim|thick|thin|'
            r'affordable|expensive|cheap|pricey|costly|'
            r'fantastic|terrible|horrible|wonderful|'
            r'impressed|disappointed|satisfied|unsatisfied|'
            r'whole|entire|complete|partial|incomplete|'
            r'aromatic|fragrant|smelly|odorless|'
            r'buggy|glitchy|smooth|seamless|'
            r'slow|fast|quick|rapid|speedy|'
            r'cute|adorable|charming|lovely|'
            r'little|small|tiny|miniature|petite|'
            r'tasty|flavorful|bland|tasteless|'
            r'fun|boring|entertaining|dull|'
            r'overall|general|specific|particular|'
            r'decent|acceptable|satisfactory|unsatisfactory|'
            r'quick|rapid|swift|slow|leisurely|'
            r'young|old|elderly|youthful|aged|'
            r'fabulous|marvelous|spectacular|magnificent|'
            r'hand|manual|automatic|mechanical|'
            r'dense|sparse|concentrated|diluted|'
            r'creamy|watery|liquid|solid|'
            r'sweet|sour|bitter|salty|savory|umami|'
            r'quickly|slowly|rapidly|gradually|'
            r'big|large|huge|enormous|gigantic|'
            r'ol|ole|old)\s+',
            '', text, flags=re.IGNORECASE
        )

        # Remove leading articles
        text = re.sub(r'^(a|an|the)\s+', '', text, flags=re.IGNORECASE)

        # Remove leading possessives/pronouns - EXPANDED
        text = re.sub(r'^(my|their|our|your|his|her|its|one of|some of|all of|most of|many of|few of|several of)\s+',
                      '', text, flags=re.IGNORECASE)

        # Remove leading intensifiers + adjectives - EXPANDED
        text = re.sub(
            r'^(very|really|so|super|quite|extremely|incredibly|unbelievably|'
            r'too|way|pretty|fairly|rather|somewhat|slightly|'
            r'a bit|a little|kind of|sort of|lot of|lots of|'
            r'even though|although|though|however|but|yet|still|'
            r'much|more|most|less|least|fewer|fewest|'
            r'absolutely|totally|completely|entirely|utterly|thoroughly|'
            r'both|either|neither|all|any|some|each|every|'
            r'always|never|sometimes|often|rarely|seldom|frequently|'
            r'especially|particularly|specifically|generally|usually)\s+'
            r'(good|bad|tasty|nice|sweet|bitter|sour|salty|savory|'
            r'delicious|gross|nasty|yummy|bland|flavorful|'
            r'friendly|rude|polite|courteous|helpful|unhelpful|'
            r'artificial|natural|real|fake|authentic|genuine|'
            r'cute|adorable|lovely|beautiful|ugly|hideous|'
            r'clean|dirty|messy|tidy|neat|organized|'
            r'soft|hard|firm|tender|tough|chewy|crunchy|'
            r'watery|creamy|smooth|chunky|lumpy|'
            r'rude|polite|kind|mean|nice|nasty|'
            r'fast|slow|quick|rapid|sluggish|speedy|'
            r'expensive|cheap|affordable|pricey|costly|'
            r'fresh|stale|old|new|rotten|spoiled)\s+',
            '', text, flags=re.IGNORECASE
        )

        # Remove remaining single intensifiers
        text = re.sub(
            r'^(very|really|so|super|quite|extremely|incredibly|unbelievably|'
            r'too|way|pretty|fairly|rather|somewhat|slightly|'
            r'a bit|a little|kind of|sort of|lot of|lots of|plenty of|'
            r'even though|although|though|however|but|yet|still|nevertheless|'
            r'much|more|most|less|least|fewer|fewest|'
            r'absolutely|totally|completely|entirely|utterly|thoroughly|fully|'
            r'both|either|neither|all|any|some|each|every|another|other|'
            r'always|never|sometimes|often|rarely|seldom|frequently|occasionally|'
            r'especially|particularly|specifically|generally|usually|normally|typically)\s+',
            '', text, flags=re.IGNORECASE
        )

        # Remove temporal/season descriptors - EXPANDED
        text = re.sub(r'^(fall|winter|spring|summer|autumn|seasonal|'
                      r'season|special|featured|rotating|limited|exclusive|'
                      r'near|nearby|close|far|distant|'
                      r'constant|frequent|occasional|rare|'
                      r'late|early|mid|'
                      r'night|day|morning|afternoon|evening|'
                      r'daily|weekly|monthly|yearly|annual)\s+', '', text,
                      flags=re.IGNORECASE)

        # Remove "small town/ice cream/bubble tea" before nouns
        text = re.sub(r'^(small\s+town|big\s+city|small|medium|large|huge|tiny|'
                      r'ice\s+cream|bubble\s+tea|boba\s+tea|sweet\s+tea|iced\s+tea|'
                      r'foot|hand|finger|body)\s+', '', text,
                      flags=re.IGNORECASE)

        # Remove business names - EXPANDED
        text = re.sub(r'^(dairy\s+barn|sidney\s+dairy\s+barn|rewind|dripps|'
                      r'baskin\s+robbins|cold\s+stone|marble\s+slab|'
                      r'ben\s+jerry|haagen\s+dazs)\s*', '', text, flags=re.IGNORECASE)

        # Remove corporate/local/vegan/artisanal descriptors - EXPANDED
        text = re.sub(r'^(corporate|chain|franchise|franchised|'
                      r'local|locally|regional|national|international|'
                      r'vegan|vegetarian|non\s+vegan|non-vegan|'
                      r'organic|natural|artificial|synthetic|'
                      r'fresh|stale|frozen|chilled|'
                      r'artisanal|gourmet|premium|luxury|basic|standard|'
                      r'hard\s+scoop|soft\s+serve|'
                      r'hand\s+made|handmade|hand-made|'
                      r'home\s+made|homemade|home-made|'
                      r'house\s+made|housemade|house-made)\s+', '',
                      text, flags=re.IGNORECASE)

        # Remove price/online indicators - EXPANDED
        text = re.sub(r'^(1970s|1980s|1990s|2000s|retro|vintage|classic|'
                      r'cheap|expensive|pricey|costly|affordable|reasonable|'
                      r'pricier|cheaper|budget|premium|'
                      r'online|offline|digital|virtual|physical|'
                      r'takeout|take-out|dine-in|dine\s+in|delivery)\s+', '', text, flags=re.IGNORECASE)

        # Remove "with" constructions like "with mix-ins"
        text = re.sub(r'\s+with\s+.*$', '', text, flags=re.IGNORECASE)

        # Remove "for" constructions like "for dessert"
        text = re.sub(r'\s+for\s+.*$', '', text, flags=re.IGNORECASE)

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

                    if (single_i in candidate_norm.lower() and
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