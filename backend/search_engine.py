# # import torch
# # import clip
# # import numpy as np
# # import faiss
# # from PIL import Image
# # import os
# # import logging

# # logger = logging.getLogger("SearchEngine")

# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # CLIP_MODEL = "ViT-B/32"

# # # Extended emoji to descriptive phrase mapping
# # EMOJI_MAP = {
# #     "🐶": "dog", "🐱": "cat", "🏖️": "beach", "🎂": "birthday cake",
# #     "🍕": "pizza food", "🚗": "car vehicle", "🌲": "trees nature forest",
# #     "🏙️": "city skyline", "🌅": "sunset", "🌊": "ocean waves",
# #     "🏔️": "mountain", "🎉": "party celebration", "👶": "baby child",
# #     "🐦": "bird", "🌸": "flowers", "🌙": "night sky moon",
# #     "☀️": "sunny day", "❄️": "snow winter", "🌧️": "rain",
# #     "🏠": "house home", "✈️": "airplane travel", "🍔": "burger food",
# #     "🎵": "music concert", "📚": "books reading", "🏋️": "gym workout"
# # }

# # # Common synonym expansions to improve recall
# # QUERY_SYNONYMS = {
# #     "dog": "dog puppy canine",
# #     "cat": "cat kitten feline",
# #     "baby": "baby infant toddler child",
# #     "car": "car vehicle automobile",
# #     "food": "food meal eating restaurant",
# #     "sunset": "sunset golden hour dusk",
# #     "beach": "beach ocean sea shore",
# #     "mountain": "mountain hill hiking",
# #     "party": "party celebration event gathering",
# #     "selfie": "selfie portrait face",
# #     "snow": "snow winter cold",
# #     "flower": "flower bloom garden",
# #     "night": "night dark evening",
# #     "wedding": "wedding ceremony bride groom",
# #     "travel": "travel trip vacation journey",
# #     "forest": "forest trees woods nature",
# # }


# # class SearchEngine:
# #     def __init__(self, dimension=512):
# #         self.dimension = dimension
# #         self.model = None
# #         self.preprocess = None
# #         self.index = None
# #         self._load_clip()

# #     def _load_clip(self):
# #         logger.info(f"Loading CLIP {CLIP_MODEL} on {DEVICE}...")
# #         self.model, self.preprocess = clip.load(CLIP_MODEL, device=DEVICE)
# #         self.model.eval()

# #     def get_text_embedding(self, text, use_prompt_ensemble=True):
# #         """
# #         Returns a text embedding. Optionally averages across multiple photo-style
# #         prompts for better accuracy (prompt ensembling).
# #         """
# #         try:
# #             if use_prompt_ensemble:
# #                 # Craft multiple prompts and average them — significantly improves CLIP accuracy
# #                 prompts = [
# #                     f"a photo of {text}",
# #                     f"a photograph of {text}",
# #                     f"{text}",
# #                     f"an image of {text}",
# #                     f"a picture of {text}",
# #                 ]
# #                 all_features = []
# #                 for prompt in prompts:
# #                     tokens = clip.tokenize([prompt], truncate=True).to(DEVICE)
# #                     with torch.no_grad():
# #                         features = self.model.encode_text(tokens)
# #                         features /= features.norm(dim=-1, keepdim=True)
# #                     all_features.append(features.cpu().numpy().flatten())

# #                 # Average and re-normalize
# #                 avg = np.mean(all_features, axis=0)
# #                 avg /= (np.linalg.norm(avg) + 1e-10)
# #                 return avg
# #             else:
# #                 tokens = clip.tokenize([text], truncate=True).to(DEVICE)
# #                 with torch.no_grad():
# #                     features = self.model.encode_text(tokens)
# #                     features /= features.norm(dim=-1, keepdim=True)
# #                 return features.cpu().numpy().flatten()
# #         except Exception as e:
# #             logger.error(f"Error encoding text: {e}")
# #             return None

# #     def get_image_embedding(self, image_path):
# #         try:
# #             image = Image.open(image_path).convert("RGB")
# #             image_input = self.preprocess(image).unsqueeze(0).to(DEVICE)
# #             with torch.no_grad():
# #                 features = self.model.encode_image(image_input)
# #                 features /= features.norm(dim=-1, keepdim=True)
# #             return features.cpu().numpy().flatten()
# #         except Exception as e:
# #             logger.error(f"Error encoding image {image_path}: {e}")
# #             return None

# #     def hybrid_rank(self, clip_score, ocr_score=0):
# #         """
# #         Weighted hybrid score. CLIP (semantic) dominates; OCR adds a small bonus.
# #         OCR score is capped to prevent it from overwhelming semantic similarity.
# #         """
# #         # Cap OCR contribution so it only nudges, not controls ranking
# #         ocr_contribution = min(ocr_score, 0.1)
# #         return (0.85 * clip_score) + (0.15 * ocr_contribution)


# # def resolve_query(query: str) -> str:
# #     """
# #     Cleans query: maps emojis to words, expands common synonyms.
# #     """
# #     processed = query.strip()

# #     # Replace emojis
# #     for emoji, text in EMOJI_MAP.items():
# #         processed = processed.replace(emoji, text)

# #     # Expand known synonyms for better recall
# #     words = processed.lower().split()
# #     expanded_words = []
# #     for word in words:
# #         if word in QUERY_SYNONYMS:
# #             expanded_words.append(QUERY_SYNONYMS[word])
# #         else:
# #             expanded_words.append(word)

# #     return " ".join(expanded_words)


# # search_engine = SearchEngine()
# # FIXED search_engine.py
# import torch
# import clip
# import numpy as np
# import faiss
# from PIL import Image
# import os
# import logging

# logger = logging.getLogger("SearchEngine")

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CLIP_MODEL = "ViT-B/32"

# # Extended emoji to descriptive phrase mapping
# EMOJI_MAP = {
#     "🐶": "dog", "🐱": "cat", "🏖️": "beach", "🎂": "birthday cake",
#     "🍕": "pizza food", "🚗": "car vehicle", "🌲": "trees nature forest",
#     "🏙️": "city skyline", "🌅": "sunset", "🌊": "ocean waves",
#     "🏔️": "mountain", "🎉": "party celebration", "👶": "baby child",
#     "🐦": "bird", "🌸": "flowers", "🌙": "night sky moon",
#     "☀️": "sunny day", "❄️": "snow winter", "🌧️": "rain",
#     "🏠": "house home", "✈️": "airplane travel", "🍔": "burger food",
#     "🎵": "music concert", "📚": "books reading", "🏋️": "gym workout"
# }

# # Common synonym expansions to improve recall
# QUERY_SYNONYMS = {
#     "dog": "dog puppy canine",
#     "cat": "cat kitten feline",
#     "baby": "baby infant toddler child",
#     "car": "car vehicle automobile",
#     "food": "food meal eating restaurant",
#     "sunset": "sunset golden hour dusk",
#     "beach": "beach ocean sea shore",
#     "mountain": "mountain hill hiking",
#     "party": "party celebration event gathering",
#     "selfie": "selfie portrait face",
#     "snow": "snow winter cold",
#     "flower": "flower bloom garden",
#     "night": "night dark evening",
#     "wedding": "wedding ceremony bride groom",
#     "travel": "travel trip vacation journey",
#     "forest": "forest trees woods nature",
# }


# class SearchEngine:
#     def __init__(self, dimension=512):
#         self.dimension = dimension
#         self.model = None
#         self.preprocess = None
#         self.index = None
#         self._load_clip()

#     def _load_clip(self):
#         logger.info(f"Loading CLIP {CLIP_MODEL} on {DEVICE}...")
#         self.model, self.preprocess = clip.load(CLIP_MODEL, device=DEVICE)
#         self.model.eval()

#     def get_text_embedding(self, text, use_prompt_ensemble=True):
#         """
#         Returns a text embedding. Optionally averages across multiple photo-style
#         prompts for better accuracy (prompt ensembling).
#         """
#         try:
#             if use_prompt_ensemble:
#                 # Craft multiple prompts and average them — significantly improves CLIP accuracy
#                 prompts = [
#                     f"a photo of {text}",
#                     f"a photograph of {text}",
#                     f"{text}",
#                     f"an image of {text}",
#                     f"a picture of {text}",
#                 ]
#                 all_features = []
#                 for prompt in prompts:
#                     tokens = clip.tokenize([prompt], truncate=True).to(DEVICE)
#                     with torch.no_grad():
#                         features = self.model.encode_text(tokens)
#                         features /= features.norm(dim=-1, keepdim=True)
#                     all_features.append(features.cpu().numpy().flatten())

#                 # Average and re-normalize
#                 avg = np.mean(all_features, axis=0)
#                 avg /= (np.linalg.norm(avg) + 1e-10)
#                 return avg
#             else:
#                 tokens = clip.tokenize([text], truncate=True).to(DEVICE)
#                 with torch.no_grad():
#                     features = self.model.encode_text(tokens)
#                     features /= features.norm(dim=-1, keepdim=True)
#                 return features.cpu().numpy().flatten()
#         except Exception as e:
#             logger.error(f"Error encoding text: {e}")
#             return None

#     def get_image_embedding(self, image_path):
#         try:
#             image = Image.open(image_path).convert("RGB")
#             image_input = self.preprocess(image).unsqueeze(0).to(DEVICE)
#             with torch.no_grad():
#                 features = self.model.encode_image(image_input)
#                 features /= features.norm(dim=-1, keepdim=True)
#             return features.cpu().numpy().flatten()
#         except Exception as e:
#             logger.error(f"Error encoding image {image_path}: {e}")
#             return None

#     def hybrid_rank(self, clip_score, ocr_bonus=0.0, color_bonus=0.0, tag_bonus=0.0):
#         """
#         Weighted hybrid score combining CLIP similarity, OCR match bonus,
#         optional colour similarity and object/tag bonus.

#         Default weights: 60% CLIP, 20% OCR, 10% colour, 10% tag.
#         The bonus values should already be capped between 0 and 1 before
#         calling this function.
#         """
#         # Ensure scores are in [0, 1]
#         clip_score = max(0.0, min(1.0, clip_score))
#         ocr_bonus = max(0.0, min(1.0, ocr_bonus))
#         color_bonus = max(0.0, min(1.0, color_bonus))
#         tag_bonus = max(0.0, min(1.0, tag_bonus))

#         # weighted combination (favor CLIP more)
#         final_score = (
#             (0.60 * clip_score) +
#             (0.20 * ocr_bonus) +
#             (0.10 * color_bonus) +
#             (0.10 * tag_bonus)
#         )

#         return final_score


# def resolve_query(query: str) -> str:
#     """
#     Cleans query: maps emojis to words, expands common synonyms.
#     """
#     processed = query.strip()

#     # Replace emojis
#     for emoji, text in EMOJI_MAP.items():
#         processed = processed.replace(emoji, text)

#     # Expand known synonyms for better recall
#     words = processed.lower().split()
#     expanded_words = []
#     for word in words:
#         if word in QUERY_SYNONYMS:
#             expanded_words.append(QUERY_SYNONYMS[word])
#         else:
#             expanded_words.append(word)

#     return " ".join(expanded_words)


# search_engine = SearchEngine()

# import torch
# import clip
# import numpy as np
# import faiss
# from PIL import Image
# import os
# import logging

# logger = logging.getLogger("SearchEngine")

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CLIP_MODEL = "ViT-B/32"

# # Extended emoji to descriptive phrase mapping
# EMOJI_MAP = {
#     "🐶": "dog", "🐱": "cat", "🏖️": "beach", "🎂": "birthday cake",
#     "🍕": "pizza food", "🚗": "car vehicle", "🌲": "trees nature forest",
#     "🏙️": "city skyline", "🌅": "sunset", "🌊": "ocean waves",
#     "🏔️": "mountain", "🎉": "party celebration", "👶": "baby child",
#     "🐦": "bird", "🌸": "flowers", "🌙": "night sky moon",
#     "☀️": "sunny day", "❄️": "snow winter", "🌧️": "rain",
#     "🏠": "house home", "✈️": "airplane travel", "🍔": "burger food",
#     "🎵": "music concert", "📚": "books reading", "🏋️": "gym workout"
# }

# # Common synonym expansions to improve recall
# QUERY_SYNONYMS = {
#     "dog": "dog puppy canine",
#     "cat": "cat kitten feline",
#     "baby": "baby infant toddler child",
#     "car": "car vehicle automobile",
#     "food": "food meal eating restaurant",
#     "sunset": "sunset golden hour dusk",
#     "beach": "beach ocean sea shore",
#     "mountain": "mountain hill hiking",
#     "party": "party celebration event gathering",
#     "selfie": "selfie portrait face",
#     "snow": "snow winter cold",
#     "flower": "flower bloom garden",
#     "night": "night dark evening",
#     "wedding": "wedding ceremony bride groom",
#     "travel": "travel trip vacation journey",
#     "forest": "forest trees woods nature",
# }


# class SearchEngine:
#     def __init__(self, dimension=512):
#         self.dimension = dimension
#         self.model = None
#         self.preprocess = None
#         self.index = None
#         self._load_clip()

#     def _load_clip(self):
#         logger.info(f"Loading CLIP {CLIP_MODEL} on {DEVICE}...")
#         self.model, self.preprocess = clip.load(CLIP_MODEL, device=DEVICE)
#         self.model.eval()

#     def get_text_embedding(self, text, use_prompt_ensemble=True):
#         """
#         Returns a text embedding. Optionally averages across multiple photo-style
#         prompts for better accuracy (prompt ensembling).
#         """
#         try:
#             if use_prompt_ensemble:
#                 # Craft multiple prompts and average them — significantly improves CLIP accuracy
#                 prompts = [
#                     f"a photo of {text}",
#                     f"a photograph of {text}",
#                     f"{text}",
#                     f"an image of {text}",
#                     f"a picture of {text}",
#                 ]
#                 all_features = []
#                 for prompt in prompts:
#                     tokens = clip.tokenize([prompt], truncate=True).to(DEVICE)
#                     with torch.no_grad():
#                         features = self.model.encode_text(tokens)
#                         features /= features.norm(dim=-1, keepdim=True)
#                     all_features.append(features.cpu().numpy().flatten())

#                 # Average and re-normalize
#                 avg = np.mean(all_features, axis=0)
#                 avg /= (np.linalg.norm(avg) + 1e-10)
#                 return avg
#             else:
#                 tokens = clip.tokenize([text], truncate=True).to(DEVICE)
#                 with torch.no_grad():
#                     features = self.model.encode_text(tokens)
#                     features /= features.norm(dim=-1, keepdim=True)
#                 return features.cpu().numpy().flatten()
#         except Exception as e:
#             logger.error(f"Error encoding text: {e}")
#             return None

#     def get_image_embedding(self, image_path):
#         try:
#             image = Image.open(image_path).convert("RGB")
#             image_input = self.preprocess(image).unsqueeze(0).to(DEVICE)
#             with torch.no_grad():
#                 features = self.model.encode_image(image_input)
#                 features /= features.norm(dim=-1, keepdim=True)
#             return features.cpu().numpy().flatten()
#         except Exception as e:
#             logger.error(f"Error encoding image {image_path}: {e}")
#             return None

#     def hybrid_rank(self, clip_score, ocr_score=0):
#         """
#         Weighted hybrid score. CLIP (semantic) dominates; OCR adds a small bonus.
#         OCR score is capped to prevent it from overwhelming semantic similarity.
#         """
#         # Cap OCR contribution so it only nudges, not controls ranking
#         ocr_contribution = min(ocr_score, 0.1)
#         return (0.85 * clip_score) + (0.15 * ocr_contribution)


# def resolve_query(query: str) -> str:
#     """
#     Cleans query: maps emojis to words, expands common synonyms.
#     """
#     processed = query.strip()

#     # Replace emojis
#     for emoji, text in EMOJI_MAP.items():
#         processed = processed.replace(emoji, text)

#     # Expand known synonyms for better recall
#     words = processed.lower().split()
#     expanded_words = []
#     for word in words:
#         if word in QUERY_SYNONYMS:
#             expanded_words.append(QUERY_SYNONYMS[word])
#         else:
#             expanded_words.append(word)

#     return " ".join(expanded_words)


# search_engine = SearchEngine()
# FIXED search_engine.py
import torch
import clip
import numpy as np
import faiss
from PIL import Image
import os
import logging

logger = logging.getLogger("SearchEngine")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "ViT-B/32"

# Extended emoji to descriptive phrase mapping
EMOJI_MAP = {
    "🐶": "dog", "🐱": "cat", "🏖️": "beach", "🎂": "birthday cake",
    "🍕": "pizza food", "🚗": "car vehicle", "🌲": "trees nature forest",
    "🏙️": "city skyline", "🌅": "sunset", "🌊": "ocean waves",
    "🏔️": "mountain", "🎉": "party celebration", "👶": "baby child",
    "🐦": "bird", "🌸": "flowers", "🌙": "night sky moon",
    "☀️": "sunny day", "❄️": "snow winter", "🌧️": "rain",
    "🏠": "house home", "✈️": "airplane travel", "🍔": "burger food",
    "🎵": "music concert", "📚": "books reading", "🏋️": "gym workout"
}

# Common synonym expansions to improve recall.
# Keep expansions tight — over-expanding broad queries ("man", "dog") increases
# false-positive tag matches and dilutes the CLIP embedding.
QUERY_SYNONYMS = {
    # Animals — specific enough that expansion helps recall
    "dog":      "dog puppy",
    "cat":      "cat kitten",
    "horse":    "horse pony",
    "bird":     "bird",
    # Food / places
    "food":     "food meal",
    "beach":    "beach shore",
    "mountain": "mountain hill",
    "forest":   "forest trees",
    "snow":     "snow winter",
    "flower":   "flower bloom",
    # Events
    "party":    "party celebration",
    "wedding":  "wedding ceremony",
    "travel":   "travel vacation",
    # People — intentionally NOT expanded: "man", "boy", "people", "person", "woman"
    # are already specific; expanding them leads to broad false matches.
    "selfie":   "selfie portrait",
    "baby":     "baby infant",
    "sunset":   "sunset dusk",
    "car":      "car vehicle",
    "night":    "night evening",
}


class SearchEngine:
    def __init__(self, dimension=512):
        self.dimension = dimension
        self.model = None
        self.preprocess = None
        self.index = None
        self._load_clip()

    def _load_clip(self):
        logger.info(f"Loading CLIP {CLIP_MODEL} on {DEVICE}...")
        self.model, self.preprocess = clip.load(CLIP_MODEL, device=DEVICE)
        self.model.eval()

    def get_text_embedding(self, text, use_prompt_ensemble=True):
        """
        Returns a text embedding. Optionally averages across multiple photo-style
        prompts for better accuracy (prompt ensembling).
        """
        try:
            if use_prompt_ensemble:
                # Craft multiple prompts and average them — significantly improves CLIP accuracy
                prompts = [
                    f"a photo of {text}",
                    f"a photograph of {text}",
                    f"{text}",
                    f"an image of {text}",
                    f"a picture of {text}",
                ]
                all_features = []
                for prompt in prompts:
                    tokens = clip.tokenize([prompt], truncate=True).to(DEVICE)
                    with torch.no_grad():
                        features = self.model.encode_text(tokens)
                        features /= features.norm(dim=-1, keepdim=True)
                    all_features.append(features.cpu().numpy().flatten())

                # Average and re-normalize
                avg = np.mean(all_features, axis=0)
                avg /= (np.linalg.norm(avg) + 1e-10)
                return avg
            else:
                tokens = clip.tokenize([text], truncate=True).to(DEVICE)
                with torch.no_grad():
                    features = self.model.encode_text(tokens)
                    features /= features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None

    def get_image_embedding(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                features = self.model.encode_image(image_input)
                features /= features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None

    def hybrid_rank(self, clip_score, ocr_bonus=0.0, color_bonus=0.0, tag_bonus=0.0):
        """
        Weighted hybrid score.  CLIP dominates (75 %) so that OCR / tag bonuses
        can only nudge results, never override semantic relevance.

        Previous weights (60/20/10/10) let a tag match boost a semantically
        unrelated image high enough to pass the final score threshold.
        """
        clip_score  = max(0.0, min(1.0, clip_score))
        ocr_bonus   = max(0.0, min(1.0, ocr_bonus))
        color_bonus = max(0.0, min(1.0, color_bonus))
        tag_bonus   = max(0.0, min(1.0, tag_bonus))

        return (
            (0.75 * clip_score)   +
            (0.12 * ocr_bonus)    +
            (0.07 * color_bonus)  +
            (0.06 * tag_bonus)
        )


def resolve_query(query: str) -> str:
    """
    Cleans query: maps emojis to words, expands common synonyms.
    """
    processed = query.strip()

    # Replace emojis
    for emoji, text in EMOJI_MAP.items():
        processed = processed.replace(emoji, text)

    # Expand known synonyms for better recall
    words = processed.lower().split()
    expanded_words = []
    for word in words:
        if word in QUERY_SYNONYMS:
            expanded_words.append(QUERY_SYNONYMS[word])
        else:
            expanded_words.append(word)

    return " ".join(expanded_words)


search_engine = SearchEngine()