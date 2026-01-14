from sentence_transformers import SentenceTransformer, util
from urllib.parse import urlparse

# Load AI model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Domain popularity scores (editable anytime)
DOMAIN_SCORES = {
    "medium.com": 0.9,
    "towardsdatascience.com": 0.85,
    "geeksforgeeks.org": 0.8,
    "wikipedia.org": 0.95
}

def get_domain_score(url):
    domain = urlparse(url).netloc.replace("www.", "")
    return DOMAIN_SCORES.get(domain, 0.4)  # default score

def classify_level(text):
    text = text.lower()

    beginner_keywords = [
        "introduction", "intro", "basic", "basics", "beginner",
        "simple", "overview", "what is", "explained", "for beginners"
    ]

    intermediate_keywords = [
        "tutorial", "guide", "how to", "hands-on", "practical",
        "example", "implementation", "step by step"
    ]

    advanced_keywords = [
        "advanced", "theory", "derivation", "optimization",
        "research", "paper", "mathematical", "proof", "architecture"
    ]

    beginner_score = sum(word in text for word in beginner_keywords)
    intermediate_score = sum(word in text for word in intermediate_keywords)
    advanced_score = sum(word in text for word in advanced_keywords)

    # Length-based signal
    if len(text.split()) < 25:
        beginner_score += 1
    elif len(text.split()) > 60:
        advanced_score += 1

    # Decision logic
    if advanced_score >= 2:
        return "Advanced"
    elif beginner_score >= 2:
        return "Beginner"
    else:
        return "Intermediate"

def rank_blogs(query, blogs):
    query_embedding = model.encode(query, convert_to_tensor=True)
    ranked_results = []

    for blog in blogs:
        content_embedding = model.encode(blog["snippet"], convert_to_tensor=True)
        relevance = util.cos_sim(query_embedding, content_embedding).item()

        popularity = get_domain_score(blog["link"])

        final_score = (0.7 * relevance) + (0.3 * popularity)

        level = classify_level(blog["snippet"] + " " + blog["title"])

        ranked_results.append({
            "title": blog["title"],
            "link": blog["link"],
            "snippet": blog["snippet"],
            "level": level,
            "relevance": round(relevance, 3),
            "popularity": popularity,
            "final_score": round(final_score, 3)
})


    ranked_results.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked_results
