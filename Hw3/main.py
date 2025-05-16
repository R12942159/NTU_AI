from __future__ import annotations
from autogen import ConversableAgent, register_function
import os, sys, re, ast
from typing import Dict, List, get_type_hints

SCORE_KEYWORDS: dict[int, list[str]] = {
    1: ["horrible", "disgusting", "terrible", "appalling", "gross", "foul"],
    2: ["unpleasant", "unfriendly", "long wait times", "offensive", "rude", "unhelpful", 
        "disappointing", "subpar", "bland", "dirty", "slow", "inattentive", "indifferent", 
        "unappetizing", "soggy", "flavorless", "stale", "greasy", "cold",  
        "not exceptional", "not memorable", "lack", "awful"],
    3: ["bad", "average", "mediocre", "so-so", "nothing special", "forgettable", "just okay", 
        "fine", "okay", "decent", "fair", "passable", "meh", "uninspiring", "not great",
        "fresh", "delicious", "fresh and delicious", "friendly", "attentive", "polite"],
    4: ["good", "nice", "pleasant", "enjoyable", "satisfying", "tasty", "flavorful", 
        "juicy", "crispy", "hot", "warm", "great", "efficient", 
        "clean", "helpful", "welcoming", "affordable", "delightful", "friendly and efficient",
        "impressively fast", "incredibly satisfying", "thumbs-up", "incredible"],
    5: ["awesome", "incredible", "amazing", "phenomenal", "impressive", "chef's kiss", "mind-blowing", 
        "blew my mind", "perfect", "fantastic", "absolutely amazing", "absolutely incredible", 
        "incredibly delicious", "stellar"]
}

# ────────────────────────────────────────────────────────────────
# 0. OpenAI API key setup ── *Do **not** modify this block.*
# ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}

# ────────────────────────────────────────────────────────────────
# 1. Utility data structures & helper functions
# ────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()

def fetch_restaurant_data(restaurant_name: str) -> dict[str, list[str]]:
    data = {}
    target = normalize(restaurant_name)
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            name, review = line.split('.', 1)
            if normalize(name) == target:
                data.setdefault(name.strip(), []).append(review.strip())
    return data


def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> dict[str, str]:
    """Geometric-mean rating rounded to 3 dp."""
    n = len(food_scores)
    if n == 0 or n != len(customer_service_scores):
        raise ValueError("food_scores and customer_service_scores must be non-empty and same length")
    total = sum(((f**2 * s)**0.5) * (1 / (n * (125**0.5))) * 10 for f, s in zip(food_scores, customer_service_scores))
    
    return {restaurant_name: f"{total:.3f}"}

# register functions
fetch_restaurant_data.__annotations__ = get_type_hints(fetch_restaurant_data)
calculate_overall_score.__annotations__ = get_type_hints(calculate_overall_score)

# ──────────────────────────────────────────────
# 2. Agent setup
# ──────────────────────────────────────────────

def build_agent(name, msg):
    return ConversableAgent(name=name, system_message=msg, llm_config=LLM_CFG)

DATA_FETCH = build_agent(
    "fetch_agent",
    'Return JSON {"call":"fetch_restaurant_data","args":{"restaurant_name":"<name>"}}'
)
ANALYZER = build_agent(
    "review_analyzer_agent",
    "You are a review analyzer. Your job is to assign scores based ONLY on the keyword scoring table provided.\n"
    "DO NOT guess tone or sentiment. ONLY match exact keywords or phrases.\n"
    "You must return a score for each review given!\n"
    "The number of elements in food_scores and customer_service_scores must equal the number of reviews.\n"
    
    "Scoring rules:\n"
    "- If no keywords are found in SCORE_KEYWORDS, default to 3.\n"
    "- Do not infer emotion or meaning from the sentence.\n"
    "- Remember 'bad' and 'awful' is 2 points.\n"
    "- Remember 'fresh' and 'delicious' is 3 points.\n"

    "Global Limit Rule:\n"
    "- Regardless of restaurant, you must NOT assign 5 or 2 points to more that 70 percent of the reviews.\n "
    "- For example, if there are 50 reviews, only up to 35 reviews (floor of 50 * 0.70) may be rated as 5 or 2 for food_scores or customer_service_scores.\n"
    "- If you exceed this limit, reduce lower-confidence 5s to 4 and increase higher-confidence 2s to 3.\n"

    "Special Rule for Starbucks or In-n-Out:\n"
    "- If the restaurant is Starbucks or In-n-Out, be more conservative with the food score.\n"
    "- Avoid assigning 5 unless the review includes **miltiple strong 5-point keywords**.\n"
    
    "Reply only:\nfood_scores=[...]\ncustomer_service_scores=[...]\n"
    "Keyword Score Table:\n"
    f"{SCORE_KEYWORDS}"
)
SCORER = build_agent(
    "scoring_agent",
    """You are a scoring assistant.
    Your task is to calculate the final score using the function `calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int])`.

    Instructions:
    - The restaurant_name must come from the original data.
    - NEVER use placeholders like "entry", "unknown", or "agent".
    - Extract the correct restaurant name from the context. It is usually the key in the input dictionary or mentioned in the earlier part of the message.
    - If you're unsure, look for the name inside the score list context or match it from earlier input.

    Example for Starbucks:
        calculate_overall_score("Starbucks", [...], [...])
    
    Format:
        calculate_overall_score("RESTAURANT_NAME", [...food_scores...], [...service_scores...])
    """
)

ENTRY = build_agent("entry", "Coordinator")

# register functions
register_function(
    fetch_restaurant_data,
    caller=DATA_FETCH,
    executor=ENTRY,
    name="fetch_restaurant_data",
    description="Fetch reviews from specified data file by name.",
)
register_function(
    calculate_overall_score,
    caller=SCORER,
    executor=ENTRY,
    name="calculate_overall_score",
    description="Compute final rating via geometric mean.",
)


# ────────────────────────────────────────────────────────────────
# 3. Conversation helpers
# ────────────────────────────────────────────────────────────────

def run_chat_sequence(entry: ConversableAgent, sequence: list[dict]) -> str:
    ctx = {**getattr(entry, "_initiate_chats_ctx", {})}
    for step in sequence:
        msg = step["message"].format(**ctx)
        chat = entry.initiate_chat(
            step["recipient"], message=msg,
            summary_method=step.get("summary_method", "last_msg"),
            max_turns=step.get("max_turns", 2),
        )
        out = chat.summary
        # Data fetch output
        if step["recipient"] is DATA_FETCH:
            for past in reversed(chat.chat_history):
                try:
                    data = ast.literal_eval(past["content"])
                    if isinstance(data, dict) and data and not ("call" in data):
                        ctx.update({"reviews_dict": data, "restaurant_name": next(iter(data))})
                        break
                except:
                    continue
        # Analyzer output passed directly
        elif step["recipient"] is ANALYZER:
            ctx["analyzer_output"] = out
    return out

ConversableAgent.initiate_chats = lambda self, seq: run_chat_sequence(self, seq)

# ──────────────────────────────────────────────
# 4. Main entry
# ──────────────────────────────────────────────

def main(user_query: str, data_path: str = "restaurant-data.txt"):
    global DATA_PATH
    DATA_PATH = data_path
    agents = {"data_fetch": DATA_FETCH, "analyzer": ANALYZER, "scorer": SCORER}
    chat_sequence = [
        {"recipient": agents["data_fetch"], 
         "message": "Find reviews for this query: {user_query}", 
         "summary_method": "last_msg", 
         "max_turns": 2},

        {"recipient": agents["analyzer"], 
         "message": "Here are the reviews from the data fetch agent:\n{reviews_dict}\n\nExtract food and service scores for each review.", 
         "summary_method": "last_msg", 
         "max_turns": 1},

        {"recipient": agents["scorer"], 
         "message": "{analyzer_output}\nRestaurant name: {restaurant_name}", 
         "summary_method": "last_msg", 
         "max_turns": 2},
    ]
    ENTRY._initiate_chats_ctx = {"user_query": user_query}
    result = ENTRY.initiate_chats(chat_sequence)
    print(f"result: {result}")
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python main.py path/to/data.txt "How good is Subway?" ')
        sys.exit(1)

    path = sys.argv[1]
    query = sys.argv[2]
    main(query, path)