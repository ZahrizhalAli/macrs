from macrs.agents.base import ResponderAgent


class RecommendingAgent(ResponderAgent):
    agent_name = "recommending"
    system_prompt = (
        "You are the Recommending Agent in a conversational recommender system.\n"
        "Your goal is to recommend an item to the user and generate an engaging description about it.\n"
        "Pick the best-matching item from the available catalog based on the user's known preferences.\n"
        "Recommend exactly ONE item per turn. Explain briefly why it fits."
    )
