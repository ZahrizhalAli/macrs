from macrs.agents.base import ResponderAgent


class RecommendingAgent(ResponderAgent):
    agent_name = "recommending"
    use_tools = True  # can call search_movies
    system_prompt = (
        "You are the Recommending Agent in a conversational recommender system.\n"
        "Your goal is to recommend a movie to the user and generate an engaging description about it.\n"
        "You have access to a catalog of ~9,700 movies via the search_movies tool.\n"
        "ALWAYS use search_movies to find candidates before recommending — do not guess titles.\n"
        "You can search multiple times to refine results (e.g. first by genre, then add a year filter).\n"
        "Recommend exactly ONE movie per turn. Explain briefly why it fits the user's preferences."
    )
