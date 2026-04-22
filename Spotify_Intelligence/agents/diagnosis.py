import json
import re

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_FALLBACK_MODELS
from ml.features import FEATURE_LABELS
from agents.tools import execute_tool

MAX_AGENT_TURNS = 8

SYSTEM_PROMPT = """You are a music listening behaviour analyst embedded inside a production ML system.
A classical XGBoost model has already processed the user's raw data and produced calibrated signals.
Your job: use the available tools to investigate those signals, then synthesise a diagnosis.

Available tools:
  • analyze_genre_entropy   — call this to understand genre diversity collapse
  • analyze_mood_trajectory — call this to understand the user's current emotional state from audio features
  • evaluate_discovery_health — call this to assess new-artist discovery and repeat-play behaviour

Rules:
  - You MUST call all three tools before writing the final diagnosis
  - Reason explicitly across the tool outputs — don't just repeat them
  - Reference specific values (e.g. "entropy drop of 0.38", "withdrawn mood quadrant")
  - The final JSON must be valid. No markdown outside the JSON block."""

INITIAL_PROMPT_TEMPLATE = """
=== XGBoost Model Output ===
Churn Probability : {prob:.1%}
Risk Level        : {risk_level}
Base value        : {base:.3f}

Top SHAP drivers:
{drivers_text}

=== User Context ===
Top genres  : {genres}
Top artists : {artists}
Country     : {country}

Use the three tools to investigate these signals, then output ONLY a valid JSON object:
{{
  "diagnosis"      : "<2-3 sentences — what the listening pattern shows>",
  "hypothesis"     : "<1-2 sentences — likely underlying cause (life event / mood)>",
  "strategy"       : "<1-2 sentences — concrete re-engagement plan with specific artist/genre>",
  "strategy_genre" : "<single genre for Spotify search, e.g. 'jazz'>",
  "strategy_artist": "<single artist name exactly as on Spotify>",
  "urgency"        : "<'monitor' | 'act_soon' | 'act_now'>"
}}"""

# ── Tool declarations (sent to Gemini so it knows what's available) ────────────

_TOOL_DECLARATIONS = [
    {
        "name": "analyze_genre_entropy",
        "description": (
            "Analyze the user's genre diversity trend to detect musical tunnel vision. "
            "Returns entropy drop severity, dominant genre, concentration, and recommendation hint."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "focus_period": {
                    "type": "string",
                    "description": "Time window to analyse: 'recent_week' (default) or 'month'",
                    "enum": ["recent_week", "month"],
                }
            },
            "required": ["focus_period"],
        },
    },
    {
        "name": "analyze_mood_trajectory",
        "description": (
            "Analyze valence and energy trends from audio features to map the user's current "
            "emotional state. Returns mood quadrant, listen depth assessment, and schedule shift."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "depth": {
                    "type": "string",
                    "description": "'full' for complete analysis (default) or 'quick' for summary only",
                    "enum": ["full", "quick"],
                }
            },
            "required": ["depth"],
        },
    },
    {
        "name": "evaluate_discovery_health",
        "description": (
            "Evaluate new-artist discovery rate, repeat-play ratio, and session frequency trend. "
            "Returns a composite discovery health score and skip-rate interpretation."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def _build_gemini_tools() -> list:
    return [types.Tool(function_declarations=_TOOL_DECLARATIONS)]


def _format_drivers(top_drivers: list, feature_values: dict) -> str:
    lines = []
    for feat, sv in top_drivers:
        label = FEATURE_LABELS.get(feat, feat)
        val = feature_values.get(feat, 0.0)
        arrow = "↑ increases churn risk" if sv > 0 else "↓ reduces churn risk"
        lines.append(f"  • {label}: {val:.3f}  ({arrow}, SHAP={sv:+.3f})")
    return "\n".join(lines)


def _extract_function_calls(response) -> list:
    calls = []
    try:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call:
                calls.append(part.function_call)
    except Exception:
        pass
    return calls


def _extract_text(response) -> str:
    try:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                return part.text.strip()
    except Exception:
        pass
    return ""


def _call_gemini(model: str, contents: list, tools: list, client: genai.Client):
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=tools,
            temperature=0.35,
            max_output_tokens=2048,
        ),
    )


def generate_diagnosis(model_result: dict, user_profile: dict) -> tuple[dict, list]:
    """
    Run the multi-step tool-calling agent.

    Returns:
      diagnosis   – dict with keys: diagnosis, hypothesis, strategy, strategy_genre,
                    strategy_artist, urgency
      chain       – list of {"tool": name, "args": ..., "result": ...} for UI display
    """
    features = model_result["feature_values"]
    top_genres = user_profile.get("top_genres", [])[:5]
    top_artists = [a["name"] for a in user_profile.get("top_artists", [])][:5]

    initial_text = INITIAL_PROMPT_TEMPLATE.format(
        prob=model_result["churn_probability"],
        risk_level=model_result["risk_level"],
        base=model_result.get("base_value", 0.0),
        drivers_text=_format_drivers(model_result["top_drivers"], features),
        genres=", ".join(top_genres) if top_genres else "unknown",
        artists=", ".join(top_artists) if top_artists else "unknown",
        country=user_profile.get("country", "unknown"),
    )

    client = genai.Client(api_key=GEMINI_API_KEY)
    gemini_tools = _build_gemini_tools()
    chain: list[dict] = []

    contents = [
        types.Content(role="user", parts=[types.Part(text=initial_text)])
    ]

    last_err = None
    for model_name in [GEMINI_MODEL] + GEMINI_FALLBACK_MODELS:
        try:
            for _ in range(MAX_AGENT_TURNS):
                response = _call_gemini(model_name, contents, gemini_tools, client)

                fn_calls = _extract_function_calls(response)

                if not fn_calls:
                    # Agent has finished — extract final JSON
                    raw = _extract_text(response)
                    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
                    result = json.loads(raw)
                    for key in ("diagnosis", "hypothesis", "strategy", "strategy_genre", "strategy_artist"):
                        if key not in result:
                            result[key] = ""
                    if "urgency" not in result:
                        result["urgency"] = _default_urgency(model_result["churn_probability"])
                    return result, chain

                # Add model turn to history
                contents.append(response.candidates[0].content)

                # Execute each tool and add results
                fn_response_parts = []
                for fc in fn_calls:
                    args = dict(fc.args) if fc.args else {}
                    tool_result = execute_tool(fc.name, args, features, user_profile)
                    chain.append({"tool": fc.name, "args": args, "result": tool_result})

                    fn_response_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response=tool_result,
                            )
                        )
                    )

                contents.append(
                    types.Content(role="user", parts=fn_response_parts)
                )

            # Fell through max turns — try to get whatever text the model produced
            raw = _extract_text(response)
            if raw:
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
                return json.loads(raw), chain

        except Exception as e:
            last_err = e
            contents = [  # reset for next model attempt
                types.Content(role="user", parts=[types.Part(text=initial_text)])
            ]
            chain = []
            continue

    # All models failed — return rule-based fallback
    return _fallback(model_result, user_profile, last_err), chain


def _default_urgency(prob: float) -> str:
    return "act_now" if prob > 0.65 else "act_soon" if prob > 0.38 else "monitor"


def _fallback(model_result: dict, user_profile: dict, err: Exception) -> dict:
    prob = model_result["churn_probability"]
    genres = user_profile.get("top_genres", [])
    artists = [a["name"] for a in user_profile.get("top_artists", [])]
    top_genre = genres[0] if genres else "familiar genre"
    top_artist = artists[0] if artists else "a favourite artist"

    return {
        "diagnosis": (
            f"Engagement signals show a {prob:.0%} churn risk. "
            f"Top drivers: {', '.join(f[0] for f in model_result['top_drivers'][:2])}."
        ),
        "hypothesis": (
            "This pattern often coincides with periods of high stress, routine changes, "
            "or simply needing a musical refresh after overexposure to the same catalogue."
        ),
        "strategy": (
            f"Re-engage via {top_genre} through {top_artist} — a familiar anchor "
            "that connects to the user's core musical identity."
        ),
        "strategy_genre": top_genre,
        "strategy_artist": top_artist,
        "urgency": _default_urgency(prob),
        "_fallback": True,
        "_error": str(err) if err else "unknown",
    }
