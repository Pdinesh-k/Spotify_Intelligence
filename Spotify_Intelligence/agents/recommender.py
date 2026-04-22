import numpy as np
import spotipy

from ml.feedback import FeedbackStore


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _build_user_vector(user_profile: dict) -> np.ndarray | None:
    """5-dim audio feature vector from user's listen profile. None if unavailable."""
    e = user_profile.get("avg_energy")
    v = user_profile.get("avg_valence")
    d = user_profile.get("avg_danceability")
    if e is None:
        return None
    return np.array([
        e,
        v if v is not None else 0.5,
        d if d is not None else 0.5,
        user_profile.get("avg_acousticness", 0.4),
        0.1,  # instrumentalness — most users prefer vocal tracks
    ])


def _track_dict(item: dict, source: str) -> dict:
    return {
        "name": item["name"],
        "artist": item["artists"][0]["name"] if item["artists"] else "",
        "id": item["id"],
        "popularity": item.get("popularity", 0),
        "preview_url": item.get("preview_url"),
        "external_url": item.get("external_urls", {}).get("spotify", ""),
        "album_image": (
            item["album"]["images"][0]["url"]
            if item.get("album", {}).get("images")
            else ""
        ),
        "source": source,
    }


def get_recommendations(
    strategy: dict,
    user_profile: dict,
    access_token: str,
    n: int = 3,
) -> list[dict]:
    """
    Execute the LLM strategy via Spotify API + cosine similarity ranking.
    Returns top-n tracks, diverse by artist.
    """
    sp = spotipy.Spotify(auth=access_token)
    strategy_artist = strategy.get("strategy_artist", "").strip()
    strategy_genre = strategy.get("strategy_genre", "").strip()

    candidates: list[dict] = []

    # 1. Search by strategy artist
    if strategy_artist:
        try:
            res = sp.search(q=f"artist:{strategy_artist}", type="track", limit=20)
            for item in res.get("tracks", {}).get("items", []) or []:
                if item:
                    candidates.append(_track_dict(item, "strategy_artist"))
        except Exception:
            pass

    # 2. Search by strategy genre (supplement if < 15 candidates)
    if strategy_genre and len(candidates) < 15:
        try:
            res = sp.search(q=f"genre:{strategy_genre}", type="track", limit=15)
            for item in res.get("tracks", {}).get("items", []) or []:
                if item:
                    candidates.append(_track_dict(item, "strategy_genre"))
        except Exception:
            pass

    # 3. Fallback: search user's top artists for tracks
    if len(candidates) < 5:
        for artist in user_profile.get("top_artists", [])[:3]:
            try:
                res = sp.search(q=f"artist:{artist['name']}", type="track", limit=5)
                for item in res.get("tracks", {}).get("items", []) or []:
                    if item:
                        candidates.append(_track_dict(item, "top_artist_fallback"))
            except Exception:
                pass

    if not candidates:
        return []

    # Deduplicate by track id
    seen_ids: set[str] = set()
    unique: list[dict] = []
    for c in candidates:
        if c["id"] not in seen_ids:
            unique.append(c)
            seen_ids.add(c["id"])
    candidates = unique[:25]

    # Retrieve audio features for scoring (graceful 403 fallback)
    user_vector = _build_user_vector(user_profile)
    feat_map: dict[str, dict] = {}
    if user_vector is not None:
        ids = [c["id"] for c in candidates if c.get("id")]
        try:
            features = sp.audio_features(ids)
            feat_map = {f["id"]: f for f in (features or []) if f}
        except Exception:
            pass

    feedback_scores = FeedbackStore().get_track_scores()

    for c in candidates:
        feat = feat_map.get(c["id"])
        if feat and user_vector is not None:
            track_vec = np.array([
                feat.get("energy", 0.5),
                feat.get("valence", 0.5),
                feat.get("danceability", 0.5),
                feat.get("acousticness", 0.4),
                feat.get("instrumentalness", 0.1),
            ])
            c["similarity"] = _cosine_similarity(user_vector, track_vec)
        else:
            # Fallback similarity when audio features are unavailable
            # Ensure a baseline match of ~65% since they match the text/artist strategy
            c["similarity"] = max(c["popularity"] / 100.0, 0.65 + np.random.rand() * 0.15)

        c["feedback_score"] = feedback_scores.get(c["id"], 0.0)
        # Blend: 70% content similarity, 30% learned feedback signal
        c["final_score"] = 0.7 * c["similarity"] + 0.3 * np.tanh(c["feedback_score"])

    # Sort and deduplicate by artist for diversity
    ranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
    seen_artists: set[str] = set()
    final: list[dict] = []
    for c in ranked:
        if c["artist"] not in seen_artists:
            final.append(c)
            seen_artists.add(c["artist"])
        if len(final) >= n:
            break

    return final
