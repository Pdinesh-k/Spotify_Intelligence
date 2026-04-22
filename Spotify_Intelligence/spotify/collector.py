import spotipy
from typing import Optional


def collect_user_data(access_token: str) -> dict:
    sp = spotipy.Spotify(auth=access_token)
    result: dict = {}

    me = sp.current_user()
    result["display_name"] = me.get("display_name") or me.get("id", "User")
    result["user_id"] = me.get("id", "")
    result["followers"] = me.get("followers", {}).get("total", 0)
    result["country"] = me.get("country", "")

    top_short = sp.current_user_top_tracks(limit=20, time_range="short_term")
    top_med = sp.current_user_top_tracks(limit=20, time_range="medium_term")

    def _track(t: dict) -> dict:
        return {
            "name": t["name"],
            "artist": t["artists"][0]["name"] if t["artists"] else "",
            "id": t["id"],
            "popularity": t.get("popularity", 0),
            "explicit": t.get("explicit", False),
            "album_image": (
                t["album"]["images"][0]["url"] if t.get("album", {}).get("images") else ""
            ),
            "external_url": t.get("external_urls", {}).get("spotify", ""),
        }

    result["top_tracks_recent"] = [_track(t) for t in top_short.get("items", [])]
    result["top_tracks_alltime"] = [_track(t) for t in top_med.get("items", [])]

    top_artists = sp.current_user_top_artists(limit=15, time_range="medium_term")
    result["top_artists"] = [
        {
            "name": a["name"],
            "id": a["id"],
            "genres": a.get("genres", []),
            "popularity": a.get("popularity", 0),
        }
        for a in top_artists.get("items", [])
    ]

    all_genres: list[str] = []
    for a in top_artists.get("items", []):
        all_genres.extend(a.get("genres", []))
    genre_counts: dict[str, int] = {}
    for g in all_genres:
        genre_counts[g] = genre_counts.get(g, 0) + 1
    result["top_genres"] = sorted(genre_counts, key=genre_counts.get, reverse=True)[:10]
    result["genre_counts"] = genre_counts

    recent = sp.current_user_recently_played(limit=50)
    result["recently_played"] = [
        {
            "name": item["track"]["name"],
            "artist": item["track"]["artists"][0]["name"] if item["track"]["artists"] else "",
            "id": item["track"]["id"],
            "played_at": item["played_at"],
            "ms_played": None,
        }
        for item in recent.get("items", [])
    ]

    track_ids = [t["id"] for t in result["top_tracks_recent"] if t.get("id")][:20]
    if track_ids:
        try:
            features = sp.audio_features(track_ids)
            valid = [f for f in (features or []) if f]
            if valid:
                result["avg_energy"] = round(sum(f["energy"] for f in valid) / len(valid), 3)
                result["avg_valence"] = round(sum(f["valence"] for f in valid) / len(valid), 3)
                result["avg_danceability"] = round(
                    sum(f["danceability"] for f in valid) / len(valid), 3
                )
                result["avg_tempo"] = round(sum(f["tempo"] for f in valid) / len(valid))
                result["avg_acousticness"] = round(
                    sum(f["acousticness"] for f in valid) / len(valid), 3
                )
                result["audio_features_available"] = True
        except Exception:
            result["audio_features_available"] = False

    playlists = sp.current_user_playlists(limit=50)
    result["playlists"] = [
        {
            "name": p["name"],
            "track_count": p.get("tracks", {}).get("total", 0),
            "public": p.get("public", False),
        }
        for p in playlists.get("items", [])
        if p
    ]

    return result
