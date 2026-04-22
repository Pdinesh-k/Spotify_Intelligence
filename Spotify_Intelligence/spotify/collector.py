import spotipy


def _track(t: dict) -> dict:
    return {
        "name": t["name"],
        "artist": t["artists"][0]["name"] if t["artists"] else "",
        "artist_id": t["artists"][0]["id"] if t["artists"] else "",
        "id": t["id"],
        "popularity": t.get("popularity", 0),
        "explicit": t.get("explicit", False),
        "duration_ms": t.get("duration_ms", 210_000),
        "album_image": (
            t["album"]["images"][0]["url"] if t.get("album", {}).get("images") else ""
        ),
        "external_url": t.get("external_urls", {}).get("spotify", ""),
    }


def _artist(a: dict) -> dict:
    return {
        "name": a["name"],
        "id": a["id"],
        "genres": a.get("genres", []),
        "popularity": a.get("popularity", 0),
        "followers": a.get("followers", {}).get("total", 0),
        "image": a["images"][0]["url"] if a.get("images") else "",
        "external_url": a.get("external_urls", {}).get("spotify", ""),
    }


def collect_user_data(access_token: str) -> dict:
    sp = spotipy.Spotify(auth=access_token)
    result: dict = {}

    # ── Profile ───────────────────────────────────────────────────────────────
    me = sp.current_user()
    result["display_name"] = me.get("display_name") or me.get("id", "User")
    result["user_id"] = me.get("id", "")
    result["followers"] = me.get("followers", {}).get("total", 0)
    result["country"] = me.get("country", "")
    result["product"] = me.get("product", "free")  # free / premium

    # ── Top Tracks — all 3 time ranges ────────────────────────────────────────
    try:
        short = sp.current_user_top_tracks(limit=50, time_range="short_term")
        result["top_tracks_short"] = [_track(t) for t in short.get("items", [])]
    except Exception:
        result["top_tracks_short"] = []

    try:
        medium = sp.current_user_top_tracks(limit=50, time_range="medium_term")
        result["top_tracks_medium"] = [_track(t) for t in medium.get("items", [])]
    except Exception:
        result["top_tracks_medium"] = []

    try:
        long_ = sp.current_user_top_tracks(limit=50, time_range="long_term")
        result["top_tracks_long"] = [_track(t) for t in long_.get("items", [])]
    except Exception:
        result["top_tracks_long"] = []

    # Legacy aliases expected by downstream code
    result["top_tracks_recent"] = result["top_tracks_short"]
    result["top_tracks_alltime"] = result["top_tracks_long"]

    # ── Top Artists — all 3 time ranges ───────────────────────────────────────
    try:
        artists_short = sp.current_user_top_artists(limit=50, time_range="short_term")
        result["top_artists_short"] = [_artist(a) for a in artists_short.get("items", [])]
    except Exception:
        result["top_artists_short"] = []

    try:
        artists_med = sp.current_user_top_artists(limit=50, time_range="medium_term")
        result["top_artists_medium"] = [_artist(a) for a in artists_med.get("items", [])]
    except Exception:
        result["top_artists_medium"] = []

    try:
        artists_long = sp.current_user_top_artists(limit=50, time_range="long_term")
        result["top_artists_long"] = [_artist(a) for a in artists_long.get("items", [])]
    except Exception:
        result["top_artists_long"] = []

    # Legacy alias
    result["top_artists"] = result["top_artists_medium"]

    # ── Genre aggregation across all artists ─────────────────────────────────
    all_genres: list[str] = []
    for a in result["top_artists_medium"]:
        all_genres.extend(a.get("genres", []))
    genre_counts: dict[str, int] = {}
    for g in all_genres:
        genre_counts[g] = genre_counts.get(g, 0) + 1
    result["top_genres"] = sorted(genre_counts, key=genre_counts.get, reverse=True)[:10]
    result["genre_counts"] = genre_counts

    # ── Recently Played (last 50, with full timestamps) ───────────────────────
    try:
        recent = sp.current_user_recently_played(limit=50)
        result["recently_played"] = [
            {
                "name": item["track"]["name"],
                "artist": item["track"]["artists"][0]["name"] if item["track"]["artists"] else "",
                "id": item["track"]["id"],
                "played_at": item["played_at"],
                "ms_played": item["track"].get("duration_ms", 210_000),
            }
            for item in recent.get("items", [])
        ]
    except Exception:
        result["recently_played"] = []

    # ── Audio Features (graceful 403 fallback) ────────────────────────────────
    track_ids = [t["id"] for t in result["top_tracks_short"] if t.get("id")][:50]
    result["track_audio_features"] = {}
    if track_ids:
        try:
            features = sp.audio_features(track_ids)
            valid = [f for f in (features or []) if f]
            result["track_audio_features"] = {f["id"]: f for f in valid}
            if valid:
                result["avg_energy"] = round(sum(f["energy"] for f in valid) / len(valid), 3)
                result["avg_valence"] = round(sum(f["valence"] for f in valid) / len(valid), 3)
                result["avg_danceability"] = round(sum(f["danceability"] for f in valid) / len(valid), 3)
                result["avg_tempo"] = round(sum(f["tempo"] for f in valid) / len(valid))
                result["avg_acousticness"] = round(sum(f["acousticness"] for f in valid) / len(valid), 3)
                result["avg_speechiness"] = round(sum(f["speechiness"] for f in valid) / len(valid), 3)
                result["avg_liveness"] = round(sum(f["liveness"] for f in valid) / len(valid), 3)
                result["audio_features_available"] = True
        except Exception:
            result["audio_features_available"] = False
    else:
        result["audio_features_available"] = False

    # ── Related Artists for top 3 artists ─────────────────────────────────────
    result["related_artists"] = []
    for artist in result["top_artists_short"][:3]:
        try:
            related = sp.artist_related_artists(artist["id"])
            for ra in related.get("artists", [])[:5]:
                result["related_artists"].append({
                    "name": ra["name"],
                    "id": ra["id"],
                    "genres": ra.get("genres", []),
                    "popularity": ra.get("popularity", 0),
                    "anchor_artist": artist["name"],
                })
        except Exception:
            pass

    # ── Saved Tracks count ────────────────────────────────────────────────────
    try:
        saved = sp.current_user_saved_tracks(limit=1)
        result["saved_tracks_total"] = saved.get("total", 0)
    except Exception:
        result["saved_tracks_total"] = 0

    # ── Playlists ─────────────────────────────────────────────────────────────
    try:
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
    except Exception:
        result["playlists"] = []

    return result
