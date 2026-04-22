import spotipy
from spotipy.oauth2 import SpotifyOAuth
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI

SCOPES = " ".join([
    "user-top-read",
    "user-read-recently-played",
    "playlist-read-private",
    "user-read-private",
])


def _make_oauth(**kwargs) -> SpotifyOAuth:
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPES,
        cache_handler=spotipy.cache_handler.MemoryCacheHandler(),
        open_browser=False,
        **kwargs,
    )


def get_oauth_url(state: str = "intelligence") -> str:
    return _make_oauth(state=state, show_dialog=True).get_authorize_url()


def exchange_code(code: str) -> dict:
    return _make_oauth().get_access_token(code, as_dict=True)
