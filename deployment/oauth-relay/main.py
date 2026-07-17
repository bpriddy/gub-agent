"""
OAuth relay — preserves query params that Discovery Engine strips from Google endpoints.

Agentspace calls this URL instead of accounts.google.com directly.
This function 302-redirects to Google's OAuth endpoint with the full query string intact.
"""

import functions_framework
from flask import redirect, request

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"


@functions_framework.http
def oauth_relay(request):
    qs = request.query_string.decode()
    target = f"{GOOGLE_AUTH_URL}?{qs}"
    # Log param NAMES only — the query string carries OAuth authorize params
    # (client_id, redirect_uri, state) that don't belong in function logs.
    print(f"RELAY params: {sorted(request.args.keys())}")
    return redirect(target, code=302)
