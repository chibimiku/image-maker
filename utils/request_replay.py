"""Credential-free request snapshots; transport inputs are never mutated."""
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

REDACTED = "<REDACTED_CREDENTIAL>"


def sensitive_name(name):
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return key in {"authorization", "proxyauthorization", "cookie", "setcookie",
                   "apikey", "xapikey", "xgoogapikey", "token", "accesstoken",
                   "refreshtoken", "xauthtoken", "password", "secret", "clientsecret", "key"}


def redact_body(value):
    if isinstance(value, dict):
        return {key: REDACTED if sensitive_name(key) else redact_body(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact_body(item) for item in value]
    return value


def safe_request_snapshot(url, headers, body):
    safe_headers, credentials = {}, {}
    for key, value in (headers or {}).items():
        if sensitive_name(key):
            safe_headers[key] = REDACTED
            prefix = "Bearer " if str(value).lower().startswith("bearer ") else ""
            env = "IMAGE_MAKER_REPLAY_" + re.sub(r"[^A-Z0-9]", "_", key.upper())
            credentials[key] = {"env": env, "prefix": prefix}
        else:
            safe_headers[key] = value
    parts = urlsplit(str(url or ""))
    query = parse_qsl(parts.query, keep_blank_values=True)
    url_secret = bool(parts.username or parts.password or any(sensitive_name(k) for k, _ in query))
    netloc = parts.netloc.rsplit("@", 1)[-1]
    safe_url = urlunsplit((parts.scheme, netloc, parts.path,
                          urlencode([(k, REDACTED if sensitive_name(k) else v) for k, v in query]),
                          parts.fragment)) if url_secret else str(url or "")
    safe_body = redact_body(body)
    return {"url": safe_url, "headers": safe_headers, "body": safe_body,
            "credential_headers": credentials,
            "url_env": "IMAGE_MAKER_REPLAY_URL" if url_secret else None,
            "body_env": "IMAGE_MAKER_REPLAY_BODY_JSON" if safe_body != body else None}
