# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Admin login: a password, a signed session cookie, and a brake on guessing.

Annotators keep their personal link (/a/<token>): low stakes, and a login would
only get in their way. The admin panel can delete people and answers, so it
sits behind a password instead of a token in the URL (which ends up in browser
history, screenshots and server logs).

  * The password is stored as PBKDF2-SHA256 with a random salt, never in clear.
  * A fresh database starts with DEFAULT_PASSWORD. The code is public, so that
    default is public too: until it is changed, every admin action except
    changing it is refused.
  * A login sets an HttpOnly, SameSite=Strict cookie "<expiry>.<hmac>" signed
    with a per-database secret. Changing the password rotates the secret, which
    logs out every other session.
  * Wrong passwords are slowed down, and an address is locked out for a while
    after too many of them.
  * Plain HTTP carries the password in clear: outside localhost or an SSH
    tunnel, put the server behind HTTPS.
"""
import hashlib
import hmac
import secrets
import threading
import time

from . import db

DEFAULT_PASSWORD = "dialfork-coling-2027"
COOKIE = "fb_admin"
SESSION_SECONDS = 12 * 3600
MIN_LENGTH = 8
_ITER = 200000

_fails = {}          # address -> [times of recent failed logins]
_lock = threading.Lock()
MAX_FAILS, WINDOW = 8, 600


def _hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode(), _ITER)
    return "pbkdf2_sha256$%d$%s$%s" % (_ITER, salt, dk.hex())


def _check(password, stored):
    try:
        _, it, salt, want = stored.split("$")
    except (AttributeError, ValueError):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode(), int(it))
    return hmac.compare_digest(dk.hex(), want)


def ensure(con):
    """First run: install the default password and a session secret."""
    if db.meta_get(con, "admin_pw") is None:
        db.meta_set(con, "admin_pw", _hash(DEFAULT_PASSWORD))
        db.meta_set(con, "admin_pw_default", "1")
    if db.meta_get(con, "session_secret") is None:
        db.meta_set(con, "session_secret", secrets.token_hex(32))


def using_default(con):
    return db.meta_get(con, "admin_pw_default") == "1"


def set_password(con, new):
    """Store a new password; rotating the secret ends every existing session."""
    if len(new or "") < MIN_LENGTH:
        raise ValueError("the password needs at least %d characters" % MIN_LENGTH)
    if new == DEFAULT_PASSWORD:
        raise ValueError("choose something other than the default password")
    db.meta_set(con, "admin_pw", _hash(new))
    db.meta_set(con, "admin_pw_default", "0")
    db.meta_set(con, "session_secret", secrets.token_hex(32))


def locked_out(addr):
    now = time.time()
    with _lock:
        recent = [t for t in _fails.get(addr, []) if now - t < WINDOW]
        _fails[addr] = recent
        return len(recent) >= MAX_FAILS


def login(con, addr, password):
    """-> a cookie value, or None. Failures are slowed down and counted."""
    if _check(password or "", db.meta_get(con, "admin_pw")):
        with _lock:
            _fails.pop(addr, None)
        return new_session(con)
    with _lock:
        _fails.setdefault(addr, []).append(time.time())
    time.sleep(0.8)
    return None


def new_session(con):
    exp = str(int(time.time()) + SESSION_SECONDS)
    return exp + "." + _sign(con, exp)


def _sign(con, exp):
    secret = (db.meta_get(con, "session_secret") or "").encode()
    return hmac.new(secret, exp.encode(), hashlib.sha256).hexdigest()


def valid(con, cookie_value):
    try:
        exp, sig = (cookie_value or "").split(".", 1)
        if int(exp) < time.time():
            return False
    except ValueError:
        return False
    return hmac.compare_digest(sig, _sign(con, exp))


def cookie_header(value, max_age=SESSION_SECONDS):
    return ("%s=%s; Path=/; HttpOnly; SameSite=Strict; Max-Age=%d"
            % (COOKIE, value, max_age))
