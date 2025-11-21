"""
CogNeo AI — Secure Account Access
- Email/password signup and login (no third-party providers)
- Minimalist SaaS-style UI, no backend details exposed
"""

import re
import os
import streamlit as st
from sqlalchemy.exc import IntegrityError
from db.store import (
    create_user, get_user_by_email, check_password, set_last_login
)

# Resolve path to repository logo
LOGO_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "static", "logo.jpg"))

# Page config and minimal chrome
st.set_page_config(page_title="CogNeo AI • Login", layout="centered")
st.markdown("""
<style>
/* Hide sidebar for a focused login experience */
[data-testid="stSidebar"], .css-1lcbmhc, .css-164nlkn { display: none !important; }

/* Full-page gradient background */
body { background: linear-gradient(165deg, #f7faff 0%, #fafcfe 33%, #ffffff 100%) !important; }

/* Centered auth card */
.auth-ct {
  max-width: 440px;
  margin: 3vh auto 6vh auto;
  padding: 24px 24px 20px 24px;
  border-radius: 12px;
  border: 1px solid #e9eef6;
  background: #ffffff;
  box-shadow: 0 8px 22px -14px rgba(39,78,120,0.22);
}

/* Branding header */
.brand-title {
  text-align:center;
  font-weight: 800;
  font-size: 2.0em;
  letter-spacing: .3px;
  color: #1a3a69;
  margin-bottom: 4px;
}
/* Centered brand logo image above title */
.brand-logo-img {
  display: block;
  height: 58px;
  width: auto;
  margin: 0 auto 8px auto;
  border-radius: 12px;
  box-shadow: 0 6px 16px rgba(216,27,96,0.30);
}
/* Gradient brand text next to logo */
.brand-gradient {
  font-weight: 800;
  font-size: 2.4em;
  background: linear-gradient(135deg, #ff3b8e, #d81b60);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  line-height: 1;
}

/* Subheader */
.subtle { color:#6a7d98; font-size:.96em; margin-bottom: 12px; }

/* Inputs/buttons tweaks */
input, textarea { border-radius: 9px !important; }
.stButton>button {
  width: 100%;
  border-radius: 10px;
  background: linear-gradient(135deg, #ff3b8e 0%, #d81b60 100%);
  border: 1px solid #c2185b;
  color: #fff;
  font-weight: 600;
  letter-spacing: .3px;
}
.stButton>button:disabled {
  opacity: .55; cursor: not-allowed;
}

/* Switch links */
.switch-row {
  display:flex; gap:8px; justify-content:center; align-items:center; margin-top:10px;
  color:#5c6f8a; font-size:.95em;
}
.switch-link { color:#2a67d8; text-decoration: underline; cursor:pointer; }

/* Error / success styling alignment */
.block-note { margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"  # or "signup"

def valid_email(s: str) -> bool:
    if not s: return False
    # Basic RFC-like pattern, sufficient for UI validation
    return re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", s) is not None

def render_logo():
    try:
        mid = st.columns([1, 2, 1])[1]
        with mid:
            c1, c2 = st.columns([0.25, 0.75])
            if os.path.exists(LOGO_PATH):
                c1.image(LOGO_PATH, width=80)
            else:
                c1.markdown(
                    "<div style='height:80px;width:80px;border-radius:14px;"
                    "background:linear-gradient(135deg,#ff3b8e,#d81b60);"
                    "box-shadow:0 6px 16px rgba(216,27,96,0.30);'></div>",
                    unsafe_allow_html=True
                )
            c2.markdown("<div style='display:flex;align-items:center;height:80px;'><div class='brand-gradient'>CogNeo AI</div></div>", unsafe_allow_html=True)
    except Exception:
        st.markdown(
            "<div style='display:flex;justify-content:center;align-items:center;gap:12px;margin-bottom:6px;'>"
            "<div style='height:80px;width:80px;border-radius:14px;background:linear-gradient(135deg,#ff3b8e,#d81b60);"
            "box-shadow:0 6px 16px rgba(216,27,96,0.30);'></div>"
            "<div class='brand-gradient'>CogNeo AI</div>"
            "</div>",
            unsafe_allow_html=True
        )

def do_signup():
    render_logo()
    st.markdown("<div class='subtle'>Create a new account</div>", unsafe_allow_html=True)

    email = st.text_input("Email address", key="signup_email", placeholder="you@company.com")
    password = st.text_input("Password", type="password", key="signup_pw", placeholder="At least 8 characters")
    confirm = st.text_input("Confirm password", type="password", key="signup_pw2", placeholder="Re-enter password")
    name = st.text_input("Full name (optional)", key="signup_name", placeholder="Jane Doe")

    # Client-side checks
    email_ok = valid_email(email)
    pw_ok = bool(password) and len(password) >= 8
    match_ok = password == confirm and bool(confirm)
    create_disabled = not (email_ok and pw_ok and match_ok)

    if not email_ok and email:
        st.error("Please enter a valid email address.", icon=":material/warning:")
    if password and len(password) < 8:
        st.error("Password must be at least 8 characters.", icon=":material/warning:")
    if confirm and not match_ok:
        st.error("Passwords do not match.", icon=":material/warning:")

    sub = st.button("Create Account", disabled=create_disabled)
    if sub:
        try:
            # Server-side checks (no backend details in messages)
            if get_user_by_email(email):
                st.error("This email is already registered.", icon=":material/error:")
            else:
                create_user(email=email, password=password, name=name or None)
                st.success("Account created. You can sign in now.", icon=":material/check_circle:")
                st.session_state["auth_mode"] = "login"
                st.rerun()
        except IntegrityError:
            # Race condition: email just registered
            st.error("This email is already registered.", icon=":material/error:")
        except Exception:
            st.error("Unable to create account at this time. Please try again.", icon=":material/error:")

    # Replace hyperlink with a working button to avoid broken JS link behavior
    st.markdown("<div class='switch-row'>Already have an account?</div>", unsafe_allow_html=True)
    if st.button("Sign in"):
        st.session_state["auth_mode"] = "login"
        st.rerun()


def do_login():
    render_logo()
    st.markdown("<div class='subtle'>Sign in to continue</div>", unsafe_allow_html=True)

    email = st.text_input("Email address", key="login_email", placeholder="you@company.com")
    password = st.text_input("Password", type="password", key="login_pw", placeholder="Your password")

    email_ok = valid_email(email)
    login_disabled = not (email_ok and bool(password))

    if email and not email_ok:
        st.error("Please enter a valid email address.", icon=":material/warning:")

    login = st.button("Sign in", disabled=login_disabled)
    if login:
        try:
            user = get_user_by_email(email)
            # Generic message to avoid leaking whether email exists
            if not user or not user.password_hash or not check_password(password, user.password_hash):
                st.error("Invalid email or password.", icon=":material/error:")
            else:
                set_last_login(user.id)
                st.session_state["user"] = {
                    "id": user.id, "email": user.email, "name": user.name
                }
                st.success("Welcome back.", icon=":material/check_circle:")
                # Redirect after successful login
                if hasattr(st, "switch_page"):
                    st.switch_page("pages/chat.py")
                else:
                    st.rerun()
        except Exception:
            st.error("Unable to sign in right now. Please try again.", icon=":material/error:")

    # Minimalist signup CTA (centered)
    # Replace hyperlink with a working button to avoid broken JS link behavior
    st.markdown("<div class='switch-row'>New to CogNeo AI?</div>", unsafe_allow_html=True)
    if st.button("Create a new account"):
        st.session_state["auth_mode"] = "signup"
        st.rerun()


# Router
if st.session_state["auth_mode"] == "signup":
    do_signup()
else:
    do_login()
