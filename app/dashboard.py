
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st


API_URL = os.getenv("FOOD_API_URL", "http://127.0.0.1:8000").rstrip("/")
BASE_DIR = Path(__file__).resolve().parent.parent
USER_STATE_FILE = BASE_DIR / "data" / "demo_user_state.json"
USER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Food Delivery",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1200px; padding-top: 1.4rem; }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.18); }
    .brand { font-size: 29px; font-weight: 800; }
    .hero { padding: 30px; border-radius: 22px; border: 1px solid rgba(128,128,128,.18); margin-bottom: 22px; background: linear-gradient(135deg,#fff7ed,#ffffff); }
    .hero h1 { margin: 0 0 8px 0; font-size: 38px; }
    .name { font-size: 20px; font-weight: 750; }
    .meta { color: #777; font-size: 13px; margin-bottom: 8px; }
    .note { padding: 13px 15px; border-radius: 12px; background: #f6f8fa; color: #555; }
    .login-box { padding: 28px; border-radius: 20px; border: 1px solid rgba(128,128,128,.18); max-width: 760px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# User state
# ============================================================

def current_user_key():
    user = st.session_state.get("current_user")
    return user["user_id"] if user else "guest"


def read_user_state():
    try:
        if USER_STATE_FILE.exists():
            data = json.loads(USER_STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def save_user_state():
    try:
        payload = read_user_state()
        key = current_user_key()
        payload[key] = {
            "cart": st.session_state.get("cart", []),
            "orders": st.session_state.get("orders", []),
        }
        USER_STATE_FILE.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        st.session_state._loaded_user_key = key
    except OSError:
        pass


def load_user_state(force=False):
    user_key = current_user_key()

    if (
        not force
        and st.session_state.get("_loaded_user_key") == user_key
        and "cart" in st.session_state
        and "orders" in st.session_state
    ):
        return

    payload = read_user_state()
    data = payload.get(user_key, {})

    st.session_state.cart = list(data.get("cart", []) or [])
    st.session_state.orders = list(data.get("orders", []) or [])
    st.session_state._loaded_user_key = user_key


if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "location" not in st.session_state:
    st.session_state.location = {
        "name": "Ranchi",
        "lat": 23.3538,
        "lon": 85.3270,
    }
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "selected" not in st.session_state:
    st.session_state.selected = None
if "cart_notice" not in st.session_state:
    st.session_state.cart_notice = None

# Always make sure the current session is using the correct user's cart.
load_user_state()


# ============================================================
# API helpers
# ============================================================

def api_get(path, params=None):
    try:
        response = requests.get(
            f"{API_URL}{path}",
            params=params,
            timeout=30,
        )
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            return {"error": f"HTTP {response.status_code}: {detail}"}
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "error": "FastAPI is not running. Start: python -m uvicorn app.api:app --reload"
        }
    except requests.exceptions.Timeout:
        return {"error": "Backend request timed out."}
    except Exception as exc:
        return {"error": str(exc)}


def api_post(path, payload):
    try:
        response = requests.post(
            f"{API_URL}{path}",
            json=payload,
            timeout=30,
        )
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            return {"error": f"HTTP {response.status_code}: {detail}"}
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "error": "FastAPI is not running. Start: python -m uvicorn app.api:app --reload"
        }
    except requests.exceptions.Timeout:
        return {"error": "Backend request timed out."}
    except Exception as exc:
        return {"error": str(exc)}


# ============================================================
# Utility functions
# ============================================================

def display_name(item):
    raw = str(item.get("restaurant_name", "Restaurant"))
    if raw.startswith("RestaurantProxy_"):
        return "Restaurant · " + raw.replace("RestaurantProxy_", "")
    return raw


def image_for(kind):
    kind = str(kind or "").lower()
    urls = {
        "meal": "https://images.unsplash.com/photo-1547592180-85f173990554?auto=format&fit=crop&w=900&q=80",
        "snack": "https://images.unsplash.com/photo-1626082927389-6cd097cdc6ec?auto=format&fit=crop&w=900&q=80",
        "drinks": "https://images.unsplash.com/photo-1513558161293-cdaf765ed2fd?auto=format&fit=crop&w=900&q=80",
        "buffet": "https://images.unsplash.com/photo-1547592180-85f173990554?auto=format&fit=crop&w=900&q=80",
    }
    return urls.get(kind, urls["meal"])


def add_to_cart(item):
    clean_item = {
        "restaurant_proxy_id": item.get("restaurant_proxy_id"),
        "restaurant_name": str(item.get("restaurant_name", "Restaurant")),
        "city": item.get("city"),
        "order_type": item.get("order_type"),
        "rating": item.get("rating"),
        "customer_distance_km": item.get("customer_distance_km"),
        "predicted_eta_min": item.get("predicted_eta_min"),
        "orders": item.get("orders"),
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }

    proxy_id = clean_item["restaurant_proxy_id"]
    if proxy_id:
        st.session_state.cart = [
            x for x in st.session_state.cart
            if x.get("restaurant_proxy_id") != proxy_id
        ]

    st.session_state.cart.append(clean_item)
    save_user_state()

    # Keep the selected user's cart visible immediately after adding.
    st.session_state.page = "Cart"
    st.session_state.cart_notice = (
        f"Added {display_name(item)} to your cart."
    )


def search(query, top_k=8, min_rating=4.0, max_distance=20):
    now = datetime.now()
    params = {
        "query": query,
        "top_k": top_k,
        "user_lat": st.session_state.location["lat"],
        "user_lon": st.session_state.location["lon"],
        "min_rating": min_rating,
        "max_distance_km": max_distance,
        "pickup_hour": now.hour,
        "day_of_week": now.weekday(),
        "is_peak": int(now.hour in (12, 13, 19, 20, 21)),
        "vehicle_condition": 1,
        "multiple_deliveries": 0,
        "delivery_person_age": 30,
    }
    result = api_get("/recommend", params=params)
    if "results" in result:
        st.session_state.last_results = result["results"]
    return result


def logout():
    save_user_state()
    st.session_state.current_user = None
    st.session_state.page = "Home"
    st.session_state.selected = None
    st.session_state.last_results = []
    load_user_state()
    st.rerun()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="brand">🍽️ Food Delivery</div>',
        unsafe_allow_html=True,
    )
    st.caption("Data-driven restaurant discovery")

    if st.session_state.current_user:
        user = st.session_state.current_user
        st.success(f"Signed in as {user['name']}")
        st.caption(f"User ID: {user['user_id']}")
        st.caption(f"Mobile: {user['mobile']}")
        if st.button("Logout", use_container_width=True):
            logout()
    else:
        st.info("Guest mode")
        if st.button("🔐 Login / Sign up", use_container_width=True):
            st.session_state.page = "Auth"
            st.rerun()

    st.divider()

    st.markdown(f"**🛒 Cart: {len(st.session_state.cart)} item(s)**")
    if st.button(
        f"🛒 Open cart ({len(st.session_state.cart)})",
        key="sidebar_open_cart",
        use_container_width=True,
    ):
        st.session_state.page = "Cart"
        st.rerun()

    menu_pages = [
        "Home",
        "Find Food",
        "For You",
        "Cart",
        "Orders",
        "Account",
    ]

    if st.session_state.page in menu_pages:
        selected = st.radio(
            "Menu",
            menu_pages,
            index=menu_pages.index(st.session_state.page),
        )
        if selected != st.session_state.page:
            st.session_state.page = selected
            st.rerun()

    st.divider()
    st.markdown("### 📍 Delivery location")

    loc_name = st.text_input(
        "Location",
        value=st.session_state.location["name"],
    )
    lat = st.number_input(
        "Latitude",
        value=float(st.session_state.location["lat"]),
        format="%.5f",
    )
    lon = st.number_input(
        "Longitude",
        value=float(st.session_state.location["lon"]),
        format="%.5f",
    )

    if st.button("Save location", use_container_width=True):
        st.session_state.location = {
            "name": loc_name,
            "lat": lat,
            "lon": lon,
        }
        st.success("Location saved")

    health = api_get("/health")
    if health.get("status") == "ok":
        st.success("Backend online")
    elif "error" in health:
        st.error(health["error"])


# ============================================================
# AUTH
# ============================================================

if st.session_state.page == "Auth":
    st.markdown(
        """
        <div class="login-box">
            <h1>Welcome back 👋</h1>
            <p>Login with your User ID or mobile number.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, register_tab = st.tabs(
        ["Login", "Create account"]
    )

    with login_tab:
        identifier = st.text_input(
            "User ID or mobile",
            placeholder="e.g. aman or 9876543210",
            key="auth_login_identifier",
        )
        password = st.text_input(
            "Password",
            type="password",
            key="auth_login_password",
        )

        if st.button(
            "Login",
            type="primary",
            use_container_width=True,
            key="auth_login_button",
        ):
            identifier_clean = identifier.strip()
            password_clean = password.strip()

            if not identifier_clean or not password_clean:
                st.warning("Please enter both User ID/mobile and password.")
            else:
                result = api_post(
                    "/auth/login",
                    {
                        "identifier": identifier_clean,
                        "password": password_clean,
                    },
                )

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.current_user = result["user"]
                    load_user_state()
                    st.session_state.page = "Home"
                    st.success(
                        f"Welcome back, {result['user']['name']}!"
                    )
                    st.rerun()

    with register_tab:
        name = st.text_input(
            "Full name",
            key="auth_register_name",
        )
        user_id = st.text_input(
            "User ID",
            placeholder="aman_01",
            key="auth_register_user_id",
        )
        mobile = st.text_input(
            "Mobile number",
            placeholder="9876543210",
            key="auth_register_mobile",
        )
        password1 = st.text_input(
            "Password",
            type="password",
            key="auth_register_password",
        )
        password2 = st.text_input(
            "Confirm password",
            type="password",
            key="auth_register_password_confirm",
        )

        if st.button(
            "Create account",
            type="primary",
            use_container_width=True,
            key="auth_register_button",
        ):
            name_clean = name.strip()
            user_id_clean = user_id.strip()
            mobile_clean = mobile.strip()
            password1_clean = password1.strip()
            password2_clean = password2.strip()

            if not name_clean or not user_id_clean or not mobile_clean:
                st.warning("Please fill in name, User ID, and mobile number.")
            elif not password1_clean or not password2_clean:
                st.warning("Please enter and confirm your password.")
            elif password1_clean != password2_clean:
                st.error("Passwords do not match.")
            else:
                result = api_post(
                    "/auth/register",
                    {
                        "name": name_clean,
                        "user_id": user_id_clean,
                        "mobile": mobile_clean,
                        "password": password1_clean,
                    },
                )

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.current_user = result["user"]
                    st.session_state.cart = []
                    st.session_state.orders = []
                    save_user_state()
                    st.session_state.page = "Home"
                    st.success("Account created successfully.")
                    st.rerun()

    st.divider()
    if st.button(
        "Continue as Guest",
        use_container_width=True,
    ):
        st.session_state.current_user = None
        load_user_state()
        st.session_state.page = "Home"
        st.rerun()


# ============================================================
# RESTAURANT CARD
# ============================================================

def render_card(item, key):
    with st.container(border=True):
        c1, c2 = st.columns([1, 2.7])

        with c1:
            st.image(
                image_for(item.get("order_type")),
                use_container_width=True,
            )

        with c2:
            st.markdown(
                f'<div class="name">{display_name(item)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="meta">{item.get("city", "Unknown city")} · {item.get("order_type", "Unknown")}</div>',
                unsafe_allow_html=True,
            )

            values = []
            if item.get("rating") is not None:
                values.append(f"⭐ {float(item['rating']):.2f}")
            if item.get("customer_distance_km") is not None:
                values.append(
                    f"📍 {float(item['customer_distance_km']):.2f} km"
                )
            if item.get("predicted_eta_min") is not None:
                values.append(
                    f"⏱️ {float(item['predicted_eta_min']):.0f} min"
                )
            if item.get("orders") is not None:
                values.append(
                    f"{int(item['orders']):,} historical records"
                )

            if values:
                st.write(" · ".join(values))

            view_col, cart_col = st.columns(2)

            with view_col:
                if st.button(
                    "View",
                    key=f"view_{key}",
                    use_container_width=True,
                ):
                    st.session_state.selected = item
                    st.session_state.page = "Restaurant"
                    st.rerun()

            with cart_col:
                if st.button(
                    "🛒 Add to cart",
                    key=f"add_{key}",
                    use_container_width=True,
                ):
                    add_to_cart(item)
                    st.rerun()


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>Find food that fits your day.</h1>
            <p>
                Discover restaurants from the real generated catalog,
                ranked using location, rating, popularity and predicted ETA.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Search restaurants or food context",
        placeholder="Meal, Snack, Drinks, Buffet",
    )

    if st.button(
        "Search",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.page = "Find Food"
        st.session_state.search_query = query or "Meal"
        st.rerun()

    st.subheader("Browse by order type")

    cols = st.columns(4)
    for i, category in enumerate(
        ["Meal", "Snack", "Drinks", "Buffet"]
    ):
        with cols[i]:
            st.image(
                image_for(category),
                use_container_width=True,
            )
            if st.button(
                category,
                key=f"cat_{category}",
                use_container_width=True,
            ):
                st.session_state.page = "Find Food"
                st.session_state.search_query = category
                st.rerun()

    st.subheader("Popular around your location")

    result = search(
        "Meal",
        top_k=6,
        min_rating=4.0,
        max_distance=20,
    )

    if "error" in result:
        st.error(result["error"])
    elif not result.get("results"):
        st.info(
            "No matching restaurants were returned."
        )
    else:
        for i, item in enumerate(result["results"]):
            render_card(item, f"home_{i}")


# ============================================================
# SEARCH
# ============================================================

elif st.session_state.page == "Find Food":
    st.title("Find Food")

    query = st.text_input(
        "What are you looking for?",
        value=st.session_state.get(
            "search_query",
            "Meal",
        ),
    )

    col1, col2 = st.columns(2)

    with col1:
        min_rating = st.slider(
            "Minimum rating",
            0.0,
            5.0,
            4.0,
            0.1,
        )

    with col2:
        max_distance = st.slider(
            "Maximum distance (km)",
            1,
            50,
            20,
        )

    if st.button(
        "Find restaurants",
        type="primary",
        use_container_width=True,
    ):
        result = search(
            query,
            top_k=10,
            min_rating=min_rating,
            max_distance=max_distance,
        )

        if "error" in result:
            st.error(result["error"])
        elif not result.get("results"):
            st.warning(
                "No restaurants matched those conditions."
            )
        else:
            st.success(
                f"Found {len(result['results'])} restaurant option(s)."
            )

    for i, item in enumerate(
        st.session_state.last_results
    ):
        render_card(item, f"search_{i}")


# ============================================================
# FOR YOU
# ============================================================

elif st.session_state.page == "For You":
    st.title("For You")

    st.write(
        "Choose a context and the project's recommendation "
        "pipeline ranks the available restaurants."
    )

    context = st.selectbox(
        "Choose a context",
        ["Meal", "Snack", "Drinks", "Buffet"],
    )

    minimum = st.slider(
        "Minimum rating",
        0.0,
        5.0,
        4.0,
        0.1,
    )

    if st.button(
        "Show recommendations",
        type="primary",
        use_container_width=True,
    ):
        result = search(
            context,
            top_k=8,
            min_rating=minimum,
            max_distance=20,
        )

        if "error" in result:
            st.error(result["error"])
        elif not result.get("results"):
            st.info(
                "No matching restaurants were returned."
            )
        else:
            for i, item in enumerate(
                result["results"]
            ):
                render_card(
                    item,
                    f"foryou_{i}",
                )


# ============================================================
# RESTAURANT DETAILS
# ============================================================

elif st.session_state.page == "Restaurant":
    if st.button("← Back"):
        st.session_state.page = "Find Food"
        st.rerun()

    item = st.session_state.selected

    if not item:
        st.info("Select a restaurant first.")
    else:
        name = item.get(
            "restaurant_name",
            "Restaurant",
        )

        st.title(display_name(item))

        st.caption(
            "Restaurant identity is represented by a "
            "coordinate-based proxy from the public dataset."
        )

        col1, col2 = st.columns([1.1, 2.9])

        with col1:
            st.image(
                image_for(item.get("order_type")),
                use_container_width=True,
            )

        with col2:
            st.write(
                f"**City:** {item.get('city', '—')}"
            )
            st.write(
                f"**Order type:** {item.get('order_type', '—')}"
            )
            st.write(
                f"**Rating:** {float(item.get('rating', 0)):.2f}"
            )

            if item.get("customer_distance_km") is not None:
                st.write(
                    f"**Distance:** "
                    f"{float(item['customer_distance_km']):.2f} km"
                )

            if item.get("predicted_eta_min") is not None:
                st.write(
                    f"**Estimated delivery:** "
                    f"{float(item['predicted_eta_min']):.0f} min"
                )

        st.markdown(
            """
            <div class="note">
                The public source dataset does not contain reliable
                item-level menus or prices, so this demo does not invent them.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "🛒 Add to cart",
            type="primary",
            use_container_width=True,
        ):
            add_to_cart(item)
            st.rerun()

        details = api_get(
            f"/restaurant/{quote(str(name), safe='')}"
        )

        if (
            "error" not in details
            and details.get("similar")
        ):
            st.subheader("Similar available restaurants")

            for i, similar in enumerate(
                details["similar"]
            ):
                if (
                    similar.get("restaurant_name")
                    != name
                ):
                    render_card(
                        similar,
                        f"similar_{i}",
                    )


# ============================================================
# CART
# ============================================================

elif st.session_state.page == "Cart":
    st.title("🛒 Your Cart")

    current_user = st.session_state.get("current_user")
    if current_user:
        st.caption(
            f"Cart for {current_user['name']} · User ID: {current_user['user_id']}"
        )
    else:
        st.caption("Guest cart")

    if st.session_state.get("cart_notice"):
        st.success(
            st.session_state.cart_notice
        )
        st.session_state.cart_notice = None

    if not isinstance(st.session_state.get("cart"), list):
        st.session_state.cart = list(st.session_state.get("cart", []) or [])

    cart_count = len(st.session_state.cart)

    if cart_count:
        st.caption(
            f"{cart_count} restaurant selection(s) "
            f"in your cart."
        )

    if not cart_count:
        st.info(
            "Your cart is empty. Add a restaurant "
            "from Home, Find Food or For You."
        )

        if st.button(
            "Continue shopping",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.page = "Find Food"
            st.rerun()

    else:
        for i, item in enumerate(
            st.session_state.cart
        ):
            with st.container(border=True):
                st.markdown(
                    f"### {display_name(item)}"
                )
                st.caption(
                    f"{item.get('city', '—')} · "
                    f"{item.get('order_type', '—')}"
                )

                info = []

                if item.get("rating") is not None:
                    info.append(
                        f"⭐ {float(item['rating']):.2f}"
                    )

                if item.get("customer_distance_km") is not None:
                    info.append(
                        f"📍 {float(item['customer_distance_km']):.2f} km"
                    )

                if item.get("predicted_eta_min") is not None:
                    info.append(
                        f"⏱️ {float(item['predicted_eta_min']):.0f} min"
                    )

                if info:
                    st.write(" · ".join(info))

                if st.button(
                    "Remove",
                    key=f"remove_{i}",
                ):
                    st.session_state.cart.pop(i)
                    save_user_state()
                    st.rerun()

        st.markdown(
            """
            <div class="note">
                This is a demo order flow. The source dataset does not
                contain item prices, menus or payment transactions.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "Continue shopping",
                use_container_width=True,
            ):
                st.session_state.page = "Find Food"
                st.rerun()

        with col2:
            if st.button(
                "Proceed to checkout",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.page = "Checkout"
                st.rerun()


# ============================================================
# CHECKOUT
# ============================================================

elif st.session_state.page == "Checkout":
    st.title("Checkout")

    if not st.session_state.cart:
        st.info(
            "Your cart is empty. Add a restaurant first."
        )
    else:
        st.subheader("Your selection")

        for item in st.session_state.cart:
            st.write(
                f"• {display_name(item)}"
            )

        address = st.text_area(
            "Delivery address",
            value=st.session_state.location["name"],
        )

        payment = st.radio(
            "Payment method",
            ["Cash on Delivery", "UPI", "Card"],
        )

        st.markdown(
            """
            <div class="note">
                Demo checkout only. No real payment is processed and
                no fabricated item price is displayed.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Place demo order",
            type="primary",
            use_container_width=True,
        ):
            now = datetime.now()

            for item in st.session_state.cart:
                st.session_state.orders.insert(
                    0,
                    {
                        **item,
                        "address": address,
                        "payment": payment,
                        "placed_at": now.strftime(
                            "%d %b %Y, %I:%M %p"
                        ),
                    },
                )

            st.session_state.cart = []
            save_user_state()

            st.session_state.page = "Orders"
            st.rerun()


# ============================================================
# ORDERS
# ============================================================

elif st.session_state.page == "Orders":
    st.title("Orders")

    if not st.session_state.orders:
        st.info("No orders yet.")
    else:
        for item in st.session_state.orders:
            with st.container(border=True):
                st.markdown(
                    f"### {display_name(item)}"
                )

                st.caption(
                    f"Placed {item.get('placed_at', '—')} · "
                    f"{item.get('payment', '—')}"
                )

                if item.get(
                    "predicted_eta_min"
                ) is not None:
                    st.metric(
                        "Estimated delivery",
                        f"{float(item['predicted_eta_min']):.0f} min",
                    )

                st.write(
                    f"Delivery address: {item.get('address', '—')}"
                )

                st.write(
                    "✅ Order placed → "
                    "✅ Restaurant confirmed → "
                    "🚴 Delivery in progress"
                )

                st.caption(
                    "Tracking is a demo state; live courier telemetry "
                    "is not part of the source dataset."
                )


# ============================================================
# ACCOUNT
# ============================================================

elif st.session_state.page == "Account":
    st.title("Account")

    if st.session_state.current_user:
        user = st.session_state.current_user

        st.success(
            f"Signed in as {user['name']}"
        )

        st.write(
            f"**User ID:** {user['user_id']}"
        )
        st.write(
            f"**Mobile:** {user['mobile']}"
        )

        if st.button(
            "Logout",
            type="secondary",
        ):
            logout()
    else:
        st.info(
            "You are browsing as a guest."
        )

        if st.button(
            "Login / Create account",
            type="primary",
        ):
            st.session_state.page = "Auth"
            st.rerun()

    st.divider()

    st.write(
        f"**Delivery location:** "
        f"{st.session_state.location['name']}"
    )

    st.markdown(
        """
        <div class="note">
            Restaurant identities are coordinate-based proxies generated
            from the public delivery dataset. The customer interface does
            not invent menus, prices or live delivery telemetry.
        </div>
        """,
        unsafe_allow_html=True,
    )
