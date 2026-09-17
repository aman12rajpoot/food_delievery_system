import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="QuickBite",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.18);
    }

    .brand {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .tagline {
        color: #777;
        font-size: 14px;
        margin-bottom: 22px;
    }

    .hero {
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(128,128,128,0.18);
        margin-bottom: 25px;
    }

    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 36px;
    }

    .hero p {
        margin: 0;
        color: #777;
        font-size: 16px;
    }

    .feature-card {
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.18);
        min-height: 145px;
        margin-bottom: 15px;
    }

    .feature-card h3 {
        margin-top: 0;
        margin-bottom: 8px;
    }

    .feature-card p {
        color: #777;
        font-size: 14px;
    }

    .restaurant-card {
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.18);
        margin-bottom: 14px;
    }

    .restaurant-name {
        font-size: 20px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .restaurant-meta {
        color: #777;
        font-size: 13px;
        margin-bottom: 12px;
    }

    .restaurant-stats {
        font-size: 14px;
    }

    .recommended {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .big-result {
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(128,128,128,0.18);
        text-align: center;
        margin: 20px 0;
    }

    .big-number {
        font-size: 48px;
        font-weight: 800;
    }

    .big-label {
        color: #777;
        font-size: 15px;
    }

    .info-card {
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.18);
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(endpoint, params=None):
    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Backend service is unavailable. Please start FastAPI."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except requests.exceptions.HTTPError as exc:
        return {"error": f"Request failed: HTTP {exc.response.status_code}"}
    except Exception as exc:
        return {"error": str(exc)}


def api_post(endpoint, payload):
    try:
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Backend service is unavailable. Please start FastAPI."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except requests.exceptions.HTTPError as exc:
        return {"error": f"Request failed: HTTP {exc.response.status_code}"}
    except Exception as exc:
        return {"error": str(exc)}


def number(value, decimals=1):
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def day_name(day):
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    return days[int(day)]


# -------------------- SIDEBAR --------------------

st.sidebar.markdown(
    """
    <div class="brand">🍔 QuickBite</div>
    <div class="tagline">Smart food delivery</div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Menu",
    [
        "Home",
        "Find Food",
        "Delivery Time",
        "Demand Outlook",
        "Delivery Operations",
        "Performance Insights",
    ],
)

health = api_get("/health")

if "error" not in health and health.get("status") == "ok":
    st.sidebar.success("● Service online")
else:
    st.sidebar.error("● Service unavailable")


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <h1>Good food, smarter delivery. 🍔</h1>
            <p>
                Discover restaurants, estimate delivery time and understand
                food delivery demand from one place.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("What would you like to do?")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🍽️ Find Food</h3>
                <p>
                    Discover nearby restaurants based on food choice,
                    ratings, distance and delivery time.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>⏱️ Check Delivery Time</h3>
                <p>
                    Get an estimated delivery time before placing an order.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📈 Demand Outlook</h3>
                <p>
                    Understand expected order volume for the upcoming period.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader("Platform status")

    if "error" not in health and health.get("status") == "ok":
        a, b, c, d = st.columns(4)

        a.metric("Restaurant discovery", "Ready")
        b.metric("Delivery estimates", "Ready")
        c.metric("Demand outlook", "Ready")
        d.metric("Delivery operations", "Ready")
    else:
        st.warning("Please start the backend service to use the platform.")


# ============================================================
# FIND FOOD
# ============================================================

elif page == "Find Food":

    st.markdown(
        """
        <div class="hero">
            <h1>Find your next meal 🍽️</h1>
            <p>
                Tell us what you want and we'll find options that balance
                quality, distance and delivery time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        query = st.text_input(
            "What are you craving?",
            value="Meal",
            placeholder="Meal, Snack, Drinks...",
        )

        min_rating = st.slider(
            "Minimum rating",
            0.0,
            5.0,
            4.0,
            0.1,
        )

        max_distance = st.slider(
            "Maximum distance",
            1,
            50,
            20,
            1,
        )

    with right:
        number_of_places = st.slider(
            "Number of places",
            1,
            10,
            5,
            1,
        )

        use_location = st.checkbox(
            "Use my location",
            value=True,
        )

        if use_location:
            user_lat = st.number_input(
                "Latitude",
                value=23.3538,
                format="%.4f",
            )

            user_lon = st.number_input(
                "Longitude",
                value=85.3270,
                format="%.4f",
            )
        else:
            user_lat = None
            user_lon = None

    with st.expander("More preferences"):
        p1, p2, p3 = st.columns(3)

        with p1:
            pickup_hour = st.slider(
                "Order time",
                0,
                23,
                19,
            )

        with p2:
            day_of_week = st.selectbox(
                "Day",
                list(range(7)),
                index=4,
                format_func=day_name,
            )

        with p3:
            busy_period = st.checkbox(
                "Busy period",
                value=True,
            )

    if st.button(
        "🔎 Find restaurants",
        type="primary",
        use_container_width=True,
    ):

        params = {
            "query": query.strip() or "Meal",
            "top_k": number_of_places,
            "user_lat": user_lat,
            "user_lon": user_lon,
            "min_rating": min_rating,
            "max_distance_km": max_distance,
            "pickup_hour": pickup_hour,
            "day_of_week": day_of_week,
            "is_peak": int(busy_period),
            "vehicle_condition": 1,
            "multiple_deliveries": 0,
            "delivery_person_age": 30,
        }

        with st.spinner("Finding restaurants..."):
            result = api_get("/recommend", params=params)

        if "error" in result:
            st.error(result["error"])
        else:
            results = result.get("results", [])

            if not results:
                st.warning(
                    "No restaurants matched your preferences. "
                    "Try increasing the distance or lowering the minimum rating."
                )
            else:
                st.success(f"We found {len(results)} option(s) for you.")

                for index, restaurant in enumerate(results, start=1):

                    restaurant_name = restaurant.get(
                        "restaurant_name",
                        f"Nearby Restaurant {index}",
                    )

                    rating = restaurant.get("rating", 0)
                    distance = restaurant.get("customer_distance_km", 0)
                    eta = restaurant.get("predicted_eta_min", 0)
                    orders = restaurant.get("orders", 0)
                    city = restaurant.get("city", "")
                    order_type = restaurant.get("order_type", query)

                    badge = ""

                    if index == 1:
                        badge = """
                        <div class="recommended">
                            ✨ Recommended for you
                        </div>
                        """

                    st.markdown(
                        f"""
                        <div class="restaurant-card">
                            {badge}

                            <div class="restaurant-name">
                                {restaurant_name}
                            </div>

                            <div class="restaurant-meta">
                                {city} · {order_type}
                            </div>

                            <div class="restaurant-stats">
                                ⭐ <b>{number(rating, 2)}</b>
                                &nbsp;&nbsp; • &nbsp;&nbsp;
                                🚴 <b>{number(distance, 2)} km</b>
                                &nbsp;&nbsp; • &nbsp;&nbsp;
                                ⏱️ <b>{number(eta, 0)} min</b>
                                &nbsp;&nbsp; • &nbsp;&nbsp;
                                👥 <b>{int(orders)}</b> orders
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# DELIVERY TIME
# ============================================================

elif page == "Delivery Time":

    st.markdown(
        """
        <div class="hero">
            <h1>Delivery time estimate ⏱️</h1>
            <p>
                Get an estimated delivery time based on your order conditions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        distance = st.number_input(
            "Restaurant distance (km)",
            min_value=0.1,
            max_value=100.0,
            value=7.5,
            step=0.1,
        )

        rating = st.number_input(
            "Delivery partner rating",
            min_value=0.0,
            max_value=5.0,
            value=4.7,
            step=0.1,
        )

        age = st.number_input(
            "Delivery partner age",
            min_value=18,
            max_value=70,
            value=28,
        )

    with right:
        vehicle_condition = st.slider(
            "Vehicle condition",
            0,
            5,
            1,
        )

        other_deliveries = st.number_input(
            "Other deliveries on route",
            0,
            5,
            0,
        )

        order_hour = st.slider(
            "Order hour",
            0,
            23,
            19,
        )

        order_day = st.selectbox(
            "Order day",
            list(range(7)),
            index=5,
            format_func=day_name,
        )

        busy_period = st.checkbox(
            "Busy period",
            value=True,
        )

    if st.button(
        "⏱️ Estimate delivery time",
        type="primary",
        use_container_width=True,
    ):

        payload = {
            "distance_km": distance,
            "age": age,
            "rating": rating,
            "vehicle_condition": vehicle_condition,
            "multiple_deliveries": other_deliveries,
            "pickup_hour": order_hour,
            "day_of_week": order_day,
            "is_peak": int(busy_period),
            "prep_time_min": 12,
        }

        with st.spinner("Calculating your estimate..."):
            result = api_post("/predict-eta", payload)

        if "error" in result:
            st.error(result["error"])
        else:
            eta = result.get(
                "predicted_delivery_minutes",
                0,
            )

            st.markdown(
                f"""
                <div class="big-result">
                    <div class="big-label">
                        Estimated delivery time
                    </div>

                    <div class="big-number">
                        {number(eta, 0)} min
                    </div>

                    <div class="big-label">
                        Based on current order conditions
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            a, b, c = st.columns(3)

            a.metric(
                "Distance",
                f"{number(distance, 1)} km",
            )

            b.metric(
                "Order period",
                "Busy" if busy_period else "Normal",
            )

            c.metric(
                "Other deliveries",
                int(other_deliveries),
            )

            st.info(
                "This is an estimated delivery time based on historical "
                "delivery patterns. Actual delivery time may vary."
            )


# ============================================================
# DEMAND OUTLOOK
# ============================================================

elif page == "Demand Outlook":

    st.markdown(
        """
        <div class="hero">
            <h1>Demand outlook 📈</h1>
            <p>
                Understand expected order volume so food delivery operations
                can prepare ahead.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "📈 Refresh demand outlook",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner("Preparing demand outlook..."):
            result = api_get("/forecast-demand")

        if "error" in result:
            st.error(result["error"])
        else:
            predicted = result.get(
                "predicted_demand",
                0,
            )

            latest = result.get(
                "latest_observed_orders",
                0,
            )

            difference = predicted - latest

            a, b, c = st.columns(3)

            a.metric(
                "Expected orders",
                number(predicted, 0),
            )

            b.metric(
                "Latest observed",
                number(latest, 0),
            )

            c.metric(
                "Change",
                f"{difference:+.0f}",
            )

            st.markdown("---")

            if predicted > latest:
                st.info(
                    f"Expected demand is about {number(difference, 0)} orders "
                    "higher than the latest observed period."
                )
            elif predicted < latest:
                st.info(
                    f"Expected demand is about {number(abs(difference), 0)} orders "
                    "lower than the latest observed period."
                )
            else:
                st.info(
                    "Expected demand is close to the latest observed level."
                )

            st.caption(
                f"Forecast period: {result.get('forecast_for', 'Latest available period')}"
            )

            st.caption(
                "Forecast is based on the historical data available to the platform."
            )


# ============================================================
# DELIVERY OPERATIONS
# ============================================================

elif page == "Delivery Operations":

    st.markdown(
        """
        <div class="hero">
            <h1>Delivery operations 🚚</h1>
            <p>
                Plan delivery assignments while keeping workloads within
                available capacity.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "This demonstration uses a small delivery-partner pool to show "
        "how delivery assignments can be optimized."
    )

    if st.button(
        "🚚 Optimize delivery plan",
        type="primary",
        use_container_width=True,
    ):

        payload = {
            "orders": [
                {"distance_km": 3.5, "delivery_time": 22},
                {"distance_km": 7.1, "delivery_time": 30},
                {"distance_km": 12.0, "delivery_time": 38},
                {"distance_km": 5.3, "delivery_time": 27},
            ],
            "partners": [
                {"partner_id": "P1", "base_distance": 2.0, "capacity": 2},
                {"partner_id": "P2", "base_distance": 5.0, "capacity": 2},
                {"partner_id": "P3", "base_distance": 9.0, "capacity": 2},
            ],
        }

        with st.spinner("Preparing the delivery plan..."):
            result = api_post(
                "/optimize",
                payload,
            )

        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Delivery plan prepared successfully.")

            assigned = result.get(
                "assigned_orders",
                result.get("total_orders", 0),
            )

            st.metric(
                "Orders assigned",
                assigned,
            )

            assignments = result.get(
                "assignments",
                [],
            )

            if assignments:

                st.subheader("Delivery plan")

                clean_rows = []

                for index, item in enumerate(assignments, start=1):

                    clean_rows.append(
                        {
                            "Order": item.get(
                                "order_id",
                                item.get(
                                    "order",
                                    f"Order {index}",
                                ),
                            ),
                            "Delivery partner": item.get(
                                "partner_id",
                                item.get(
                                    "assigned_partner",
                                    "Assigned",
                                ),
                            ),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(clean_rows),
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# PERFORMANCE INSIGHTS
# ============================================================

elif page == "Performance Insights":

    st.markdown(
        """
        <div class="hero">
            <h1>Performance insights 🧪</h1>
            <p>
                Compare standard restaurant discovery with a
                context-aware approach using an offline experiment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "🧪 View experiment results",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner("Loading performance results..."):
            result = api_get("/ab-test")

        if "error" in result:
            st.error(result["error"])
        else:

            control = float(
                result.get("control_conversion", 0)
            )

            treatment = float(
                result.get("treatment_conversion", 0)
            )

            conversion_lift = float(
                result.get("relative_conversion_lift", 0)
            )

            revenue_lift = float(
                result.get("relative_revenue_per_user_lift", 0)
            )

            p_value = result.get(
                "p_value",
                None,
            )

            a, b, c = st.columns(3)

            a.metric(
                "Standard discovery",
                f"{control * 100:.2f}%",
            )

            b.metric(
                "Context-aware discovery",
                f"{treatment * 100:.2f}%",
            )

            c.metric(
                "Conversion change",
                f"{conversion_lift * 100:.2f}%",
            )

            st.markdown("---")

            a, b = st.columns(2)

            with a:

                st.markdown(
                    """
                    <div class="info-card">
                        <h3>💰 Revenue per customer</h3>
                        <p>
                            Change observed under the simulated experiment.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.metric(
                    "Change",
                    f"{revenue_lift * 100:.2f}%",
                )

            with b:

                st.markdown(
                    """
                    <div class="info-card">
                        <h3>📊 Statistical result</h3>
                        <p>
                            Statistical evidence for the simulated
                            conversion difference.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if p_value is not None:
                    st.metric(
                        "P-value",
                        f"{float(p_value):.5f}",
                    )

            st.markdown("---")

            st.subheader("Experiment context")

            st.write(
                "The comparison evaluates standard restaurant discovery "
                "against a context-aware approach that considers food "
                "relevance, restaurant quality, popularity, distance and "
                "expected delivery time."
            )

            st.warning(
                "This is an offline simulated experiment. The results "
                "demonstrate experimentation methodology and should not be "
                "interpreted as measured production impact."
            )
