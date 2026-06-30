"""Single source of truth for vocabularies, frozen categories, and thresholds.

Both the synthetic data generator and the backend import from here so the two can
never drift. The categorical vocabularies are *frozen*: a value outside these
lists is encoded as all-zeros (handle_unknown="ignore") rather than growing the
vector dimension and breaking the Ahnlich store.
"""

from __future__ import annotations

# --- Frozen categorical vocabularies -------------------------------------------------
PAYMENT_TYPES: list[str] = [
    "credit_card",
    "debit_card",
    "wire",
    "crypto",
    "paypal",
    "gift_card",
    # Nigerian payment channels (CBN demonstration domain).
    "bank_transfer",  # NIP / NIBSS instant transfer
    "ussd",
    "pos",
    "mobile_money",
]

PRODUCT_CATEGORIES: list[str] = [
    "electronics",
    "apparel",
    "grocery",
    "digital_goods",
    "jewelry",
    "travel",
    "home",
    "toys",
    # Nigerian-relevant categories.
    "airtime",
    "fx_remittance",
]

# Frozen billing-country vocabulary. "US" is the home country; the rest are used by
# the account-takeover / structuring typologies. Unseen countries encode to zeros.
BILLING_COUNTRIES: list[str] = [
    "US",
    "CA",
    "GB",
    "DE",
    "FR",
    "BR",
    "IN",
    "NG",
    "RU",
    "CN",
]

HOME_COUNTRY = "US"
NIGERIA_COUNTRY = "NG"
# Both regions are first-class/legitimate in this CBN demonstration, so neither is
# inherently "foreign". The geo heuristic flags billing outside this set.
DOMESTIC_COUNTRIES: list[str] = ["US", "NG"]
FOREIGN_COUNTRIES: list[str] = [c for c in BILLING_COUNTRIES if c != HOME_COUNTRY]

# Region-appropriate channel/category pools so legitimate traffic differs by region
# and fraud doesn't separate trivially on country alone.
US_PAYMENT_TYPES: list[str] = ["credit_card", "debit_card", "paypal", "gift_card", "wire"]
NG_PAYMENT_TYPES: list[str] = ["bank_transfer", "ussd", "pos", "mobile_money", "debit_card"]
US_PRODUCT_CATEGORIES: list[str] = ["electronics", "apparel", "grocery", "digital_goods", "travel", "home", "toys", "jewelry"]
NG_PRODUCT_CATEGORIES: list[str] = ["airtime", "grocery", "electronics", "apparel", "fx_remittance", "digital_goods"]

# --- Fraud typologies (ground truth only; never enters the feature vector) ------------
FRAUD_SCENARIOS: list[str] = [
    "card_testing",
    "structuring",
    "whale_anomaly",
    "account_takeover",
    "odd_hour_burst",
    # CBN demonstration typologies.
    "fx_structuring",  # amounts just under the FX/BTA threshold via wire/crypto/transfer
    "crypto_fx_evasion",  # large naira→crypto buys to bypass FX controls
    "pos_agent_cashout",  # burst of mid-value POS withdrawals via an agent
    "ussd_micro_burst",  # burst of micro USSD transfers (mobile card-testing analogue)
    "sim_swap_takeover",  # odd-hour high-value transfer after a SIM-swap
]

# --- Feature column groupings ---------------------------------------------------------
NUMERICAL_FEATURES: list[str] = ["order_price", "billing_latitude", "billing_longitude"]
CATEGORICAL_FEATURES: list[str] = ["payment_type", "product_category", "billing_country"]
TIMESTAMP_FEATURE = "event_timestamp"
TEXT_FEATURE = "text_blob"  # composed at transform time from job + merchant

# Frozen categories aligned to CATEGORICAL_FEATURES order, for the OneHotEncoder.
FROZEN_CATEGORIES: list[list[str]] = [
    PAYMENT_TYPES,
    PRODUCT_CATEGORIES,
    BILLING_COUNTRIES,
]

# --- Reason-derivation thresholds (business knobs; one place to change) ---------------
MICRO_AMOUNT = 5.0  # below this looks like card-testing
STRUCTURING_LOW = 9000.0
STRUCTURING_HIGH = 10000.0
HIGH_AMOUNT = 50000.0  # whale-scale
LARGE_TRANSFER = 5000.0  # large wire/crypto
OFF_HOURS_END = 5  # hours <= this are "off hours"
