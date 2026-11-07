"""The coin universe the data pipeline can pick from, by category (same lists as crypto-signal-prediction's
data/defaults.py COINS_BY_CATEGORY at the time of porting). A category is only a convenient way to name coins:
``SourceSettings.coins`` always wins when it is non-empty."""

COINS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "Large_Caps": (
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "DOGEUSDT",
        "TRXUSDT", "LINKUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "TONUSDT", "ICPUSDT", "ETCUSDT", "FILUSDT",
        "APTUSDT", "XLMUSDT", "INJUSDT", "NEARUSDT", "XMRUSDT", "OPUSDT",
    ),
    "DeFi": (
        "AAVEUSDT", "COMPUSDT", "SNXUSDT", "SUSHIUSDT", "LDOUSDT", "RUNEUSDT", "FXSUSDT", "PENDLEUSDT",
        "DYDXUSDT",
    ),
    "AI_BigData": (
        "TAOUSDT", "FETUSDT", "RNDRUSDT", "AKTUSDT", "NMRUSDT", "GRTUSDT", "PHBUSDT", "AIUSDT",
    ),
    "Gaming_Metaverse": (
        "SANDUSDT", "MANAUSDT", "IMXUSDT", "GALAUSDT", "AXSUSDT", "ENJUSDT", "YGGUSDT", "MAGICUSDT",
        "PORTALUSDT", "PIXELUSDT",
    ),
    "Meme": (
        "1000SHIBUSDT", "1000PEPEUSDT",
    ),
    "Layer1_Layer2": (
        "SUIUSDT", "SEIUSDT", "ALGOUSDT", "EGLDUSDT", "ARBUSDT", "STRKUSDT", "ZETAUSDT", "KASUSDT",
        "TIAUSDT", "HBARUSDT", "EOSUSDT", "XTZUSDT",
    ),
    "Various_Services": (
        "VETUSDT", "THETAUSDT", "QNTUSDT", "CHZUSDT", "HOTUSDT", "BATUSDT", "ZILUSDT", "ONTUSDT", "IOTXUSDT",
        "ONEUSDT", "CELOUSDT", "CFXUSDT", "ANKRUSDT", "STORJUSDT", "DENTUSDT", "GASUSDT", "NEOUSDT",
        "DASHUSDT",
    ),
}

#: categories used when ``SourceSettings.categories`` is empty
DEFAULT_CATEGORIES: tuple[str, ...] = ("Large_Caps",)


def coins_of(categories: tuple[str, ...] = ()) -> list[str]:
    """Flat coin list of `categories` (empty = DEFAULT_CATEGORIES; ("all",) = every category); unknown names raise."""
    names = categories or DEFAULT_CATEGORIES
    if tuple(names) == ("all",):
        names = tuple(COINS_BY_CATEGORY)
    unknown = [c for c in names if c not in COINS_BY_CATEGORY]
    if unknown:
        raise ValueError(f"unknown coin categories {unknown}; known: {sorted(COINS_BY_CATEGORY)}")
    return [coin for name in names for coin in COINS_BY_CATEGORY[name]]
