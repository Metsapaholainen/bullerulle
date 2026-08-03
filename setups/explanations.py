"""Human-readable explanations of what each setup means when it's found --
shown in the app next to scan matches so a match isn't just a row of numbers.
"""

PATTERN_EXPLANATIONS = {
    "breakout": (
        "Qullamaggie Breakout. A stock already made a big move, then paused to consolidate for a "
        "few weeks in a tight range while 'surfing' its 10/20-day moving average -- a sign strong "
        "hands are holding, not dumping. A breakout to new highs on above-average volume means "
        "demand is stepping back in and the stock is resuming its prior trend. This is a "
        "momentum-continuation trade, not a reversal. Three optional, stricter checks are available "
        "but off by default (Designer tab): the 10/20-day EMAs actually sloping upward (not just "
        "sat above by price), genuinely higher lows in the base (not just a tight range), and the "
        "prior move being a real net advance rather than just a wide/choppy range -- each cuts "
        "match volume noticeably when turned on, so try them individually rather than all at once."
    ),
    "episodic_pivot": (
        "Episodic Pivot. A sudden, large gap up (often on an earnings surprise, guidance raise, "
        "or major news) on unusually heavy volume, in a stock that had been quiet or overlooked "
        "beforehand. Big growth numbers (EPS/revenue) are required whenever fundamentals data is "
        "available for the symbol, not just a scoring bonus, and a genuine earnings beat vs. "
        "analyst estimates adds further to the score when that data is available. The gap reflects "
        "a sharp, sudden re-rating of the business by the market -- these can be the start of a "
        "large new trend, but are also more news-driven and volatile than a technical breakout, so "
        "risk control matters even more here. Note: this scanner only ever sees end-of-day bars, so "
        "it can't verify the stricter 'gap confirmed in the first 5-30 minutes' intraday timing -- "
        "that confirmation is left to the trader."
    ),
    "parabolic_short": (
        "Parabolic Short (extension/reversal). A stock has run up so far, so fast, that it's "
        "trading many multiples of its normal daily range above its short-term moving average. "
        "This is a bet that the move has become unsustainable and is due for a sharp mean-reversion "
        "or blow-off-top reversal -- the counter-trend, short-side setup in the playbook. Higher "
        "risk than the long setups: you're betting against momentum, not with it."
    ),
    "cup_with_handle": (
        "Cup with Handle (O'Neil). The single most famous pattern in 'How to Make Money in Stocks'. "
        "After an uptrend, the stock corrects gradually and rounds out at the bottom like the base "
        "of a cup (not a sharp V), climbs back toward its old high, then forms a small 'handle' -- "
        "a shallow, low-volume pullback in the upper half of the cup that shakes out the last "
        "impatient holders. A breakout above the old high (and the handle) on a surge of volume "
        "signals that institutions are accumulating again and a new leg up is starting."
    ),
    "double_bottom": (
        "Double Bottom / 'W' (O'Neil). The stock makes a low, rallies, then comes back down to a "
        "similar (often slightly lower) low before rallying again -- tracing a 'W' shape. The "
        "second low undercutting the first is a classic shakeout of weak holders; the fact that "
        "sellers couldn't push price meaningfully lower the second time shows selling pressure is "
        "drying up. A breakout above the peak between the two lows, on volume, is the buy signal."
    ),
    "flat_base": (
        "Flat Base (O'Neil). A tight, boring, sideways consolidation -- no more than roughly "
        "10-15% from top to bottom -- lasting at least five weeks, usually after an existing "
        "uptrend or an earlier base. The tightness itself is the signal: nobody's in a hurry to "
        "sell, which means supply has dried up. A breakout above the top of the range on volume "
        "suggests the stock is ready to resume its advance."
    ),
    "ascending_base": (
        "Ascending Base (O'Neil). A choppier, staircase-shaped base: a series of pullbacks (often "
        "three), each with a higher low and a higher high than the last, each pullback shallower "
        "than a typical 20%+ correction. It looks messier than a cup or flat base, but the "
        "ascending structure shows buyers keep stepping in at higher prices -- persistent, "
        "building demand rather than a clean rest. Breakout above the most recent high on volume "
        "confirms it."
    ),
    "high_tight_flag": (
        "High Tight Flag (O'Neil). The rarest and most explosive pattern in the book: the stock "
        "rockets up 100%+ (sometimes 120%+) in just 4-8 weeks, then pauses in a remarkably tight "
        "flag for 3-5 weeks with a pullback of only ~10-25% -- astonishingly shallow given how far "
        "and fast it just ran. That tightness after such an extreme move suggests almost nobody is "
        "taking profits. When it works, it can mean a stock is just getting started on a much "
        "bigger move. It also has one of the highest failure rates of any pattern here precisely "
        "because the setup itself is so extreme -- size positions accordingly."
    ),
}
