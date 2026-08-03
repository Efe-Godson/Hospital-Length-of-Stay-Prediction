"""Small display-formatting helpers used by the UI layer."""


def format_days(value: float) -> str:
    return f"{value:.1f} days"


def format_signed(value: float, decimals: int = 2) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}"


def yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"
