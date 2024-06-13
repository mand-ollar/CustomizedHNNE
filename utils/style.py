"""Style the printing text."""


def style_it(
    color="default",
    style="default",
    text="",
):
    """Colorize the text."""

    color_table = {
        "green": "\033[32m",
        "red": "\033[31m",
        "blue": "\033[34m",
        "yellow": "\033[33m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "black": "\033[30m",
        "default": "\033[39m",
    }

    style_table = {
        "bold": "\033[1m",
        "underline": "\033[4m",
        "blink": "\033[5m",
        "reverse": "\033[7m",
        "invisible": "\033[8m",
        "default": "\033[0m",
    }

    default = "\033[0m"

    # If the input is not valid
    if color not in color_table:
        raise ValueError(f"Invalid color: {color}")

    if style not in style_table:
        raise ValueError(f"Invalid style: {style}")

    return f"{color_table[color]}{style_table[style]}{text}{default}"
