from pathlib import Path

TURNBASED_DIR = Path("/app/assets/presets/turnbased")


def main() -> None:
    """Make the default English text preset useful for smoke testing.

    The upstream ``english_call`` preset is voice-cloning oriented and can echo
    very short text prompts.  This does not affect the audio/omni duplex pages.
    """

    (TURNBASED_DIR / "english_call.yaml").write_text(
        """id: english_call
order: 1
name: "Helpful Chat"
description: "Plain English assistant for smoke-testing MiniCPM-o 4.5"
system_content:
  - type: text
    text: |
      You are MiniCPM-o 4.5 running on Modal. Be helpful, direct, and natural.
      Answer normally and do not repeat the user's words unless asked.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
