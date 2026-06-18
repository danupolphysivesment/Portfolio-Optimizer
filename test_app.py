"""Headless end-to-end checks for the Streamlit UI via AppTest.

Exercises every workbench and every optimization method on deterministic
synthetic data, asserting the app never raises. Run: python test_app.py
"""
from streamlit.testing.v1 import AppTest
from portlab import optimizers as opt

SYNTH = "Synthetic (offline demo)"


def base_app():
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    at.selectbox(key="cfg_source").set_value(SYNTH).run()
    return at


def check(at, label):
    if at.exception:
        print(f"  ✗ {label}: {at.exception[0].value}")
        return False
    print(f"  ✓ {label}")
    return True


def main():
    ok = True

    # --- Optimizer: every method ---------------------------------------
    print("Optimizer — all methods:")
    for method in opt.ALL_METHODS:
        at = base_app()
        at.selectbox(key="opt_method").set_value(method).run()
        ok &= check(at, method)

    # --- Other workbenches ---------------------------------------------
    print("Other workbenches:")
    for sec in ["🧪 Backtester", "📚 Asset Universe", "📖 Methods"]:
        at = base_app()
        at.radio(key="cfg_section").set_value(sec).run()
        ok &= check(at, sec)

    # --- Backtester: walk-forward mode ---------------------------------
    print("Backtester — walk-forward:")
    at = base_app()
    at.radio(key="cfg_section").set_value("🧪 Backtester").run()
    # The mode radio is the first radio on the Backtester page after cfg_section.
    mode_radios = [r for r in at.radio if r.key != "cfg_section"]
    if mode_radios:
        mode_radios[0].set_value("Walk-forward optimization").run()
        ok &= check(at, "walk-forward run")
    else:
        print("  ! mode radio not found")

    print("\nRESULT:", "ALL PASS ✅" if ok else "FAILURES ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
